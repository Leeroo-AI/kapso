from __future__ import annotations

import json
import math
import os
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


DAY_NS = 86_400_000_000_000
CAT_COLUMNS = [
    "driver_id",
    "driver_nationality",
    "constructor_id",
    "constructor_nationality",
    "last_status_id",
]
FEATURE_CACHE_KEY = "lane0_driver_position_temporal_rank_v3"


def _code(value) -> int:
    if pd.isna(value):
        return -1
    return int(zlib.crc32(str(value).encode("utf-8")) & 0x7FFFFFFF)


def _date_values(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).astype("int64").to_numpy()


def _numeric(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)


def _group_arrays(df: pd.DataFrame, key: str, columns: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    if df.empty:
        return {}
    ordered = df.sort_values([key, "date"], kind="stable")
    groups = {}
    for value, part in ordered.groupby(key, sort=False, observed=True):
        groups[int(value)] = (
            _date_values(part["date"]),
            part[columns].to_numpy(dtype=np.float64),
        )
    return groups


def _slice(
    history: tuple[np.ndarray, np.ndarray] | None,
    timestamp: int,
    days: int | None = None,
    last_n: int | None = None,
) -> np.ndarray:
    if history is None:
        return np.empty((0, 0), dtype=np.float64)
    dates, values = history
    end = int(np.searchsorted(dates, timestamp, side="right"))
    start = 0
    if days is not None:
        start = int(np.searchsorted(dates, timestamp - days * DAY_NS, side="left"))
    if last_n is not None:
        start = max(start, end - last_n)
    return values[start:end]


def _last_date(history: tuple[np.ndarray, np.ndarray] | None, timestamp: int) -> float:
    if history is None:
        return np.nan
    dates = history[0]
    end = int(np.searchsorted(dates, timestamp, side="right"))
    if end == 0:
        return np.nan
    return (timestamp - int(dates[end - 1])) / DAY_NS


def _finite_mean(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else np.nan


def _finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _finite_std(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.std()) if len(values) else np.nan


def _finite_min(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.min()) if len(values) else np.nan


def _finite_max(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(values.max()) if len(values) else np.nan


def _slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(values)
    if mask.sum() < 2:
        return np.nan
    y = values[mask]
    x = np.arange(len(values), dtype=np.float64)[mask]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return float(np.dot(x, y - y.mean()) / denom) if denom else 0.0


def _summary(target: dict[str, float], prefix: str, values: np.ndarray, extended: bool = False) -> None:
    if values.size == 0:
        target[f"{prefix}_count"] = 0.0
        for name in ("pos_mean", "pos_median", "pos_std", "norm_mean", "grid_mean", "points_mean", "laps_mean", "delta_mean"):
            target[f"{prefix}_{name}"] = np.nan
        if extended:
            for name in ("pos_min", "pos_max", "norm_std", "pos_slope"):
                target[f"{prefix}_{name}"] = np.nan
        for name in ("classified_rate", "finished_rate", "top3_rate", "top10_rate", "points_rate", "pace_mean", "time_gap_mean"):
            target[f"{prefix}_{name}"] = np.nan
        return
    target[f"{prefix}_count"] = float(len(values))
    target[f"{prefix}_pos_mean"] = _finite_mean(values[:, 0])
    target[f"{prefix}_pos_median"] = _finite_median(values[:, 0])
    target[f"{prefix}_pos_std"] = _finite_std(values[:, 0])
    target[f"{prefix}_norm_mean"] = _finite_mean(values[:, 1])
    target[f"{prefix}_grid_mean"] = _finite_mean(values[:, 2])
    target[f"{prefix}_points_mean"] = _finite_mean(values[:, 3])
    target[f"{prefix}_laps_mean"] = _finite_mean(values[:, 4])
    target[f"{prefix}_delta_mean"] = _finite_mean(values[:, 5])
    if values.shape[1] > 16:
        target[f"{prefix}_classified_rate"] = _finite_mean(values[:, 10])
        target[f"{prefix}_finished_rate"] = _finite_mean(values[:, 11])
        target[f"{prefix}_top3_rate"] = _finite_mean(values[:, 12])
        target[f"{prefix}_top10_rate"] = _finite_mean(values[:, 13])
        target[f"{prefix}_points_rate"] = _finite_mean(values[:, 14])
        target[f"{prefix}_pace_mean"] = _finite_mean(values[:, 15])
        target[f"{prefix}_time_gap_mean"] = _finite_mean(values[:, 16])
    else:
        for name in ("classified_rate", "finished_rate", "top3_rate", "top10_rate", "points_rate", "pace_mean", "time_gap_mean"):
            target[f"{prefix}_{name}"] = np.nan
    if extended:
        target[f"{prefix}_pos_min"] = _finite_min(values[:, 0])
        target[f"{prefix}_pos_max"] = _finite_max(values[:, 0])
        target[f"{prefix}_norm_std"] = _finite_std(values[:, 1])
        target[f"{prefix}_pos_slope"] = _slope(values[:, 0])


class FeatureBuilder:
    def __init__(self, db, labels: pd.DataFrame):
        self.labels = labels[["date", "driverId", "position"]].copy()
        self.labels["date"] = pd.to_datetime(self.labels["date"])
        self.tables = {name: table.df.copy() for name, table in db.table_dict.items()}
        self._prepare_entities()
        self._prepare_results()
        self._prepare_labels()
        self._prepare_standings()
        self._prepare_qualifying()
        self._prepare_races()

    def _prepare_entities(self) -> None:
        drivers = self.tables["drivers"].copy()
        self.driver_meta = {}
        for row in drivers.itertuples(index=False):
            self.driver_meta[int(row.driverId)] = (
                pd.Timestamp(row.dob).value if not pd.isna(row.dob) else None,
                _code(row.nationality),
            )
        constructors = self.tables["constructors"].copy()
        self.constructor_meta = {
            int(row.constructorId): _code(row.nationality)
            for row in constructors.itertuples(index=False)
        }

    def _prepare_results(self) -> None:
        results = self.tables["results"].copy()
        races = self.tables["races"][["raceId", "circuitId", "year", "round", "date"]].copy()
        fields = (
            results.groupby("raceId", observed=True)
            .agg(field_size=("driverId", "size"), constructor_count=("constructorId", "nunique"), max_laps=("laps", "max"))
            .reset_index()
        )
        results = results.merge(fields, on="raceId", how="left")
        results = results.merge(races[["raceId", "circuitId"]], on="raceId", how="left")
        denominator = np.maximum(_numeric(results["field_size"]) - 1.0, 1.0)
        position = _numeric(results["positionOrder"])
        grid = _numeric(results["grid"])
        grid = np.where(grid > 0, grid, _numeric(results["field_size"]))
        grid_norm = (grid - 1.0) / denominator
        position_norm = (position - 1.0) / denominator
        max_laps = np.maximum(_numeric(results["max_laps"]), 1.0)
        results["position_value"] = position
        results["position_norm"] = position_norm
        results["grid_norm"] = grid_norm
        results["points_value"] = _numeric(results["points"])
        results["laps_ratio"] = _numeric(results["laps"]) / max_laps
        results["finish_delta"] = position_norm - grid_norm
        results["constructor_value"] = _numeric(results["constructorId"])
        results["status_value"] = _numeric(results["statusId"])
        results["race_value"] = _numeric(results["raceId"])
        results["circuit_value"] = _numeric(results["circuitId"])
        results["classified_value"] = results["position"].notna().astype(np.float64)
        results["finished_value"] = (_numeric(results["statusId"]) == 1.0).astype(np.float64)
        results["top3_value"] = (position <= 3.0).astype(np.float64)
        results["top10_value"] = (position <= 10.0).astype(np.float64)
        results["points_flag"] = (_numeric(results["points"]) > 0.0).astype(np.float64)
        fastest_rank = _numeric(results["rank"])
        results["pace_rank_norm"] = (fastest_rank - 1.0) / denominator
        milliseconds = _numeric(results["milliseconds"])
        minimum_time = results.groupby("raceId", observed=True)["milliseconds"].transform("min").to_numpy(dtype=np.float64)
        results["time_gap_pct"] = 100.0 * (milliseconds - minimum_time) / np.maximum(minimum_time, 1.0)
        columns = [
            "position_value",
            "position_norm",
            "grid_norm",
            "points_value",
            "laps_ratio",
            "finish_delta",
            "constructor_value",
            "status_value",
            "race_value",
            "circuit_value",
            "classified_value",
            "finished_value",
            "top3_value",
            "top10_value",
            "points_flag",
            "pace_rank_norm",
            "time_gap_pct",
        ]
        self.driver_results = _group_arrays(results, "driverId", columns)
        self.constructor_results_history = _group_arrays(results, "constructorId", columns)
        self.results = results
        constructor_results = self.tables["constructor_results"].copy()
        constructor_results["result_points"] = _numeric(constructor_results["points"])
        self.constructor_points = _group_arrays(constructor_results, "constructorId", ["result_points"])

    def _prepare_labels(self) -> None:
        labels = self.labels.copy()
        labels["rank_pct"] = labels.groupby("date", observed=True)["position"].rank(pct=True)
        self.driver_labels = _group_arrays(labels, "driverId", ["position", "rank_pct"])
        cohort = (
            labels.groupby("date", observed=True)["position"]
            .agg(["mean", "median", "std", "count", "max"])
            .reset_index()
            .sort_values("date")
        )
        self.label_cohort_dates = _date_values(cohort["date"])
        self.label_cohort_values = cohort[["mean", "median", "std", "count", "max"]].to_numpy(dtype=np.float64)

    def _prepare_standings(self) -> None:
        standings = self.tables["standings"].copy()
        standings["standing_position"] = _numeric(standings["position"])
        standings["standing_points"] = _numeric(standings["points"])
        standings["standing_wins"] = _numeric(standings["wins"])
        self.driver_standings = _group_arrays(
            standings,
            "driverId",
            ["standing_position", "standing_points", "standing_wins"],
        )
        constructor_standings = self.tables["constructor_standings"].copy()
        constructor_standings["standing_position"] = _numeric(constructor_standings["position"])
        constructor_standings["standing_points"] = _numeric(constructor_standings["points"])
        constructor_standings["standing_wins"] = _numeric(constructor_standings["wins"])
        self.constructor_standings = _group_arrays(
            constructor_standings,
            "constructorId",
            ["standing_position", "standing_points", "standing_wins"],
        )

    def _prepare_qualifying(self) -> None:
        qualifying = self.tables["qualifying"].copy()
        fields = self.results[["raceId", "field_size"]].drop_duplicates("raceId")
        qualifying = qualifying.merge(fields, on="raceId", how="left")
        qualifying["qualifying_position"] = _numeric(qualifying["position"])
        qualifying["qualifying_norm"] = (
            _numeric(qualifying["position"]) - 1.0
        ) / np.maximum(_numeric(qualifying["field_size"]) - 1.0, 1.0)
        qualifying["constructor_value"] = _numeric(qualifying["constructorId"])
        self.driver_qualifying = _group_arrays(
            qualifying,
            "driverId",
            ["qualifying_position", "qualifying_norm", "constructor_value"],
        )
        self.constructor_qualifying = _group_arrays(
            qualifying,
            "constructorId",
            ["qualifying_position", "qualifying_norm"],
        )

    def _prepare_races(self) -> None:
        races = self.tables["races"].copy()
        fields = (
            self.results.groupby("raceId", observed=True)
            .agg(field_size=("driverId", "size"), constructor_count=("constructorId", "nunique"))
            .reset_index()
        )
        races = races.merge(fields, on="raceId", how="left").sort_values("date")
        races["field_size"] = _numeric(races["field_size"])
        races["constructor_count"] = _numeric(races["constructor_count"])
        races["round_value"] = _numeric(races["round"])
        races["circuit_value"] = _numeric(races["circuitId"])
        self.race_dates = _date_values(races["date"])
        self.race_values = races[["field_size", "constructor_count", "round_value", "circuit_value"]].to_numpy(dtype=np.float64)

    def _result_features(self, target: dict[str, float], driver_id: int, timestamp: int) -> tuple[int, np.ndarray]:
        history = self.driver_results.get(driver_id)
        career = _slice(history, timestamp)
        for days in (60, 120, 240, 365, 730):
            _summary(target, f"driver_d{days}", _slice(history, timestamp, days=days), extended=days in (120, 365))
        for count in (1, 3, 5, 10):
            _summary(target, f"driver_n{count}", _slice(history, timestamp, last_n=count), extended=count in (5, 10))
        _summary(target, "driver_career", career, extended=True)
        target["driver_days_since_race"] = _last_date(history, timestamp)
        year = pd.Timestamp(timestamp).year
        year_start = pd.Timestamp(year=year, month=1, day=1).value
        previous_start = pd.Timestamp(year=year - 1, month=1, day=1).value
        if history is None:
            current = np.empty((0, 0))
            previous = np.empty((0, 0))
        else:
            dates, values = history
            end = int(np.searchsorted(dates, timestamp, side="right"))
            current_start = int(np.searchsorted(dates, year_start, side="left"))
            previous_index = int(np.searchsorted(dates, previous_start, side="left"))
            current = values[current_start:end]
            previous = values[previous_index:current_start]
        _summary(target, "driver_season", current, extended=True)
        _summary(target, "driver_prior_season", previous, extended=False)
        recent = career[-6:] if len(career) else career
        target["driver_pos_trend6"] = _slope(recent[:, 0]) if len(recent) else np.nan
        target["driver_norm_trend6"] = _slope(recent[:, 1]) if len(recent) else np.nan
        if len(career):
            latest = career[-1]
            constructor_id = int(latest[6])
            target["last_position"] = float(latest[0])
            target["last_position_norm"] = float(latest[1])
            target["last_grid_norm"] = float(latest[2])
            target["last_points"] = float(latest[3])
            target["last_laps_ratio"] = float(latest[4])
            target["last_status_id"] = float(latest[7])
            target["last_classified"] = float(latest[10])
            target["last_finished"] = float(latest[11])
            target["last_pace_rank_norm"] = float(latest[15])
            target["last_time_gap_pct"] = float(latest[16])
            target["career_circuit_count"] = float(len(np.unique(career[:, 9])))
            recent_year = _slice(history, timestamp, days=365)
            target["recent_circuit_count"] = float(len(np.unique(recent_year[:, 9]))) if len(recent_year) else 0.0
            constructors = recent_year[:, 6] if len(recent_year) else np.empty(0)
            target["recent_constructor_count"] = float(len(np.unique(constructors))) if len(constructors) else 0.0
        else:
            constructor_id = -1
            for name in (
                "last_position",
                "last_position_norm",
                "last_grid_norm",
                "last_points",
                "last_laps_ratio",
                "last_status_id",
                "last_classified",
                "last_finished",
                "last_pace_rank_norm",
                "last_time_gap_pct",
                "career_circuit_count",
                "recent_circuit_count",
            ):
                target[name] = np.nan
            target["recent_constructor_count"] = 0.0
        qualifying_history = self.driver_qualifying.get(driver_id)
        if qualifying_history is not None:
            qualifying_dates, qualifying_values = qualifying_history
            qualifying_end = int(np.searchsorted(qualifying_dates, timestamp, side="right"))
            result_date = -1
            if history is not None:
                result_end = int(np.searchsorted(history[0], timestamp, side="right"))
                if result_end:
                    result_date = int(history[0][result_end - 1])
            if qualifying_end and int(qualifying_dates[qualifying_end - 1]) > result_date:
                constructor_id = int(qualifying_values[qualifying_end - 1, 2])
                target["constructor_from_qualifying"] = 1.0
            else:
                target["constructor_from_qualifying"] = 0.0
        else:
            target["constructor_from_qualifying"] = 0.0
        return constructor_id, career

    def _label_features(self, target: dict[str, float], driver_id: int, timestamp: int) -> None:
        cutoff = timestamp - 60 * DAY_NS
        history = self.driver_labels.get(driver_id)
        values = _slice(history, cutoff)
        target["label_history_count"] = float(len(values))
        target["label_days_since_origin"] = _last_date(history, cutoff)
        for count in (1, 2, 3, 5, 10):
            recent = values[-count:] if len(values) else values
            target[f"label_n{count}_mean"] = _finite_mean(recent[:, 0]) if len(recent) else np.nan
            target[f"label_n{count}_median"] = _finite_median(recent[:, 0]) if len(recent) else np.nan
        target["label_n3_std"] = _finite_std(values[-3:, 0]) if len(values) else np.nan
        target["label_n5_slope"] = _slope(values[-5:, 0]) if len(values) else np.nan
        target["label_last_rank_pct"] = float(values[-1, 1]) if len(values) else np.nan
        end = int(np.searchsorted(self.label_cohort_dates, cutoff, side="right"))
        if end:
            cohort = self.label_cohort_values[end - 1]
            target["prior_cohort_mean"] = float(cohort[0])
            target["prior_cohort_median"] = float(cohort[1])
            target["prior_cohort_std"] = float(cohort[2])
            target["prior_cohort_count"] = float(cohort[3])
            target["prior_cohort_max"] = float(cohort[4])
            target["days_since_cohort_origin"] = (cutoff - int(self.label_cohort_dates[end - 1])) / DAY_NS
        else:
            for name in ("mean", "median", "std", "count", "max"):
                target[f"prior_cohort_{name}"] = np.nan
            target["days_since_cohort_origin"] = np.nan

    def _standing_features(self, target: dict[str, float], driver_id: int, constructor_id: int, timestamp: int) -> None:
        for prefix, history in (
            ("driver_standing", self.driver_standings.get(driver_id)),
            ("constructor_standing", self.constructor_standings.get(constructor_id)),
        ):
            values = _slice(history, timestamp)
            dates = history[0] if history is not None else np.empty(0, dtype=np.int64)
            end = int(np.searchsorted(dates, timestamp, side="right"))
            year = pd.Timestamp(timestamp).year
            year_start = pd.Timestamp(year=year, month=1, day=1).value
            prior_start = pd.Timestamp(year=year - 1, month=1, day=1).value
            season_start = int(np.searchsorted(dates, year_start, side="left"))
            prior_index = int(np.searchsorted(dates, prior_start, side="left"))
            season_values = history[1][season_start:end] if history is not None else np.empty((0, 3))
            prior_values = history[1][prior_index:season_start] if history is not None else np.empty((0, 3))
            target[f"{prefix}_available"] = float(bool(len(values)))
            target[f"{prefix}_season_available"] = float(bool(len(season_values)))
            target[f"{prefix}_season_count"] = float(len(season_values))
            target[f"{prefix}_days_since"] = _last_date(history, timestamp)
            if len(values):
                target[f"{prefix}_position"] = float(values[-1, 0])
                target[f"{prefix}_points"] = float(values[-1, 1])
                target[f"{prefix}_wins"] = float(values[-1, 2])
                target[f"{prefix}_position_mean3"] = _finite_mean(values[-3:, 0])
                target[f"{prefix}_position_trend3"] = _slope(values[-3:, 0])
                target[f"{prefix}_points_gain"] = float(values[-1, 1] - values[-2, 1]) if len(values) > 1 else np.nan
            else:
                for name in ("position", "points", "wins", "position_mean3", "position_trend3", "points_gain"):
                    target[f"{prefix}_{name}"] = np.nan
            if len(prior_values):
                target[f"{prefix}_prior_position"] = float(prior_values[-1, 0])
                target[f"{prefix}_prior_points"] = float(prior_values[-1, 1])
                target[f"{prefix}_prior_wins"] = float(prior_values[-1, 2])
            else:
                target[f"{prefix}_prior_position"] = np.nan
                target[f"{prefix}_prior_points"] = np.nan
                target[f"{prefix}_prior_wins"] = np.nan
            target[f"{prefix}_position_vs_prior"] = (
                float(values[-1, 0] - prior_values[-1, 0])
                if len(values) and len(prior_values)
                else np.nan
            )

    def _qualifying_features(self, target: dict[str, float], driver_id: int, constructor_id: int, timestamp: int) -> None:
        for prefix, history in (
            ("driver_qualifying", self.driver_qualifying.get(driver_id)),
            ("constructor_qualifying", self.constructor_qualifying.get(constructor_id)),
        ):
            values = _slice(history, timestamp)
            recent = _slice(history, timestamp, days=365)
            target[f"{prefix}_available"] = float(bool(len(values)))
            target[f"{prefix}_last"] = float(values[-1, 0]) if len(values) else np.nan
            target[f"{prefix}_mean3"] = _finite_mean(values[-3:, 0]) if len(values) else np.nan
            target[f"{prefix}_mean365"] = _finite_mean(recent[:, 0]) if len(recent) else np.nan
            target[f"{prefix}_norm_mean3"] = _finite_mean(values[-3:, 1]) if len(values) else np.nan
            target[f"{prefix}_norm_mean365"] = _finite_mean(recent[:, 1]) if len(recent) else np.nan
            target[f"{prefix}_norm_std365"] = _finite_std(recent[:, 1]) if len(recent) else np.nan
            target[f"{prefix}_count365"] = float(len(recent))
            target[f"{prefix}_days_since"] = _last_date(history, timestamp)

    def _constructor_features(self, target: dict[str, float], constructor_id: int, driver_career: np.ndarray, timestamp: int) -> None:
        history = self.constructor_results_history.get(constructor_id)
        for days in (60, 120, 365, 730):
            values = _slice(history, timestamp, days=days)
            _summary(target, f"constructor_d{days}", values, extended=days in (120, 365))
            target[f"constructor_d{days}_race_count"] = float(len(np.unique(values[:, 8]))) if len(values) else 0.0
        for count in (4, 10, 20):
            _summary(target, f"constructor_n{count}", _slice(history, timestamp, last_n=count), extended=count == 10)
        points_history = self.constructor_points.get(constructor_id)
        for days in (120, 365):
            values = _slice(points_history, timestamp, days=days)
            target[f"constructor_result_points_d{days}"] = _finite_mean(values[:, 0]) if len(values) else np.nan
            target[f"constructor_result_count_d{days}"] = float(len(values))
        recent_driver = driver_career[-5:] if len(driver_career) else driver_career
        recent_team = _slice(history, timestamp, last_n=10)
        target["driver_team_pos_gap"] = (
            _finite_mean(recent_driver[:, 0]) - _finite_mean(recent_team[:, 0])
            if len(recent_driver) and len(recent_team)
            else np.nan
        )
        target["driver_team_norm_gap"] = (
            _finite_mean(recent_driver[:, 1]) - _finite_mean(recent_team[:, 1])
            if len(recent_driver) and len(recent_team)
            else np.nan
        )

    def _race_features(self, target: dict[str, float], timestamp: int) -> None:
        end = int(np.searchsorted(self.race_dates, timestamp, side="right"))
        dates = self.race_dates[:end]
        values = self.race_values[:end]
        target["days_since_any_race"] = (timestamp - int(dates[-1])) / DAY_NS if end else np.nan
        if end:
            target["last_field_size"] = float(values[-1, 0])
            target["last_constructor_count"] = float(values[-1, 1])
            target["last_race_round"] = float(values[-1, 2])
        else:
            target["last_field_size"] = np.nan
            target["last_constructor_count"] = np.nan
            target["last_race_round"] = np.nan
        for days in (60, 120, 365):
            start = int(np.searchsorted(dates, timestamp - days * DAY_NS, side="left"))
            recent = values[start:]
            target[f"races_d{days}_count"] = float(len(recent))
            target[f"races_d{days}_field_mean"] = _finite_mean(recent[:, 0]) if len(recent) else np.nan
            target[f"races_d{days}_field_std"] = _finite_std(recent[:, 0]) if len(recent) else np.nan
            target[f"races_d{days}_constructors_mean"] = _finite_mean(recent[:, 1]) if len(recent) else np.nan
            target[f"races_d{days}_circuits"] = float(len(np.unique(recent[:, 3]))) if len(recent) else 0.0

    def build(self, samples: pd.DataFrame) -> pd.DataFrame:
        rows = []
        cohort_sizes = samples.groupby("date", observed=True).size().to_dict()
        for sample in samples[["date", "driverId"]].itertuples(index=False):
            date = pd.Timestamp(sample.date)
            timestamp = date.value
            driver_id = int(sample.driverId)
            row = {}
            row["driver_id"] = float(driver_id)
            row["seed_year"] = float(date.year)
            row["seed_year_index"] = float(date.year - 1950)
            row["seed_month"] = float(date.month)
            row["seed_day_of_year"] = float(date.dayofyear)
            row["seed_sin"] = math.sin(2.0 * math.pi * date.dayofyear / 365.25)
            row["seed_cos"] = math.cos(2.0 * math.pi * date.dayofyear / 365.25)
            row["seed_cohort_size"] = float(cohort_sizes[date])
            metadata = self.driver_meta.get(driver_id, (None, -1))
            row["driver_nationality"] = float(metadata[1])
            row["driver_age"] = (timestamp - metadata[0]) / (365.25 * DAY_NS) if metadata[0] is not None else np.nan
            constructor_id, driver_career = self._result_features(row, driver_id, timestamp)
            row["constructor_id"] = float(constructor_id)
            row["constructor_nationality"] = float(self.constructor_meta.get(constructor_id, -1))
            self._label_features(row, driver_id, timestamp)
            self._standing_features(row, driver_id, constructor_id, timestamp)
            self._qualifying_features(row, driver_id, constructor_id, timestamp)
            self._constructor_features(row, constructor_id, driver_career, timestamp)
            self._race_features(row, timestamp)
            rows.append(row)
        frame = pd.DataFrame(rows)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        dates = pd.to_datetime(samples["date"]).reset_index(drop=True)
        rank_columns = [
            "label_n1_mean",
            "label_last_rank_pct",
            "driver_d60_pos_mean",
            "driver_d365_pos_mean",
            "driver_d365_norm_mean",
            "driver_n5_pos_mean",
            "driver_n5_grid_mean",
            "driver_standing_position",
            "driver_standing_prior_position",
            "driver_standing_points",
            "driver_qualifying_mean365",
            "driver_qualifying_norm_mean365",
            "constructor_d120_pos_mean",
            "constructor_d365_pos_mean",
            "constructor_d365_norm_mean",
            "constructor_standing_position",
            "constructor_standing_prior_position",
            "constructor_standing_points",
            "driver_team_pos_gap",
            "driver_d365_pace_mean",
            "driver_d365_finished_rate",
            "constructor_d365_pace_mean",
            "constructor_d365_finished_rate",
        ]
        for column in rank_columns:
            grouped = frame[column].groupby(dates, observed=True)
            frame[f"{column}_cohort_rank"] = grouped.rank(pct=True)
            median = grouped.transform("median")
            spread = grouped.transform("std")
            frame[f"{column}_cohort_delta"] = frame[column] - median
            frame[f"{column}_cohort_z"] = (frame[column] - median) / spread.replace(0.0, np.nan)
        return frame.astype(np.float32)


def build_cached_features(
    builder: FeatureBuilder,
    samples: pd.DataFrame,
    cache_directory: Path,
    enabled: bool,
) -> pd.DataFrame:
    if not enabled:
        return builder.build(samples)
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{FEATURE_CACHE_KEY}.parquet"
    key_frame = samples[["date", "driverId"]].copy().reset_index(drop=True)
    key_frame["__date_ns"] = pd.to_datetime(key_frame["date"]).astype("int64")
    key_frame["__driver_id"] = key_frame["driverId"].astype(np.int64)
    key_columns = ["__date_ns", "__driver_id"]
    if cache_path.exists():
        try:
            master = pd.read_parquet(cache_path)
        except Exception:
            master = pd.DataFrame(columns=key_columns)
    else:
        master = pd.DataFrame(columns=key_columns)
    if len(master):
        master = master.drop_duplicates(key_columns, keep="last")
        master_index = pd.MultiIndex.from_frame(master[key_columns])
    else:
        master_index = pd.MultiIndex.from_arrays([[], []], names=key_columns)
    requested_index = pd.MultiIndex.from_frame(key_frame[key_columns])
    present = requested_index.isin(master_index)
    if not np.all(present):
        missing_dates = set(key_frame.loc[~present, "__date_ns"].astype(np.int64))
        additions = []
        for date_ns in sorted(missing_dates):
            positions = np.flatnonzero(key_frame["__date_ns"].to_numpy() == date_ns)
            group = samples.iloc[positions]
            values = builder.build(group).reset_index(drop=True)
            values.insert(0, "__driver_id", group["driverId"].to_numpy(dtype=np.int64))
            values.insert(0, "__date_ns", np.full(len(group), date_ns, dtype=np.int64))
            additions.append(values)
        master = pd.concat([master, *additions], ignore_index=True)
        master = master.drop_duplicates(key_columns, keep="last")
        temporary = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        master.to_parquet(temporary, index=False)
        os.replace(temporary, cache_path)
    master = master.drop_duplicates(key_columns, keep="last").set_index(key_columns)
    selected = master.reindex(requested_index)
    if selected.isna().all(axis=1).any():
        raise RuntimeError("feature cache failed to resolve requested rows")
    return selected.reset_index(drop=True).astype(np.float32)


def recent_baseline(features: pd.DataFrame) -> np.ndarray:
    columns = [
        ("label_n1_mean", 0.40),
        ("driver_d60_pos_mean", 0.25),
        ("driver_n3_pos_mean", 0.15),
        ("driver_d365_pos_mean", 0.10),
        ("constructor_n10_pos_mean", 0.10),
    ]
    values = np.zeros(len(features), dtype=np.float64)
    weights = np.zeros(len(features), dtype=np.float64)
    for column, weight in columns:
        current = features[column].to_numpy(dtype=np.float64)
        mask = np.isfinite(current)
        values[mask] += current[mask] * weight
        weights[mask] += weight
    fallback = features["prior_cohort_median"].to_numpy(dtype=np.float64)
    fallback = np.where(np.isfinite(fallback), fallback, 13.0)
    return np.where(weights > 0, values / np.maximum(weights, 1e-9), fallback)


def _prepare_catboost(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in CAT_COLUMNS:
        result[column] = result[column].fillna(-1).round().astype(np.int64).astype(str)
    return result


def _compact_features(frame: pd.DataFrame) -> pd.DataFrame:
    excluded = [
        "classified_rate",
        "finished_rate",
        "top3_rate",
        "top10_rate",
        "points_rate",
        "pace_mean",
        "time_gap_mean",
        "last_classified",
        "last_finished",
        "last_pace_rank_norm",
        "last_time_gap_pct",
        "standing_season_",
        "standing_prior_",
        "standing_position_vs_prior",
        "qualifying_norm_",
    ]
    columns = [column for column in frame.columns if not any(token in column for token in excluded)]
    return frame[columns]


def fit_predict(
    train_features: pd.DataFrame,
    train_target: np.ndarray,
    predict_features: pd.DataFrame,
    debug: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor

    train_features = _compact_features(train_features)
    predict_features = _compact_features(predict_features)
    iterations = 80 if debug else 420
    cat_iterations = 80 if debug else 240
    train_center = (train_features["seed_cohort_size"].to_numpy(dtype=np.float64) + 1.0) / 2.0
    predict_center = (predict_features["seed_cohort_size"].to_numpy(dtype=np.float64) + 1.0) / 2.0
    lgb_centered = LGBMRegressor(
        objective="regression_l1",
        n_estimators=iterations,
        learning_rate=0.035,
        num_leaves=23,
        max_depth=-1,
        min_child_samples=28,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.82,
        reg_alpha=0.15,
        reg_lambda=2.5,
        random_state=1337,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbosity=-1,
    )
    lgb_centered.fit(train_features, train_target - train_center)
    lgb_prediction = predict_center + lgb_centered.predict(predict_features)
    cat_train = _prepare_catboost(train_features)
    cat_predict = _prepare_catboost(predict_features)
    seed_year = int(round(float(predict_features["seed_year"].mean())))
    seed_month = int(round(float(predict_features["seed_month"].mean())))
    model_seed = 5000 + (seed_year - 1950) * 7 + seed_month
    cat_model = CatBoostRegressor(
        loss_function="MAE",
        iterations=cat_iterations,
        depth=7,
        learning_rate=0.045,
        l2_leaf_reg=6.0,
        random_strength=0.25,
        random_seed=model_seed,
        verbose=False,
        allow_writing_files=False,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "1")),
    )
    cat_model.fit(cat_train, train_target, cat_features=CAT_COLUMNS)
    cat_prediction = cat_model.predict(cat_predict)
    prediction = 0.70 * lgb_prediction + 0.30 * cat_prediction
    group_keys = predict_features[["seed_year", "seed_day_of_year"]].to_numpy(dtype=np.int64)
    for key in np.unique(group_keys, axis=0):
        mask = np.all(group_keys == key, axis=1)
        center = (float(mask.sum()) + 1.0) / 2.0
        prediction[mask] += 0.20 * (center - float(prediction[mask].mean()))
    upper = np.maximum(
        18.0,
        np.nan_to_num(predict_features["races_d365_field_mean"].to_numpy(dtype=np.float64), nan=26.0) + 2.0,
    )
    prediction = np.clip(prediction, 1.0, upper)
    parts = {
        "lgb_centered": lgb_prediction,
        "catboost": cat_prediction,
        "recent": recent_baseline(predict_features),
    }
    return prediction.astype(np.float64), parts


def write_metrics(path: Path, values: dict) -> None:
    path.write_text(json.dumps(values, indent=2, sort_keys=True))
