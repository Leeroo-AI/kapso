from __future__ import annotations

import hashlib
import json
import math
import os
import zlib
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_VERSION = "lane0_all9_causal_v1"
DAY_NS = 86_400_000_000_000


def stable_code(value) -> float:
    if pd.isna(value):
        return -1.0
    return float(zlib.crc32(str(value).encode("utf-8")) % 10007)


def numeric_slope(values: np.ndarray, dates: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = dates.astype("datetime64[ns]").astype(np.int64) / DAY_NS
    x = x - x[-1]
    scale = float(np.dot(x, x))
    if scale <= 0:
        return 0.0
    y = values.astype(float)
    y = y - y[-1]
    return float(np.dot(x, y) / scale)


def weighted_mean(values: np.ndarray, decay: float = 0.85) -> float:
    if len(values) == 0:
        return np.nan
    weights = decay ** np.arange(len(values) - 1, -1, -1)
    return float(np.average(values.astype(float), weights=weights))


class FeatureBuilder:
    def __init__(self, db):
        self.circuits = db.table_dict["circuits"].df.copy()
        self.constructor_results = db.table_dict["constructor_results"].df.copy()
        self.constructor_standings = db.table_dict["constructor_standings"].df.copy()
        self.constructors = db.table_dict["constructors"].df.copy()
        self.drivers = db.table_dict["drivers"].df.copy()
        self.qualifying = db.table_dict["qualifying"].df.copy()
        self.races = db.table_dict["races"].df.copy()
        self.results = db.table_dict["results"].df.copy()
        self.standings = db.table_dict["standings"].df.copy()
        self._prepare()

    def _prepare(self) -> None:
        timed = [self.constructor_results, self.constructor_standings, self.qualifying, self.races, self.results, self.standings]
        for frame in timed:
            frame["date"] = pd.to_datetime(frame["date"])
        race_columns = self.races[["raceId", "year", "round", "circuitId"]]
        self.results = self.results.merge(race_columns, on="raceId", how="left", suffixes=("", "_race"))
        race_max_laps = self.results.groupby("raceId")["laps"].transform("max").replace(0, np.nan)
        race_field = self.results.groupby("raceId")["driverId"].transform("size").replace(0, np.nan)
        race_points = self.results.groupby("raceId")["points"].transform("sum").replace(0, np.nan)
        self.results["finish"] = self.results["statusId"].eq(1).astype(float)
        self.results["dnf"] = 1.0 - self.results["finish"]
        self.results["middle"] = self.results["statusId"].between(11, 19).astype(float)
        self.results["other"] = ((self.results["statusId"] != 1) & ~self.results["statusId"].between(11, 19)).astype(float)
        self.results["outcome_class"] = np.where(self.results["statusId"].eq(1), 0, np.where(self.results["statusId"].between(11, 19), 1, 2)).astype(int)
        self.results["laps_fraction"] = (self.results["laps"] / race_max_laps).clip(0, 1.5)
        self.results["grid_relative"] = self.results["grid"] / race_field
        self.results["position_relative"] = self.results["positionOrder"] / race_field
        self.results["points_share"] = (self.results["points"] / race_points).fillna(0.0)
        stand_field = self.standings.groupby("raceId")["driverId"].transform("size").replace(0, np.nan)
        self.standings["relative_position"] = self.standings["position"] / stand_field
        qual_field = self.qualifying.groupby("raceId")["driverId"].transform("size").replace(0, np.nan)
        self.qualifying["relative_position"] = self.qualifying["position"] / qual_field
        cr_total = self.constructor_results.groupby("raceId")["points"].transform("sum").replace(0, np.nan)
        self.constructor_results["points_share"] = (self.constructor_results["points"] / cr_total).fillna(0.0)
        cs_field = self.constructor_standings.groupby("raceId")["constructorId"].transform("size").replace(0, np.nan)
        self.constructor_standings["relative_position"] = self.constructor_standings["position"] / cs_field
        self.results = self.results.sort_values(["date", "raceId", "driverId"]).reset_index(drop=True)
        self.races = self.races.sort_values("date").reset_index(drop=True)
        self.standings = self.standings.sort_values(["date", "driverId"]).reset_index(drop=True)
        self.qualifying = self.qualifying.sort_values(["date", "driverId"]).reset_index(drop=True)
        self.constructor_results = self.constructor_results.sort_values(["date", "constructorId"]).reset_index(drop=True)
        self.constructor_standings = self.constructor_standings.sort_values(["date", "constructorId"]).reset_index(drop=True)
        self.result_driver = {int(k): v.reset_index(drop=True) for k, v in self.results.groupby("driverId", sort=False)}
        self.result_constructor = {int(k): v.reset_index(drop=True) for k, v in self.results.groupby("constructorId", sort=False)}
        self.result_circuit = {int(k): v.reset_index(drop=True) for k, v in self.results.groupby("circuitId", sort=False)}
        self.result_driver_circuit = {(int(a), int(b)): v.reset_index(drop=True) for (a, b), v in self.results.groupby(["driverId", "circuitId"], sort=False)}
        self.result_constructor_circuit = {(int(a), int(b)): v.reset_index(drop=True) for (a, b), v in self.results.groupby(["constructorId", "circuitId"], sort=False)}
        self.standing_driver = {int(k): v.reset_index(drop=True) for k, v in self.standings.groupby("driverId", sort=False)}
        self.qualifying_driver = {int(k): v.reset_index(drop=True) for k, v in self.qualifying.groupby("driverId", sort=False)}
        self.constructor_result_group = {int(k): v.reset_index(drop=True) for k, v in self.constructor_results.groupby("constructorId", sort=False)}
        self.constructor_standing_group = {int(k): v.reset_index(drop=True) for k, v in self.constructor_standings.groupby("constructorId", sort=False)}
        self.driver_map = self.drivers.set_index("driverId").to_dict("index")
        self.constructor_map = self.constructors.set_index("constructorId").to_dict("index")
        self.circuit_map = self.circuits.set_index("circuitId").to_dict("index")
        self.db_max_date = max(frame["date"].max() for frame in [self.results, self.races, self.standings, self.qualifying] if len(frame))
        self.calendar_cache: dict[tuple[pd.Timestamp, bool], tuple[dict[str, float], list[int], np.ndarray]] = {}

    def _history(self, groups: dict, key, timestamp: pd.Timestamp, inclusive: bool = True) -> pd.DataFrame:
        frame = groups.get(key)
        if frame is None or len(frame) == 0:
            return pd.DataFrame()
        values = frame["date"].to_numpy(dtype="datetime64[ns]")
        side = "right" if inclusive else "left"
        end = int(np.searchsorted(values, np.datetime64(timestamp), side=side))
        return frame.iloc[:end]

    def _result_features(self, driver_id: int, timestamp: pd.Timestamp, inclusive: bool) -> tuple[dict[str, float], int]:
        history = self._history(self.result_driver, driver_id, timestamp, inclusive)
        output: dict[str, float] = {}
        result_columns = ["dnf", "finish", "middle", "other", "grid", "grid_relative", "positionOrder", "position_relative", "points", "points_share", "laps_fraction"]
        output["res_count"] = float(len(history))
        for column in result_columns:
            output[f"res_{column}_mean"] = float(history[column].mean()) if len(history) else np.nan
            output[f"res_{column}_std"] = float(history[column].std(ddof=0)) if len(history) else np.nan
        for size in [1, 3, 5, 10, 20]:
            recent = history.tail(size)
            for column in ["dnf", "finish", "middle", "other", "grid_relative", "position_relative", "points", "points_share", "laps_fraction"]:
                output[f"res_{column}_{size}"] = float(recent[column].mean()) if len(recent) else np.nan
        for days in [30, 90, 365, 1095]:
            recent = history[history["date"] > timestamp - pd.Timedelta(days=days)] if len(history) else history
            output[f"res_count_{days}d"] = float(len(recent))
            for column in ["dnf", "finish", "middle", "other", "grid_relative", "position_relative", "points", "laps_fraction"]:
                output[f"res_{column}_{days}d"] = float(recent[column].mean()) if len(recent) else np.nan
        if len(history):
            dnf = history["dnf"].to_numpy(dtype=float)
            finish = history["finish"].to_numpy(dtype=float)
            output["res_dnf_ewma_slow"] = weighted_mean(dnf[-30:], 0.85)
            output["res_dnf_ewma_fast"] = weighted_mean(dnf[-12:], 0.65)
            output["res_grid_slope"] = numeric_slope(history["grid_relative"].tail(10).to_numpy(), history["date"].tail(10).to_numpy())
            output["res_position_slope"] = numeric_slope(history["position_relative"].tail(10).to_numpy(), history["date"].tail(10).to_numpy())
            output["res_points_slope"] = numeric_slope(history["points"].tail(10).to_numpy(), history["date"].tail(10).to_numpy())
            output["res_inactivity_days"] = float((timestamp - history["date"].iloc[-1]).total_seconds() / 86400)
            output["res_last_grid"] = float(history["grid"].iloc[-1])
            output["res_last_grid_relative"] = float(history["grid_relative"].iloc[-1])
            output["res_last_position"] = float(history["positionOrder"].iloc[-1])
            output["res_last_position_relative"] = float(history["position_relative"].iloc[-1])
            output["res_last_points"] = float(history["points"].iloc[-1])
            output["res_last_laps_fraction"] = float(history["laps_fraction"].iloc[-1])
            output["res_dnf_streak"] = float(next((i for i, value in enumerate(dnf[::-1]) if value == 0), len(dnf)))
            output["res_finish_streak"] = float(next((i for i, value in enumerate(finish[::-1]) if value == 0), len(finish)))
            teams = history["constructorId"].to_numpy(dtype=int)
            output["team_switches_10"] = float(np.sum(teams[-10:][1:] != teams[-10:][:-1])) if len(teams[-10:]) > 1 else 0.0
            current_team = int(teams[-1])
            boundary = len(teams) - 1
            while boundary > 0 and teams[boundary - 1] == current_team:
                boundary -= 1
            output["team_tenure_races"] = float(len(teams) - boundary)
            output["team_tenure_days"] = float((timestamp - history["date"].iloc[boundary]).total_seconds() / 86400)
        else:
            for name in ["res_dnf_ewma_slow", "res_dnf_ewma_fast", "res_grid_slope", "res_position_slope", "res_points_slope", "res_inactivity_days", "res_last_grid", "res_last_grid_relative", "res_last_position", "res_last_position_relative", "res_last_points", "res_last_laps_fraction", "res_dnf_streak", "res_finish_streak", "team_switches_10", "team_tenure_races", "team_tenure_days"]:
                output[name] = np.nan
            current_team = -1
        output["rookie"] = float(len(history) < 10)
        output["new_team"] = float(output.get("team_tenure_races", 0) <= 3) if len(history) else 1.0
        return output, current_team

    def _sequence_features(self, groups: dict, key: int, timestamp: pd.Timestamp, inclusive: bool, prefix: str, columns: list[str]) -> dict[str, float]:
        history = self._history(groups, key, timestamp, inclusive)
        output = {f"{prefix}_available": float(len(history) > 0), f"{prefix}_count": float(len(history))}
        for column in columns:
            values = history[column].tail(6).to_numpy(dtype=float) if len(history) else np.array([])
            dates = history["date"].tail(6).to_numpy() if len(history) else np.array([])
            output[f"{prefix}_{column}_latest"] = float(values[-1]) if len(values) else np.nan
            output[f"{prefix}_{column}_change"] = float(values[-1] - values[-2]) if len(values) > 1 else 0.0
            output[f"{prefix}_{column}_slope"] = numeric_slope(values, dates) if len(values) else 0.0
            output[f"{prefix}_{column}_mean5"] = float(np.mean(values[-5:])) if len(values) else np.nan
        output[f"{prefix}_days_since"] = float((timestamp - history["date"].iloc[-1]).total_seconds() / 86400) if len(history) else np.nan
        return output

    def _constructor_features(self, driver_id: int, constructor_id: int, timestamp: pd.Timestamp, inclusive: bool) -> dict[str, float]:
        output: dict[str, float] = {}
        history = self._history(self.result_constructor, constructor_id, timestamp, inclusive) if constructor_id >= 0 else pd.DataFrame()
        output["constructor_available"] = float(len(history) > 0)
        output["constructor_res_count"] = float(len(history))
        for suffix, recent in [("all", history), ("5", history.tail(5)), ("20", history.tail(20)), ("365d", history[history["date"] > timestamp - pd.Timedelta(days=365)] if len(history) else history)]:
            for column in ["dnf", "finish", "middle", "other", "grid_relative", "position_relative", "points", "points_share", "laps_fraction"]:
                output[f"constructor_{column}_{suffix}"] = float(recent[column].mean()) if len(recent) else np.nan
        recent_team = history[history["date"] > timestamp - pd.Timedelta(days=365)] if len(history) else history
        teammate = recent_team[recent_team["driverId"] != driver_id] if len(recent_team) else recent_team
        driver_recent = recent_team[recent_team["driverId"] == driver_id] if len(recent_team) else recent_team
        for column in ["dnf", "grid_relative", "position_relative", "points", "laps_fraction"]:
            team_value = float(teammate[column].mean()) if len(teammate) else np.nan
            driver_value = float(driver_recent[column].mean()) if len(driver_recent) else np.nan
            output[f"teammate_{column}"] = team_value
            output[f"driver_teammate_{column}_contrast"] = driver_value - team_value if np.isfinite(driver_value) and np.isfinite(team_value) else np.nan
        output.update(self._sequence_features(self.constructor_result_group, constructor_id, timestamp, inclusive, "constructor_result", ["points", "points_share"]) if constructor_id >= 0 else self._sequence_features({}, -1, timestamp, inclusive, "constructor_result", ["points", "points_share"]))
        output.update(self._sequence_features(self.constructor_standing_group, constructor_id, timestamp, inclusive, "constructor_standing", ["points", "position", "wins", "relative_position"]) if constructor_id >= 0 else self._sequence_features({}, -1, timestamp, inclusive, "constructor_standing", ["points", "position", "wins", "relative_position"]))
        constructor = self.constructor_map.get(constructor_id, {})
        output["constructor_identity"] = float(constructor_id)
        output["constructor_nationality"] = stable_code(constructor.get("nationality", np.nan))
        return output

    def _calendar_features(self, timestamp: pd.Timestamp, inclusive: bool) -> tuple[dict[str, float], list[int], np.ndarray]:
        key = (pd.Timestamp(timestamp).floor("s"), inclusive)
        cached = self.calendar_cache.get(key)
        if cached is not None:
            return cached
        side = "right" if inclusive else "left"
        end = int(np.searchsorted(self.races["date"].to_numpy(dtype="datetime64[ns]"), np.datetime64(timestamp), side=side))
        history = self.races.iloc[:end]
        year = int(timestamp.year)
        doy = int(timestamp.dayofyear)
        previous = self.races[(self.races["year"] < year) & (self.races["year"] >= year - 5)].copy()
        if len(previous):
            previous["forward_doy"] = (previous["date"].dt.dayofyear - doy) % 365
            candidates = previous[(previous["forward_doy"] > 0) & (previous["forward_doy"] <= 45)].copy()
            counts = previous[(previous["forward_doy"] > 0) & (previous["forward_doy"] <= 30)].groupby("year").size()
            khat = float(counts.reindex(range(max(int(previous["year"].min()), year - 5), year), fill_value=0).mean()) if len(counts) else 0.0
        else:
            candidates = previous
            khat = 0.0
        if len(candidates):
            candidates["weight"] = np.exp(-candidates["forward_doy"] / 20.0) * np.exp(-(year - candidates["year"] - 1) / 3.0)
            circuit_weights = candidates.groupby("circuitId")["weight"].sum().sort_values(ascending=False).head(5)
            circuit_ids = [int(value) for value in circuit_weights.index]
            weights = circuit_weights.to_numpy(dtype=float)
            weights = weights / weights.sum()
        elif len(history):
            circuit_ids = [int(history["circuitId"].iloc[-1])]
            weights = np.array([1.0])
        else:
            circuit_ids = []
            weights = np.array([], dtype=float)
        current_year = history[history["year"] == year]
        gaps = history["date"].diff().dt.total_seconds().div(86400).tail(10)
        output = {
            "calendar_khat": max(1.0, khat),
            "calendar_history_races": float(len(history)),
            "calendar_season_round": float(current_year["round"].max()) if len(current_year) else 0.0,
            "calendar_season_races_so_far": float(len(current_year)),
            "calendar_days_since_race": float((timestamp - history["date"].iloc[-1]).total_seconds() / 86400) if len(history) else np.nan,
            "calendar_gap_mean10": float(gaps.mean()) if gaps.notna().any() else np.nan,
            "calendar_gap_std10": float(gaps.std(ddof=0)) if gaps.notna().any() else np.nan,
            "calendar_likely_circuits": float(len(circuit_ids)),
        }
        self.calendar_cache[key] = (output, circuit_ids, weights)
        return output, circuit_ids, weights

    def _circuit_features(self, driver_id: int, constructor_id: int, timestamp: pd.Timestamp, inclusive: bool) -> dict[str, float]:
        calendar, circuit_ids, weights = self._calendar_features(timestamp, inclusive)
        output = dict(calendar)
        values = {name: [] for name in ["global_dnf", "driver_dnf", "constructor_dnf", "driver_count", "constructor_count", "lat", "lng", "alt", "country"]}
        for circuit_id in circuit_ids:
            general = self._history(self.result_circuit, circuit_id, timestamp, inclusive)
            driver = self._history(self.result_driver_circuit, (driver_id, circuit_id), timestamp, inclusive)
            team = self._history(self.result_constructor_circuit, (constructor_id, circuit_id), timestamp, inclusive) if constructor_id >= 0 else pd.DataFrame()
            circuit = self.circuit_map.get(circuit_id, {})
            global_rate = float(general["dnf"].mean()) if len(general) else np.nan
            driver_rate = float((driver["dnf"].sum() + 5 * global_rate) / (len(driver) + 5)) if len(driver) and np.isfinite(global_rate) else global_rate
            team_rate = float((team["dnf"].sum() + 8 * global_rate) / (len(team) + 8)) if len(team) and np.isfinite(global_rate) else global_rate
            values["global_dnf"].append(global_rate)
            values["driver_dnf"].append(driver_rate)
            values["constructor_dnf"].append(team_rate)
            values["driver_count"].append(float(len(driver)))
            values["constructor_count"].append(float(len(team)))
            values["lat"].append(float(circuit.get("lat", np.nan)))
            values["lng"].append(float(circuit.get("lng", np.nan)))
            values["alt"].append(float(circuit.get("alt", np.nan)))
            values["country"].append(stable_code(circuit.get("country", np.nan)))
        for name, items in values.items():
            array = np.asarray(items, dtype=float)
            valid = np.isfinite(array)
            output[f"circuit_mix_{name}"] = float(np.average(array[valid], weights=weights[valid])) if valid.any() else np.nan
        upper = self.results["date"] <= timestamp if inclusive else self.results["date"] < timestamp
        recent_era = self.results[upper & (self.results["date"] > timestamp - pd.Timedelta(days=1095))]
        output["state_era_dnf"] = float(recent_era["dnf"].mean()) if len(recent_era) else np.nan
        return output

    def row(self, driver_id: int, timestamp: pd.Timestamp, inclusive: bool) -> dict[str, float]:
        result, constructor_id = self._result_features(driver_id, timestamp, inclusive)
        output = dict(result)
        output.update(self._sequence_features(self.standing_driver, driver_id, timestamp, inclusive, "standing", ["points", "position", "wins", "relative_position"]))
        output.update(self._sequence_features(self.qualifying_driver, driver_id, timestamp, inclusive, "qualifying", ["position", "relative_position"]))
        output.update(self._constructor_features(driver_id, constructor_id, timestamp, inclusive))
        output.update(self._circuit_features(driver_id, constructor_id, timestamp, inclusive))
        driver = self.driver_map.get(driver_id, {})
        dob = pd.to_datetime(driver.get("dob", pd.NaT))
        output["driver_identity"] = float(driver_id)
        output["driver_age"] = float((timestamp - dob).total_seconds() / (365.25 * 86400)) if not pd.isna(dob) else np.nan
        output["driver_nationality"] = stable_code(driver.get("nationality", np.nan))
        output["driver_constructor_nationality_same"] = float(output["driver_nationality"] == output["constructor_nationality"]) if constructor_id >= 0 else 0.0
        output["seed_year"] = float(timestamp.year)
        output["seed_month_sin"] = math.sin(2 * math.pi * timestamp.month / 12)
        output["seed_month_cos"] = math.cos(2 * math.pi * timestamp.month / 12)
        output["seed_doy_sin"] = math.sin(2 * math.pi * timestamp.dayofyear / 365.25)
        output["seed_doy_cos"] = math.cos(2 * math.pi * timestamp.dayofyear / 365.25)
        output["state_driver_dnf"] = output["res_dnf_ewma_slow"]
        output["state_constructor_dnf"] = output["constructor_dnf_20"]
        output["state_circuit_dnf"] = output["circuit_mix_global_dnf"]
        output["state_driver_circuit_dnf"] = output["circuit_mix_driver_dnf"]
        output["state_constructor_circuit_dnf"] = output["circuit_mix_constructor_dnf"]
        output["state_driver_minus_era"] = output["state_driver_dnf"] - output["state_era_dnf"]
        output["state_constructor_minus_era"] = output["state_constructor_dnf"] - output["state_era_dnf"]
        return output


IMPORTANT_COHORT_FEATURES = [
    "res_dnf_mean", "res_dnf_3", "res_dnf_5", "res_dnf_10", "res_dnf_20", "res_dnf_365d",
    "res_grid_relative_5", "res_position_relative_5", "res_points_5", "res_laps_fraction_5",
    "res_count", "res_inactivity_days", "res_dnf_ewma_slow", "standing_position_latest",
    "standing_points_latest", "qualifying_relative_position_latest", "constructor_dnf_20",
    "constructor_grid_relative_20", "constructor_position_relative_20", "teammate_dnf",
    "driver_teammate_dnf_contrast", "team_tenure_races", "driver_age", "circuit_mix_driver_dnf",
    "circuit_mix_constructor_dnf", "state_driver_minus_era", "state_constructor_minus_era",
]


def add_cohort_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    groups = output.groupby("date", sort=False)
    additions = {
        "field_size": groups["driverId"].transform("size").astype(float),
        "constructor_pair_count": groups["constructor_identity"].transform(lambda values: values.map(values.value_counts())).astype(float),
        "rookie_count": groups["rookie"].transform("sum").astype(float),
    }
    for column in IMPORTANT_COHORT_FEATURES:
        values = output[column].astype(float)
        mean = groups[column].transform("mean")
        std = groups[column].transform("std").replace(0, np.nan)
        minimum = groups[column].transform("min")
        maximum = groups[column].transform("max")
        additions[f"cohort_{column}_rank"] = groups[column].rank(method="average", pct=False)
        additions[f"cohort_{column}_pct"] = groups[column].rank(method="average", pct=True)
        additions[f"cohort_{column}_z"] = (values - mean) / std
        additions[f"cohort_{column}_gap_best"] = maximum - values
        additions[f"cohort_{column}_gap_worst"] = values - minimum
        additions[f"cohort_{column}_range"] = maximum - minimum
    for column in ["res_dnf_5", "res_grid_relative_5", "standing_position_latest", "constructor_dnf_20"]:
        additions[f"field_dispersion_{column}"] = groups[column].transform("std")
    return pd.concat([output, pd.DataFrame(additions, index=output.index)], axis=1)


def _cache_identity(rows: pd.DataFrame, kind: str) -> str:
    payload = kind + "|" + "|".join(str(int(value)) for value in rows["driverId"].to_numpy())
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _source_digest(source: pd.DataFrame) -> str:
    dates = source["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    drivers = source["driverId"].to_numpy(dtype=np.int64)
    return hashlib.sha256(dates.tobytes() + drivers.tobytes()).hexdigest()


def build_features(builder: FeatureBuilder, seeds: pd.DataFrame, cache_root: Path, kind: str, rolling: bool, bundle: str | None = None) -> pd.DataFrame:
    source = seeds[["date", "driverId"]].copy()
    source["date"] = pd.to_datetime(source["date"])
    source["_row_id"] = np.arange(len(source), dtype=int)
    cache_dir = cache_root / FEATURE_VERSION / "features"
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    remaining_source = source
    bundle_data = cache_root / FEATURE_VERSION / "assembled" / f"{bundle}.pkl" if bundle else None
    bundle_meta = cache_root / FEATURE_VERSION / "assembled" / f"{bundle}.json" if bundle else None
    if bundle_data is not None and bundle_meta is not None and bundle_data.exists() and bundle_meta.exists() and len(source):
        try:
            meta = json.loads(bundle_meta.read_text())
            count = int(meta["rows"])
            cutoff = pd.Timestamp(meta["source_cutoff"])
            valid = meta.get("feature_version") == FEATURE_VERSION and meta.get("kind") == kind and count <= len(source) and cutoff <= source["date"].max()
            if valid:
                assembled = pd.read_pickle(bundle_data)
                cached_keys = pd.MultiIndex.from_frame(assembled[["date", "driverId"]])
                current_keys = pd.MultiIndex.from_frame(source[["date", "driverId"]])
                if len(assembled) == count and cached_keys.is_unique and current_keys.is_unique and meta.get("digest") == _source_digest(assembled[["date", "driverId"]]) and cached_keys.isin(current_keys).all():
                    present = current_keys.isin(cached_keys)
                    reused = assembled.set_index(["date", "driverId"]).loc[current_keys[present]].reset_index()
                    reused["_row_id"] = source.loc[present, "_row_id"].to_numpy()
                    parts.append(reused)
                    remaining_source = source.loc[~present]
        except Exception:
            remaining_source = source
    inclusive = kind == "task"
    for timestamp, group in remaining_source.groupby("date", sort=False):
        group = group.copy()
        identity = _cache_identity(group, kind)
        stamp = pd.Timestamp(timestamp).strftime("%Y%m%dT%H%M%S")
        data_path = cache_dir / f"{kind}_{stamp}_{identity}.pkl"
        meta_path = cache_dir / f"{kind}_{stamp}_{identity}.json"
        cached = None
        if data_path.exists() and meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                source_cutoff = pd.Timestamp(meta["source_cutoff"])
                if meta.get("feature_version") == FEATURE_VERSION and meta.get("identity") == identity and source_cutoff <= pd.Timestamp(timestamp):
                    cached = pd.read_pickle(data_path)
            except Exception:
                cached = None
        if cached is None:
            records = []
            for row in group.to_dict("records"):
                record = builder.row(int(row["driverId"]), pd.Timestamp(row["date"]), inclusive)
                record["date"] = pd.Timestamp(row["date"])
                record["driverId"] = int(row["driverId"])
                record["_row_id"] = int(row["_row_id"])
                records.append(record)
            cached = pd.DataFrame.from_records(records)
            cache_legal = rolling or pd.Timestamp(timestamp) <= pd.Timestamp(builder.db_max_date)
            if cache_legal:
                temp_data = Path(str(data_path) + f".{os.getpid()}.tmp")
                temp_meta = Path(str(meta_path) + f".{os.getpid()}.tmp")
                cached.to_pickle(temp_data)
                temp_meta.write_text(json.dumps({"feature_version": FEATURE_VERSION, "identity": identity, "origin": pd.Timestamp(timestamp).isoformat(), "source_cutoff": pd.Timestamp(timestamp).isoformat(), "kind": kind}))
                os.replace(temp_data, data_path)
                os.replace(temp_meta, meta_path)
        else:
            cached = cached.copy()
            cached["_row_id"] = group["_row_id"].to_numpy()
        parts.append(cached)
    combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    combined = combined.sort_values("_row_id").reset_index(drop=True)
    if bundle_data is not None and bundle_meta is not None and len(combined) == len(source) and len(source):
        bundle_data.parent.mkdir(parents=True, exist_ok=True)
        temp_data = Path(str(bundle_data) + f".{os.getpid()}.tmp")
        temp_meta = Path(str(bundle_meta) + f".{os.getpid()}.tmp")
        combined.to_pickle(temp_data)
        temp_meta.write_text(json.dumps({"feature_version": FEATURE_VERSION, "kind": kind, "rows": len(source), "source_cutoff": source["date"].max().isoformat(), "digest": _source_digest(source)}))
        os.replace(temp_data, bundle_data)
        os.replace(temp_meta, bundle_meta)
    return add_cohort_features(combined)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column not in {"date", "driverId", "_row_id"}]


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    exact = {
        "res_count", "res_dnf_mean", "res_finish_mean", "res_middle_mean", "res_other_mean",
        "res_dnf_ewma_slow", "res_dnf_ewma_fast", "res_grid_slope", "res_position_slope",
        "res_points_slope", "res_inactivity_days", "res_last_grid_relative", "res_last_position_relative",
        "res_last_points", "res_last_laps_fraction", "res_dnf_streak", "res_finish_streak",
        "team_switches_10", "team_tenure_races", "team_tenure_days", "rookie", "new_team",
        "constructor_available", "constructor_res_count", "constructor_identity", "constructor_nationality",
        "driver_identity", "driver_age", "driver_nationality", "driver_constructor_nationality_same",
        "seed_year", "seed_month_sin", "seed_month_cos", "seed_doy_sin", "seed_doy_cos",
        "field_size", "constructor_pair_count", "rookie_count",
        "field_dispersion_res_dnf_5", "field_dispersion_res_grid_relative_5",
        "field_dispersion_standing_position_latest", "field_dispersion_constructor_dnf_20",
    }
    for size in [1, 3, 5, 10, 20]:
        for measure in ["dnf", "grid_relative", "position_relative", "points", "laps_fraction"]:
            exact.add(f"res_{measure}_{size}")
        if size in [5, 20]:
            exact.add(f"res_middle_{size}")
            exact.add(f"res_other_{size}")
    for days in [30, 90, 365, 1095]:
        exact.add(f"res_count_{days}d")
        for measure in ["dnf", "grid_relative", "position_relative", "points", "laps_fraction"]:
            exact.add(f"res_{measure}_{days}d")
        if days in [365, 1095]:
            exact.add(f"res_middle_{days}d")
            exact.add(f"res_other_{days}d")
    for suffix in ["5", "20", "365d"]:
        for measure in ["dnf", "grid_relative", "position_relative", "points", "points_share", "laps_fraction"]:
            exact.add(f"constructor_{measure}_{suffix}")
    exact.add("constructor_middle_20")
    exact.add("constructor_other_20")
    cohort_bases = [
        "res_dnf_mean", "res_dnf_5", "res_dnf_20", "res_dnf_365d", "res_grid_relative_5",
        "res_position_relative_5", "res_points_5", "res_count", "res_inactivity_days", "res_dnf_ewma_slow",
        "standing_position_latest", "standing_points_latest", "qualifying_relative_position_latest",
        "constructor_dnf_20", "constructor_grid_relative_20", "constructor_position_relative_20",
        "teammate_dnf", "team_tenure_races", "driver_age", "circuit_mix_driver_dnf",
        "circuit_mix_constructor_dnf", "state_driver_minus_era", "state_constructor_minus_era",
    ]
    for base in cohort_bases:
        exact.add(f"cohort_{base}_pct")
    prefixes = ("state_", "calendar_", "circuit_mix_", "teammate_", "driver_teammate_")
    selected = []
    for column in feature_columns(frame):
        sequence = column.startswith("standing_") or column.startswith("qualifying_") or column.startswith("constructor_result_") or column.startswith("constructor_standing_")
        compact_sequence = sequence and any(token in column for token in ["available", "days_since", "latest", "change", "slope"])
        if column in exact or column.startswith(prefixes) or compact_sequence:
            selected.append(column)
    return selected
