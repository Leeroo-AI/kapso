# Imports

from __future__ import annotations

import json
import fcntl
import hashlib
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# Constants

WINDOWS = (1, 3, 5, 10)
LOWER_IS_BETTER = {
    "q_pos_pct_l1",
    "q_pos_pct_l3",
    "q_pos_pct_l5",
    "q_pos_pct_l10",
    "result_grid_l3",
    "result_grid_l5",
    "result_finish_l3",
    "result_finish_l5",
    "result_dnf_l5",
    "standing_position",
    "ctor_q_best_l3",
    "ctor_q_mean_l5",
    "ctor_standing_position",
    "ctor_reliability_l5",
}
RELATIVE_BASE = [
    "q_pos_pct_l3",
    "q_pos_pct_l10",
    "q_top3_l3",
    "q_top3_l10",
    "result_grid_l3",
    "result_finish_l3",
    "result_dnf_l5",
    "standing_position",
    "ctor_q_best_l3",
    "ctor_q_top3_l5",
    "ctor_standing_position",
    "ctor_points_share_l5",
    "driver_circuit_rate",
    "ctor_circuit_rate",
]
BASE_FEATURES = [
    *[f"q_pos_pct_l{w}" for w in WINDOWS],
    *[f"q_top3_l{w}" for w in WINDOWS],
    *[f"q_gap_l{w}" for w in (3, 5, 10)],
    *[f"q_h2h_l{w}" for w in (3, 5, 10)],
    "q_experience",
    "inactivity_days",
    "constructor_tenure",
    "constructor_changed",
    "age_years",
    "rookie",
    *[f"result_grid_l{w}" for w in (1, 3, 5, 10)],
    *[f"result_finish_l{w}" for w in (1, 3, 5, 10)],
    *[f"result_dnf_l{w}" for w in (3, 5, 10)],
    *[f"result_points_l{w}" for w in (3, 5, 10)],
    "result_experience",
    "standing_position",
    "standing_points",
    "standing_wins",
    "standing_trajectory",
    *[f"ctor_q_best_l{w}" for w in (1, 3, 5, 10)],
    *[f"ctor_q_mean_l{w}" for w in (3, 5, 10)],
    *[f"ctor_q_top3_l{w}" for w in (3, 5, 10)],
    "ctor_q_experience",
    "ctor_standing_position",
    "ctor_standing_points",
    "ctor_standing_wins",
    "ctor_standing_trajectory",
    *[f"ctor_points_share_l{w}" for w in (3, 5, 10)],
    *[f"ctor_reliability_l{w}" for w in (3, 5, 10)],
    "driver_circuit_rate",
    "driver_circuit_count",
    "ctor_circuit_rate",
    "ctor_circuit_count",
    "circuit_lat",
    "circuit_lng",
    "circuit_alt",
    "driver_home_circuit",
    "constructor_home_circuit",
    "event_year",
    "event_round",
    "event_doy_sin",
    "event_doy_cos",
]
RELATIVE_FEATURES = [
    f"{column}_{suffix}"
    for column in RELATIVE_BASE
    for suffix in ("rank", "pct", "z", "gap")
]
EVENT_FEATURES = BASE_FEATURES + RELATIVE_FEATURES
DIRECT_BASE = [
    "q_pos_pct_l1",
    "q_pos_pct_l3",
    "q_pos_pct_l10",
    "q_top3_l3",
    "q_top3_l10",
    "q_gap_l5",
    "q_h2h_l5",
    "q_experience",
    "inactivity_days",
    "constructor_tenure",
    "constructor_changed",
    "age_years",
    "rookie",
    "result_grid_l3",
    "result_grid_l5",
    "result_finish_l3",
    "result_finish_l5",
    "result_dnf_l5",
    "result_points_l5",
    "result_experience",
    "standing_position",
    "standing_trajectory",
    "ctor_q_best_l3",
    "ctor_q_top3_l5",
    "ctor_q_experience",
    "ctor_standing_position",
    "ctor_standing_trajectory",
    "ctor_points_share_l5",
    "ctor_reliability_l5",
    "origin_doy_sin",
    "origin_doy_cos",
    "schedule_expected_n",
    "schedule_entropy",
    "schedule_circuit_entropy",
    "schedule_latest_round",
    "schedule_phase",
    "schedule_cadence",
]
DIRECT_RELATIVE_BASE = [
    "q_pos_pct_l3",
    "q_pos_pct_l10",
    "q_top3_l3",
    "q_top3_l10",
    "result_grid_l3",
    "result_finish_l3",
    "standing_position",
    "ctor_q_best_l3",
    "ctor_q_top3_l5",
    "ctor_standing_position",
]
DIRECT_FEATURES = DIRECT_BASE + [
    f"{column}_{suffix}"
    for column in DIRECT_RELATIVE_BASE
    for suffix in ("rank", "pct", "z", "gap")
]
SUMMARY_FEATURES = [
    "structural_noisy_or",
    "structural_conservative",
    "event_probability_max",
    "event_probability_mean",
    "schedule_expected_n",
    "schedule_entropy",
    "schedule_circuit_entropy",
]
PSEUDO_OFFSETS = (10, 20)
PSEUDO_WEIGHT = 0.35
HALF_LIFE_DAYS = 365.25 * 5.0
PSEUDO_CACHE_VERSION = "exact_grid_v1_direct_features_v1"


# Utilities

def _rolling(grouped, column: str, window: int, shift: bool) -> pd.Series:
    return grouped[column].transform(
        lambda values: values.shift(1).rolling(window, min_periods=1).mean()
        if shift
        else values.rolling(window, min_periods=1).mean()
    )


def _asof(
    left: pd.DataFrame,
    right: pd.DataFrame,
    by: str | list[str],
    columns: list[str],
    exact: bool = True,
) -> pd.DataFrame:
    keys = [by] if isinstance(by, str) else by
    left_frame = left.copy()
    left_frame["_asof_order"] = np.arange(len(left_frame))
    right_frame = right[[*keys, "date", *columns]].copy()
    for key in keys:
        left_frame[key] = pd.to_numeric(left_frame[key], errors="coerce").fillna(-1).astype(np.int64)
        right_frame[key] = pd.to_numeric(right_frame[key], errors="coerce").fillna(-1).astype(np.int64)
    left_frame = left_frame.sort_values("date", kind="mergesort")
    right_frame = right_frame.sort_values("date", kind="mergesort")
    merged = pd.merge_asof(
        left_frame,
        right_frame,
        on="date",
        by=keys,
        direction="backward",
        allow_exact_matches=exact,
    )
    return merged.sort_values("_asof_order").drop(columns="_asof_order").reset_index(drop=True)


def _relative(frame: pd.DataFrame, group: str, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    grouped = result.groupby(group, sort=False)
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce")
        lower = column in LOWER_IS_BETTER
        rank = grouped[column].rank(method="average", ascending=lower)
        size = grouped[column].transform("count").clip(lower=1)
        strength = -values if lower else values
        mean = strength.groupby(result[group]).transform("mean")
        std = strength.groupby(result[group]).transform("std").replace(0, np.nan)
        leader = grouped[column].transform("min" if lower else "max")
        result[f"{column}_rank"] = rank
        result[f"{column}_pct"] = 1.0 - (rank - 1.0) / (size - 1.0).clip(lower=1.0)
        result[f"{column}_z"] = (strength - mean) / std
        result[f"{column}_gap"] = values - leader if lower else leader - values
    return result


def _relative_single(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(values)
        ranks = np.full(len(values), np.nan)
        percentiles = np.full(len(values), np.nan)
        zscores = np.full(len(values), np.nan)
        gaps = np.full(len(values), np.nan)
        if valid.any():
            lower = column in LOWER_IS_BETTER
            oriented = values[valid] if lower else -values[valid]
            order = np.argsort(oriented, kind="mergesort")
            ordinal = np.empty(len(order), dtype=float)
            ordinal[order] = np.arange(1.0, len(order) + 1.0)
            ranks[valid] = ordinal
            percentiles[valid] = 1.0 - (ordinal - 1.0) / max(1.0, len(order) - 1.0)
            strength = -values[valid] if lower else values[valid]
            deviation = float(np.std(strength, ddof=1)) if len(strength) > 1 else float("nan")
            zscores[valid] = (strength - float(np.mean(strength))) / deviation
            leader = float(np.min(values[valid]) if lower else np.max(values[valid]))
            gaps[valid] = values[valid] - leader if lower else leader - values[valid]
        result[f"{column}_rank"] = ranks
        result[f"{column}_pct"] = percentiles
        result[f"{column}_z"] = zscores
        result[f"{column}_gap"] = gaps
    return result


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)


def _logit(values: np.ndarray) -> np.ndarray:
    probability = _clip_probability(values)
    return np.log(probability / (1.0 - probability))


def _logit_blend(direct: np.ndarray, structural: np.ndarray, weight: float) -> np.ndarray:
    logits = (1.0 - weight) * _logit(direct) + weight * _logit(structural)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def _safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def _numeric(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.reindex(columns=columns).to_numpy(dtype=np.float64)


# Feature store

class FeatureStore:
    def __init__(self, database: dict[str, pd.DataFrame]):
        self.tables = database
        self.races = database["races"].copy().sort_values("date").reset_index(drop=True)
        self.drivers = database["drivers"].copy()
        self.constructors = database["constructors"].copy()
        self.circuits = database["circuits"].copy()
        self._schedule_cache: dict[pd.Timestamp, dict] = {}
        self._prepare_qualifying()
        self._prepare_results()
        self._prepare_standings()
        self._prepare_constructor_history()

    def _prepare_qualifying(self) -> None:
        race_columns = self.races[["raceId", "circuitId", "year", "round", "date"]].rename(
            columns={"date": "race_date"}
        )
        qualifying = self.tables["qualifying"].copy()
        qualifying = qualifying.merge(race_columns, on="raceId", how="left")
        qualifying["date"] = pd.to_datetime(qualifying["date"])
        qualifying = qualifying.sort_values(["date", "raceId", "position"]).reset_index(drop=True)
        qualifying["event_label"] = (qualifying["position"] <= 3).astype(int)
        starters = qualifying.groupby("raceId")["position"].transform("max").clip(lower=2)
        qualifying["position_pct_outcome"] = (qualifying["position"] - 1.0) / (starters - 1.0)
        team_count = qualifying.groupby(["raceId", "constructorId"])["position"].transform("count")
        team_sum = qualifying.groupby(["raceId", "constructorId"])["position"].transform("sum")
        teammate_mean = (team_sum - qualifying["position"]) / (team_count - 1).replace(0, np.nan)
        qualifying["gap_outcome"] = teammate_mean - qualifying["position"]
        qualifying["h2h_outcome"] = (qualifying["position"] < teammate_mean).astype(float).where(team_count > 1)
        qualifying["previous_constructor"] = qualifying.groupby("driverId")["constructorId"].shift(1)
        qualifying["constructor_changed"] = (
            qualifying["constructorId"] != qualifying["previous_constructor"]
        ).astype(float).where(qualifying["previous_constructor"].notna(), 0.0)
        driver_group = qualifying.groupby("driverId", sort=False)
        qualifying["q_experience"] = driver_group.cumcount().astype(float)
        previous_date = driver_group["date"].shift(1)
        qualifying["inactivity_days"] = (qualifying["date"] - previous_date).dt.days.astype(float)
        qualifying["constructor_tenure"] = qualifying.groupby(
            ["driverId", "constructorId"], sort=False
        ).cumcount().astype(float)
        for window in WINDOWS:
            qualifying[f"q_pos_pct_l{window}"] = _rolling(
                driver_group, "position_pct_outcome", window, True
            )
            qualifying[f"q_top3_l{window}"] = _rolling(driver_group, "event_label", window, True)
        for window in (3, 5, 10):
            qualifying[f"q_gap_l{window}"] = _rolling(driver_group, "gap_outcome", window, True)
            qualifying[f"q_h2h_l{window}"] = _rolling(driver_group, "h2h_outcome", window, True)
        driver_circuit = qualifying.groupby(["driverId", "circuitId"], sort=False)
        qualifying["driver_circuit_count"] = driver_circuit.cumcount().astype(float)
        qualifying["driver_circuit_success"] = driver_circuit["event_label"].cumsum() - qualifying["event_label"]
        constructor_circuit_race = qualifying.groupby(
            ["date", "raceId", "constructorId", "circuitId"], sort=False
        ).agg(
            circuit_entries=("event_label", "size"),
            circuit_successes=("event_label", "sum"),
        ).reset_index()
        constructor_circuit_race = constructor_circuit_race.sort_values(
            ["date", "raceId", "constructorId"]
        ).reset_index(drop=True)
        constructor_circuit_group = constructor_circuit_race.groupby(
            ["constructorId", "circuitId"], sort=False
        )
        constructor_circuit_race["ctor_circuit_count"] = (
            constructor_circuit_group["circuit_entries"].cumsum()
            - constructor_circuit_race["circuit_entries"]
        ).astype(float)
        constructor_circuit_race["ctor_circuit_success"] = (
            constructor_circuit_group["circuit_successes"].cumsum()
            - constructor_circuit_race["circuit_successes"]
        ).astype(float)
        qualifying = qualifying.merge(
            constructor_circuit_race[
                ["raceId", "constructorId", "ctor_circuit_count", "ctor_circuit_success"]
            ],
            on=["raceId", "constructorId"],
            how="left",
        )
        constructor_race = qualifying.groupby(
            ["date", "raceId", "constructorId"], sort=False
        ).agg(
            ctor_q_best_outcome=("position_pct_outcome", "min"),
            ctor_q_mean_outcome=("position_pct_outcome", "mean"),
            ctor_q_top3_outcome=("event_label", "mean"),
        ).reset_index()
        constructor_race = constructor_race.sort_values(["date", "raceId", "constructorId"]).reset_index(drop=True)
        constructor_group = constructor_race.groupby("constructorId", sort=False)
        constructor_race["ctor_q_experience"] = constructor_group.cumcount().astype(float)
        for window in WINDOWS:
            constructor_race[f"ctor_q_best_l{window}"] = _rolling(
                constructor_group, "ctor_q_best_outcome", window, True
            )
        for window in (3, 5, 10):
            constructor_race[f"ctor_q_mean_l{window}"] = _rolling(
                constructor_group, "ctor_q_mean_outcome", window, True
            )
            constructor_race[f"ctor_q_top3_l{window}"] = _rolling(
                constructor_group, "ctor_q_top3_outcome", window, True
            )
        constructor_pre_columns = [
            "ctor_q_experience",
            *[f"ctor_q_best_l{window}" for window in WINDOWS],
            *[f"ctor_q_mean_l{window}" for window in (3, 5, 10)],
            *[f"ctor_q_top3_l{window}" for window in (3, 5, 10)],
        ]
        qualifying = qualifying.merge(
            constructor_race[["raceId", "constructorId", *constructor_pre_columns]],
            on=["raceId", "constructorId"],
            how="left",
        )
        current = qualifying.copy()
        current_group = current.groupby("driverId", sort=False)
        for window in WINDOWS:
            current[f"q_pos_pct_l{window}"] = _rolling(
                current_group, "position_pct_outcome", window, False
            )
            current[f"q_top3_l{window}"] = _rolling(current_group, "event_label", window, False)
        for window in (3, 5, 10):
            current[f"q_gap_l{window}"] = _rolling(current_group, "gap_outcome", window, False)
            current[f"q_h2h_l{window}"] = _rolling(current_group, "h2h_outcome", window, False)
        current["q_experience"] = current_group.cumcount().astype(float) + 1.0
        current["constructor_tenure"] = current.groupby(
            ["driverId", "constructorId"], sort=False
        ).cumcount().astype(float) + 1.0
        current["last_q_date"] = current["date"]
        constructor_current = constructor_race.copy()
        constructor_current_group = constructor_current.groupby("constructorId", sort=False)
        for window in WINDOWS:
            constructor_current[f"ctor_q_best_l{window}"] = _rolling(
                constructor_current_group, "ctor_q_best_outcome", window, False
            )
        for window in (3, 5, 10):
            constructor_current[f"ctor_q_mean_l{window}"] = _rolling(
                constructor_current_group, "ctor_q_mean_outcome", window, False
            )
            constructor_current[f"ctor_q_top3_l{window}"] = _rolling(
                constructor_current_group, "ctor_q_top3_outcome", window, False
            )
        constructor_current["ctor_q_experience"] = constructor_current_group.cumcount().astype(float) + 1.0
        driver_circuit_state = qualifying[
            ["driverId", "circuitId", "date", "event_label"]
        ].copy()
        driver_circuit_current = driver_circuit_state.groupby(
            ["driverId", "circuitId"], sort=False
        )
        driver_circuit_state["driver_circuit_count"] = (
            driver_circuit_current.cumcount().astype(float) + 1.0
        )
        driver_circuit_state["driver_circuit_success"] = (
            driver_circuit_current["event_label"].cumsum().astype(float)
        )
        constructor_circuit_state = constructor_circuit_race.copy()
        constructor_circuit_state["ctor_circuit_count"] += constructor_circuit_state["circuit_entries"]
        constructor_circuit_state["ctor_circuit_success"] += constructor_circuit_state["circuit_successes"]
        self.qualifying = qualifying
        self.q_state = current
        self.constructor_q_state = constructor_current
        self.driver_circuit_state = driver_circuit_state
        self.constructor_circuit_state = constructor_circuit_state

    def _prepare_results(self) -> None:
        results = self.tables["results"].copy().sort_values(["date", "raceId", "positionOrder"])
        starters = results.groupby("raceId")["positionOrder"].transform("max").clip(lower=2)
        grid = results["grid"].where(results["grid"] > 0, starters + 1)
        results["grid_outcome"] = (grid - 1.0) / starters
        results["finish_outcome"] = (results["positionOrder"] - 1.0) / (starters - 1.0)
        results["dnf_outcome"] = results["position"].isna().astype(float)
        results["points_outcome"] = results["points"].astype(float)
        driver_group = results.groupby("driverId", sort=False)
        results["result_experience"] = driver_group.cumcount().astype(float) + 1.0
        for window in (1, 3, 5, 10):
            results[f"result_grid_l{window}"] = _rolling(driver_group, "grid_outcome", window, False)
            results[f"result_finish_l{window}"] = _rolling(driver_group, "finish_outcome", window, False)
        for window in (3, 5, 10):
            results[f"result_dnf_l{window}"] = _rolling(driver_group, "dnf_outcome", window, False)
            results[f"result_points_l{window}"] = _rolling(driver_group, "points_outcome", window, False)
        self.result_state = results
        constructor_race = results.groupby(
            ["date", "raceId", "constructorId"], sort=False
        ).agg(
            reliability_outcome=("dnf_outcome", "mean"),
            result_points=("points_outcome", "sum"),
        ).reset_index()
        race_points = constructor_race.groupby("raceId")["result_points"].transform("sum")
        constructor_race["points_share_outcome"] = constructor_race["result_points"] / race_points.replace(0, np.nan)
        constructor_race = constructor_race.sort_values(["date", "raceId", "constructorId"]).reset_index(drop=True)
        constructor_group = constructor_race.groupby("constructorId", sort=False)
        for window in (3, 5, 10):
            constructor_race[f"ctor_points_share_l{window}"] = _rolling(
                constructor_group, "points_share_outcome", window, False
            )
            constructor_race[f"ctor_reliability_l{window}"] = _rolling(
                constructor_group, "reliability_outcome", window, False
            )
        self.constructor_result_state = constructor_race

    def _prepare_standings(self) -> None:
        standings = self.tables["standings"].copy().sort_values(["date", "raceId", "position"])
        group = standings.groupby("driverId", sort=False)
        standings["standing_position"] = standings["position"].astype(float)
        standings["standing_points"] = standings["points"].astype(float)
        standings["standing_wins"] = standings["wins"].astype(float)
        standings["standing_trajectory"] = standings["standing_position"] - group["position"].shift(3)
        self.standing_state = standings
        constructor = self.tables["constructor_standings"].copy().sort_values(
            ["date", "raceId", "position"]
        )
        constructor_group = constructor.groupby("constructorId", sort=False)
        constructor["ctor_standing_position"] = constructor["position"].astype(float)
        constructor["ctor_standing_points"] = constructor["points"].astype(float)
        constructor["ctor_standing_wins"] = constructor["wins"].astype(float)
        constructor["ctor_standing_trajectory"] = (
            constructor["ctor_standing_position"] - constructor_group["position"].shift(3)
        )
        self.constructor_standing_state = constructor

    def _prepare_constructor_history(self) -> None:
        constructor_results = self.tables["constructor_results"].copy().sort_values(
            ["date", "raceId", "constructorId"]
        )
        race_points = constructor_results.groupby("raceId")["points"].transform("sum")
        constructor_results["official_points_share"] = (
            constructor_results["points"] / race_points.replace(0, np.nan)
        )
        group = constructor_results.groupby("constructorId", sort=False)
        for window in (3, 5, 10):
            constructor_results[f"official_points_share_l{window}"] = _rolling(
                group, "official_points_share", window, False
            )
        self.official_constructor_result_state = constructor_results

    def _merge_historical_state(self, frame: pd.DataFrame) -> pd.DataFrame:
        result_columns = [
            "result_experience",
            *[f"result_grid_l{window}" for window in (1, 3, 5, 10)],
            *[f"result_finish_l{window}" for window in (1, 3, 5, 10)],
            *[f"result_dnf_l{window}" for window in (3, 5, 10)],
            *[f"result_points_l{window}" for window in (3, 5, 10)],
        ]
        result = _asof(frame, self.result_state, "driverId", result_columns, False)
        standing_columns = [
            "standing_position",
            "standing_points",
            "standing_wins",
            "standing_trajectory",
        ]
        result = _asof(result, self.standing_state, "driverId", standing_columns, False)
        constructor_standing_columns = [
            "ctor_standing_position",
            "ctor_standing_points",
            "ctor_standing_wins",
            "ctor_standing_trajectory",
        ]
        result = _asof(
            result,
            self.constructor_standing_state,
            "constructorId",
            constructor_standing_columns,
            False,
        )
        constructor_result_columns = [
            *[f"ctor_points_share_l{window}" for window in (3, 5, 10)],
            *[f"ctor_reliability_l{window}" for window in (3, 5, 10)],
        ]
        result = _asof(
            result,
            self.constructor_result_state,
            "constructorId",
            constructor_result_columns,
            False,
        )
        official_columns = [f"official_points_share_l{window}" for window in (3, 5, 10)]
        official = _asof(
            result[["date", "constructorId"]],
            self.official_constructor_result_state,
            "constructorId",
            official_columns,
            False,
        )
        for window in (3, 5, 10):
            primary = result[f"ctor_points_share_l{window}"]
            result[f"ctor_points_share_l{window}"] = primary.fillna(
                official[f"official_points_share_l{window}"]
            )
        return result

    def _identity_features(self, frame: pd.DataFrame, event_dates: pd.Series) -> pd.DataFrame:
        result = frame.copy()
        driver_info = self.drivers[["driverId", "dob", "nationality"]].rename(
            columns={"nationality": "driver_nationality"}
        )
        constructor_info = self.constructors[["constructorId", "nationality"]].rename(
            columns={"nationality": "constructor_nationality"}
        )
        result = result.merge(driver_info, on="driverId", how="left")
        result = result.merge(constructor_info, on="constructorId", how="left")
        dates = pd.to_datetime(event_dates).reset_index(drop=True)
        result["age_years"] = (dates - pd.to_datetime(result["dob"])).dt.days / 365.25
        result["rookie"] = ((result["result_experience"].fillna(0) < 20) | (result["q_experience"].fillna(0) < 10)).astype(float)
        return result

    def _apply_circuit_metadata(self, frame: pd.DataFrame) -> pd.DataFrame:
        circuit_info = self.circuits[["circuitId", "lat", "lng", "alt", "country"]].rename(
            columns={
                "lat": "circuit_lat",
                "lng": "circuit_lng",
                "alt": "circuit_alt",
                "country": "circuit_country",
            }
        )
        result = frame.merge(circuit_info, on="circuitId", how="left")
        result["driver_home_circuit"] = (
            result["driver_nationality"].fillna("") == result["circuit_country"].fillna("_")
        ).astype(float)
        result["constructor_home_circuit"] = (
            result["constructor_nationality"].fillna("") == result["circuit_country"].fillna("_")
        ).astype(float)
        return result

    def _finish_events(self) -> None:
        events = self._merge_historical_state(self.qualifying.copy())
        events = self._identity_features(events, events["date"])
        events["driver_circuit_rate"] = (
            events["driver_circuit_success"] + 10.0 * events["q_top3_l10"].fillna(0.15)
        ) / (events["driver_circuit_count"] + 10.0)
        events["ctor_circuit_rate"] = (
            events["ctor_circuit_success"] + 10.0 * events["ctor_q_top3_l5"].fillna(0.15)
        ) / (events["ctor_circuit_count"] + 10.0)
        events = self._apply_circuit_metadata(events)
        events["event_year"] = events["date"].dt.year.astype(float)
        events["event_round"] = events["round"].astype(float)
        angle = 2.0 * np.pi * events["date"].dt.dayofyear / 365.25
        events["event_doy_sin"] = np.sin(angle)
        events["event_doy_cos"] = np.cos(angle)
        events = _relative(events, "raceId", RELATIVE_BASE)
        self.events = events

    def query_base(self, rows: pd.DataFrame) -> pd.DataFrame:
        base = rows[["date", "driverId"]].copy().reset_index(drop=True)
        base["date"] = pd.to_datetime(base["date"])
        q_columns = [
            "constructorId",
            "last_q_date",
            "q_experience",
            "constructor_tenure",
            "constructor_changed",
            *[f"q_pos_pct_l{window}" for window in WINDOWS],
            *[f"q_top3_l{window}" for window in WINDOWS],
            *[f"q_gap_l{window}" for window in (3, 5, 10)],
            *[f"q_h2h_l{window}" for window in (3, 5, 10)],
        ]
        base = _asof(base, self.q_state, "driverId", q_columns, True)
        base["inactivity_days"] = (base["date"] - base["last_q_date"]).dt.days.astype(float)
        result_columns = [
            "result_experience",
            *[f"result_grid_l{window}" for window in (1, 3, 5, 10)],
            *[f"result_finish_l{window}" for window in (1, 3, 5, 10)],
            *[f"result_dnf_l{window}" for window in (3, 5, 10)],
            *[f"result_points_l{window}" for window in (3, 5, 10)],
        ]
        base = _asof(base, self.result_state, "driverId", result_columns, True)
        base = _asof(
            base,
            self.standing_state,
            "driverId",
            ["standing_position", "standing_points", "standing_wins", "standing_trajectory"],
            True,
        )
        q_constructor_columns = [
            "ctor_q_experience",
            *[f"ctor_q_best_l{window}" for window in WINDOWS],
            *[f"ctor_q_mean_l{window}" for window in (3, 5, 10)],
            *[f"ctor_q_top3_l{window}" for window in (3, 5, 10)],
        ]
        base = _asof(base, self.constructor_q_state, "constructorId", q_constructor_columns, True)
        base = _asof(
            base,
            self.constructor_standing_state,
            "constructorId",
            [
                "ctor_standing_position",
                "ctor_standing_points",
                "ctor_standing_wins",
                "ctor_standing_trajectory",
            ],
            True,
        )
        base = _asof(
            base,
            self.constructor_result_state,
            "constructorId",
            [
                *[f"ctor_points_share_l{window}" for window in (3, 5, 10)],
                *[f"ctor_reliability_l{window}" for window in (3, 5, 10)],
            ],
            True,
        )
        official = _asof(
            base[["date", "constructorId"]],
            self.official_constructor_result_state,
            "constructorId",
            [f"official_points_share_l{window}" for window in (3, 5, 10)],
            True,
        )
        for window in (3, 5, 10):
            base[f"ctor_points_share_l{window}"] = base[f"ctor_points_share_l{window}"].fillna(
                official[f"official_points_share_l{window}"]
            )
        base = self._identity_features(base, base["date"])
        return base

    def event_candidates(
        self,
        base: pd.DataFrame,
        circuit_id: int,
        event_date: pd.Timestamp,
    ) -> pd.DataFrame:
        result = base.copy().reset_index(drop=True)
        result["circuitId"] = int(circuit_id)
        if circuit_id >= 0:
            driver_state = _asof(
                result[["date", "driverId", "circuitId"]],
                self.driver_circuit_state,
                ["driverId", "circuitId"],
                ["driver_circuit_count", "driver_circuit_success"],
                True,
            )
            constructor_state = _asof(
                result[["date", "constructorId", "circuitId"]],
                self.constructor_circuit_state,
                ["constructorId", "circuitId"],
                ["ctor_circuit_count", "ctor_circuit_success"],
                True,
            )
            result["driver_circuit_count"] = driver_state["driver_circuit_count"].fillna(0.0)
            result["driver_circuit_success"] = driver_state["driver_circuit_success"].fillna(0.0)
            result["ctor_circuit_count"] = constructor_state["ctor_circuit_count"].fillna(0.0)
            result["ctor_circuit_success"] = constructor_state["ctor_circuit_success"].fillna(0.0)
        else:
            result["driver_circuit_count"] = 0.0
            result["driver_circuit_success"] = 0.0
            result["ctor_circuit_count"] = 0.0
            result["ctor_circuit_success"] = 0.0
        result["driver_circuit_rate"] = (
            result["driver_circuit_success"] + 10.0 * result["q_top3_l10"].fillna(0.15)
        ) / (result["driver_circuit_count"] + 10.0)
        result["ctor_circuit_rate"] = (
            result["ctor_circuit_success"] + 10.0 * result["ctor_q_top3_l5"].fillna(0.15)
        ) / (result["ctor_circuit_count"] + 10.0)
        if circuit_id >= 0:
            result = self._apply_circuit_metadata(result)
        else:
            result["circuit_lat"] = np.nan
            result["circuit_lng"] = np.nan
            result["circuit_alt"] = np.nan
            result["circuit_country"] = ""
            result["driver_home_circuit"] = 0.0
            result["constructor_home_circuit"] = 0.0
        result["event_year"] = float(event_date.year)
        result["event_round"] = self.schedule(pd.Timestamp(base["date"].iloc[0]))["latest_round"] + 1.0
        angle = 2.0 * np.pi * event_date.dayofyear / 365.25
        result["event_doy_sin"] = math.sin(angle)
        result["event_doy_cos"] = math.cos(angle)
        result["_cohort"] = 0
        return _relative_single(result, RELATIVE_BASE)

    def schedule(self, origin: pd.Timestamp) -> dict:
        origin = pd.Timestamp(origin)
        if origin in self._schedule_cache:
            return self._schedule_cache[origin]
        available = self.races[self.races["date"] <= origin].copy()
        current = available[available["year"] == origin.year]
        latest_round = float(current["round"].max()) if len(current) else 0.0
        prior_totals = available[available["year"] < origin.year].groupby("year")["round"].max()
        typical_total = float(prior_totals.tail(3).mean()) if len(prior_totals) else 16.0
        phase = latest_round / max(typical_total, 1.0)
        recent_dates = available["date"].drop_duplicates().sort_values().tail(5)
        cadence = float(recent_dates.diff().dt.days.dropna().median()) if len(recent_dates) > 1 else 14.0
        counts = np.zeros(3, dtype=float)
        slot_votes: list[list[tuple[int, float, float]]] = [[], [], []]
        first_year = max(int(available["year"].min()) if len(available) else origin.year - 8, origin.year - 8)
        for year in range(first_year, origin.year):
            try:
                reference = origin.replace(year=year)
            except ValueError:
                reference = origin.replace(year=year, day=28)
            season = available[available["year"] == year].sort_values("date")
            future = season[(season["date"] > reference) & (season["date"] <= reference + pd.Timedelta(days=30))]
            if not len(future):
                continue
            age = origin.year - year
            weight = float(0.5 ** (age / 3.0))
            number = min(3, len(future))
            counts[number - 1] += weight
            for slot, (_, race) in enumerate(future.head(3).iterrows()):
                offset = float((race["date"] - reference).total_seconds() / 86400.0)
                slot_votes[slot].append((int(race["circuitId"]), weight, offset))
        if counts.sum() == 0:
            counts[:] = (0.25, 0.55, 0.20)
        else:
            counts += 0.05
        probabilities = counts / counts.sum()
        expected = float(np.dot(probabilities, np.arange(1, 4)))
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
        slots: list[list[tuple[int, float, float]]] = []
        circuit_entropies = []
        for slot, votes in enumerate(slot_votes):
            if not votes:
                slots.append([(-1, 1.0, min(28.0, 7.0 + 10.0 * slot))])
                circuit_entropies.append(0.0)
                continue
            totals: dict[int, float] = {}
            offsets: dict[int, float] = {}
            for circuit_id, weight, offset in votes:
                totals[circuit_id] = totals.get(circuit_id, 0.0) + weight
                offsets[circuit_id] = offsets.get(circuit_id, 0.0) + weight * offset
            ordered = sorted(totals, key=totals.get, reverse=True)[:3]
            normalizer = sum(totals[circuit] for circuit in ordered)
            mixture = [
                (
                    circuit,
                    totals[circuit] / normalizer,
                    offsets[circuit] / totals[circuit],
                )
                for circuit in ordered
            ]
            slots.append(mixture)
            weights = np.array([item[1] for item in mixture])
            circuit_entropies.append(float(-np.sum(weights * np.log(weights + 1e-12))))
        information = {
            "p_n": probabilities,
            "expected_n": expected,
            "entropy": entropy,
            "circuit_entropy": float(np.mean(circuit_entropies)),
            "latest_round": latest_round,
            "phase": phase,
            "cadence": cadence,
            "slots": slots,
        }
        self._schedule_cache[origin] = information
        return information

    def window_features(self, rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        base = self.query_base(rows)
        schedules = [self.schedule(date) for date in base["date"]]
        angle = 2.0 * np.pi * base["date"].dt.dayofyear / 365.25
        base["origin_doy_sin"] = np.sin(angle)
        base["origin_doy_cos"] = np.cos(angle)
        base["schedule_expected_n"] = [item["expected_n"] for item in schedules]
        base["schedule_entropy"] = [item["entropy"] for item in schedules]
        base["schedule_circuit_entropy"] = [item["circuit_entropy"] for item in schedules]
        base["schedule_latest_round"] = [item["latest_round"] for item in schedules]
        base["schedule_phase"] = [item["phase"] for item in schedules]
        base["schedule_cadence"] = [item["cadence"] for item in schedules]
        base["_origin_group"] = base["date"].astype("int64")
        base = _relative(base, "_origin_group", DIRECT_RELATIVE_BASE)
        return base, pd.DataFrame(_numeric(base, DIRECT_FEATURES), columns=DIRECT_FEATURES)


# Event models

class ConstantModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 1e-4, 1.0 - 1e-4))

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        probability = np.full(len(matrix), self.probability)
        return np.column_stack([1.0 - probability, probability])


def _fit_classifier(kind: str, matrix: np.ndarray, labels: np.ndarray, debug: bool):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return ConstantModel(float(labels.mean()) if len(labels) else 0.15)
    if kind == "lightgbm":
        model = LGBMClassifier(
            max_depth=4,
            num_leaves=15,
            n_estimators=50 if debug else 350,
            learning_rate=0.04,
            min_child_samples=30,
            reg_lambda=15.0,
            colsample_bytree=0.8,
            random_state=1337,
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(matrix, labels)
        return model
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.05, penalty="l2", max_iter=1000, random_state=1337),
            ),
        ]
    )
    model.fit(matrix, labels)
    return model


@dataclass
class EventBundle:
    model: object
    kind: str
    calibrator: object | None
    calibration: str
    replay: dict

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _numeric(frame, EVENT_FEATURES)
        base = _clip_probability(self.model.predict_proba(matrix)[:, 1])
        if self.calibrator is None:
            return base
        if self.calibration == "beta":
            calibration_matrix = np.column_stack([np.log(base), -np.log1p(-base)])
        else:
            calibration_matrix = _logit(base).reshape(-1, 1)
        return _clip_probability(self.calibrator.predict_proba(calibration_matrix)[:, 1])


def fit_event_bundle(events: pd.DataFrame, debug: bool) -> EventBundle:
    source = events.sort_values("date").reset_index(drop=True)
    if debug and len(source):
        boundary = source["date"].max() - pd.DateOffset(years=3)
        recent = source[source["date"] >= boundary]
        if len(recent) >= 400 and recent["event_label"].nunique() == 2:
            source = recent.reset_index(drop=True)
    dates = np.array(sorted(source["date"].drop_duplicates()))
    start = max(8, int(len(dates) * 0.38))
    chunks = [chunk for chunk in np.array_split(dates[start:], 2 if debug else 4) if len(chunk)]
    light_predictions = np.full(len(source), np.nan)
    logistic_predictions = np.full(len(source), np.nan)
    fold_ids = np.full(len(source), -1, dtype=int)
    light_scores = []
    logistic_scores = []
    matrix = _numeric(source, EVENT_FEATURES)
    labels = source["event_label"].to_numpy(dtype=int)
    for fold_id, chunk in enumerate(chunks):
        train_mask = source["date"] < chunk[0]
        validation_mask = source["date"].isin(chunk)
        if train_mask.sum() < 100 or validation_mask.sum() == 0:
            continue
        light = _fit_classifier("lightgbm", matrix[train_mask], labels[train_mask], debug)
        logistic = _fit_classifier("logistic", matrix[train_mask], labels[train_mask], debug)
        light_fold = _clip_probability(light.predict_proba(matrix[validation_mask])[:, 1])
        logistic_fold = _clip_probability(logistic.predict_proba(matrix[validation_mask])[:, 1])
        light_predictions[validation_mask] = light_fold
        logistic_predictions[validation_mask] = logistic_fold
        fold_ids[validation_mask] = fold_id
        light_scores.append(_safe_auc(labels[validation_mask], light_fold))
        logistic_scores.append(_safe_auc(labels[validation_mask], logistic_fold))
    light_mean = float(np.nanmean(light_scores)) if light_scores else 0.5
    logistic_mean = float(np.nanmean(logistic_scores)) if logistic_scores else 0.5
    kind = "lightgbm" if light_mean >= logistic_mean - 0.005 else "logistic"
    prequential = light_predictions if kind == "lightgbm" else logistic_predictions
    valid = np.isfinite(prequential)
    calibrator = None
    calibration = "none"
    if valid.sum() >= 100 and len(np.unique(labels[valid])) == 2:
        probabilities = _clip_probability(prequential[valid])
        calibration_labels = labels[valid]
        split = max(40, int(len(probabilities) * 0.6))
        split = min(split, len(probabilities) - 20)
        platt_trial = LogisticRegression(C=10.0, max_iter=500, random_state=1337)
        beta_trial = LogisticRegression(C=10.0, max_iter=500, random_state=1337)
        platt_x = _logit(probabilities).reshape(-1, 1)
        beta_x = np.column_stack([np.log(probabilities), -np.log1p(-probabilities)])
        platt_trial.fit(platt_x[:split], calibration_labels[:split])
        beta_trial.fit(beta_x[:split], calibration_labels[:split])
        platt_loss = -(
            calibration_labels[split:] * np.log(_clip_probability(platt_trial.predict_proba(platt_x[split:])[:, 1]))
            + (1 - calibration_labels[split:])
            * np.log1p(-_clip_probability(platt_trial.predict_proba(platt_x[split:])[:, 1]))
        )
        beta_loss = -(
            calibration_labels[split:] * np.log(_clip_probability(beta_trial.predict_proba(beta_x[split:])[:, 1]))
            + (1 - calibration_labels[split:])
            * np.log1p(-_clip_probability(beta_trial.predict_proba(beta_x[split:])[:, 1]))
        )
        improvement = platt_loss - beta_loss
        standard_error = float(np.std(improvement, ddof=1) / np.sqrt(max(1, len(improvement))))
        if float(np.mean(improvement)) > standard_error:
            calibration = "beta"
            calibration_matrix = beta_x
        else:
            calibration = "platt"
            calibration_matrix = platt_x
        calibrator = LogisticRegression(C=10.0, max_iter=500, random_state=1337)
        calibrator.fit(calibration_matrix, calibration_labels)
    final_model = _fit_classifier(kind, matrix, labels, debug)
    replay = {
        "kind": kind,
        "light_auc": round(light_mean, 6),
        "logistic_auc": round(logistic_mean, 6),
        "calibration": calibration,
        "rows": int(len(source)),
    }
    return EventBundle(final_model, kind, calibrator, calibration, replay)


# Structural aggregation

def structural_predictions(
    store: FeatureStore,
    rows: pd.DataFrame,
    event_bundle: EventBundle,
    debug: bool,
) -> pd.DataFrame:
    base = store.query_base(rows)
    output = pd.DataFrame(index=np.arange(len(base)), columns=SUMMARY_FEATURES, dtype=float)
    for origin, indices in base.groupby("date", sort=False).groups.items():
        cohort = base.loc[list(indices)].reset_index(drop=True)
        schedule = store.schedule(pd.Timestamp(origin))
        slots = schedule["slots"]
        if debug:
            event_date = pd.Timestamp(origin) + pd.Timedelta(days=15)
            candidates = store.event_candidates(cohort, -1, event_date)
            generic_probability = _clip_probability(event_bundle.predict(candidates))
            slot_probabilities = [generic_probability.copy() for _ in range(3)]
        else:
            slot_probabilities = []
            for mixture in slots[:3]:
                probability = np.zeros(len(cohort), dtype=float)
                for circuit_id, weight, offset in mixture:
                    event_date = pd.Timestamp(origin) + pd.Timedelta(days=float(offset))
                    candidates = store.event_candidates(cohort, int(circuit_id), event_date)
                    probability += float(weight) * event_bundle.predict(candidates)
                slot_probabilities.append(_clip_probability(probability))
        while len(slot_probabilities) < 3:
            slot_probabilities.append(slot_probabilities[-1].copy())
        slot_matrix = np.column_stack(slot_probabilities)
        p_n = np.asarray(schedule["p_n"], dtype=float)
        noisy = np.zeros(len(cohort), dtype=float)
        for number in range(1, 4):
            union = 1.0 - np.prod(1.0 - slot_matrix[:, :number], axis=1)
            noisy += p_n[number - 1] * union
        mean_probability = np.average(slot_matrix, axis=1)
        effective_count = 1.0 + 0.65 * (float(schedule["expected_n"]) - 1.0)
        conservative = 1.0 - np.power(1.0 - mean_probability, effective_count)
        destination = list(indices)
        output.loc[destination, "structural_noisy_or"] = noisy
        output.loc[destination, "structural_conservative"] = conservative
        output.loc[destination, "event_probability_max"] = slot_matrix.max(axis=1)
        output.loc[destination, "event_probability_mean"] = mean_probability
        output.loc[destination, "schedule_expected_n"] = schedule["expected_n"]
        output.loc[destination, "schedule_entropy"] = schedule["entropy"]
        output.loc[destination, "schedule_circuit_entropy"] = schedule["circuit_entropy"]
    return output.reset_index(drop=True)


# Exact labels

def make_table(qualifying: pd.DataFrame, timestamps: pd.Series) -> pd.DataFrame:
    timestamp_df = pd.DataFrame({"timestamp": pd.to_datetime(timestamps)})
    if len(timestamp_df) == 0:
        return pd.DataFrame(
            {
                "date": pd.Series(dtype="datetime64[ns]"),
                "driverId": pd.Series(dtype="int64"),
                "qualifying": pd.Series(dtype="int64"),
            }
        )
    frame = duckdb.sql(f"""
            SELECT
                t.timestamp as date,
                qu.driverId as driverId,
                CASE
                    WHEN MIN(qu.position) <= 3 THEN 1
                    ELSE 0
                END AS qualifying
            FROM
                timestamp_df t
            LEFT JOIN
                qualifying qu
            ON
                qu.date <= t.timestamp + INTERVAL '30 days'
                and qu.date > t.timestamp
            WHERE
                qu.driverId IN (
                    SELECT DISTINCT driverId
                    FROM qualifying
                    WHERE date > t.timestamp - INTERVAL '1 year'
                )
            GROUP BY t.timestamp, qu.driverId
        ;
        """).df()
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[ns]")
    frame["driverId"] = frame["driverId"].astype("int64")
    frame["qualifying"] = frame["qualifying"].astype("int64")
    return frame


def verify_official_labels(qualifying: pd.DataFrame, official: pd.DataFrame) -> dict:
    columns = ["date", "driverId", "qualifying"]
    expected = official[columns].copy()
    expected["date"] = pd.to_datetime(expected["date"]).astype("datetime64[ns]")
    expected["driverId"] = expected["driverId"].astype("int64")
    expected["qualifying"] = expected["qualifying"].astype("int64")
    expected = expected.sort_values(["date", "driverId"], kind="mergesort").reset_index(drop=True)
    actual = make_table(qualifying, pd.Series(sorted(expected["date"].unique())))
    actual = actual[columns].sort_values(["date", "driverId"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected, check_dtype=True, check_exact=True)
    return {
        "rows": int(len(actual)),
        "origins": int(actual["date"].nunique()),
        "exact": True,
    }


# Pseudo frame

def _pseudo_origins(official_dates: pd.Series, cutoff: pd.Timestamp) -> list[pd.Timestamp]:
    dates = pd.to_datetime(official_dates).drop_duplicates().sort_values()
    if len(dates) == 0:
        return []
    anchor = pd.Timestamp(dates.iloc[0])
    official = set(pd.Timestamp(value) for value in dates)
    latest = pd.Timestamp(cutoff) - pd.Timedelta(days=30)
    origins = []
    step = 0
    while anchor + pd.Timedelta(days=30 * step + PSEUDO_OFFSETS[0]) <= latest:
        for offset in PSEUDO_OFFSETS:
            origin = anchor + pd.Timedelta(days=30 * step + offset)
            if origin <= latest and origin not in official:
                origins.append(origin)
        step += 1
    return sorted(set(origins))


def _qualifying_prefix_hash(qualifying: pd.DataFrame, origin: pd.Timestamp) -> str:
    columns = ["date", "raceId", "driverId", "constructorId", "position"]
    prefix = qualifying.loc[
        pd.to_datetime(qualifying["date"]) <= origin + pd.Timedelta(days=30), columns
    ].sort_values(["date", "raceId", "driverId"], kind="mergesort")
    values = pd.util.hash_pandas_object(prefix, index=False).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(PSEUDO_CACHE_VERSION.encode())
    digest.update(origin.isoformat().encode())
    digest.update(values.tobytes())
    return digest.hexdigest()


def _register_pseudo_cache(shared: Path, cache_root: Path) -> None:
    artifacts = shared / "artifacts.json"
    lock_path = shared / "artifacts.json.lock"
    record = {
        "name": "generic_exp_5_lane1_exact_pseudo_windows_v1",
        "path": cache_root.relative_to(shared).as_posix(),
        "description": "Exact 30-day pseudo labels and origin-censored direct feature matrices keyed by origin and qualifying-prefix hash",
        "content_key": PSEUDO_CACHE_VERSION,
        "rebuild_hint": "Run main.py; missing origin files are rebuilt from the visible snapshot only",
    }
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = json.loads(artifacts.read_text()) if artifacts.exists() else []
        if not any(item.get("name") == record["name"] for item in existing):
            existing.append(record)
            temporary = shared / "artifacts.json.tmp.generic_exp_5_lane1"
            temporary.write_text(json.dumps(existing, indent=2) + "\n")
            temporary.replace(artifacts)
        fcntl.flock(lock, fcntl.LOCK_UN)


def build_pseudo_frame(
    store: FeatureStore,
    official: pd.DataFrame,
    cutoff: pd.Timestamp,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    started = time.time()
    origins = _pseudo_origins(official["date"], cutoff)
    labels = make_table(store.tables["qualifying"], pd.Series(origins))
    labels = labels.sort_values(["date", "driverId"], kind="mergesort").reset_index(drop=True)
    if len(labels) == 0:
        return labels, np.empty((0, len(DIRECT_FEATURES))), np.empty(0), {
            "origins": 0,
            "rows": 0,
            "hits": 0,
            "misses": 0,
            "seconds": round(time.time() - started, 3),
        }
    shared_value = os.environ.get("KAPSO_SHARED_CACHE_DIR", "")
    cache_root = None
    if shared_value:
        shared = Path(shared_value)
        cache_root = shared / "generic_exp_5_lane1_exact_pseudo_v1"
        cache_root.mkdir(parents=True, exist_ok=True)
        _register_pseudo_cache(shared, cache_root)
    matrices: dict[pd.Timestamp, np.ndarray] = {}
    paths: dict[pd.Timestamp, Path] = {}
    missing = []
    hits = 0
    for origin, group in labels.groupby("date", sort=True):
        origin = pd.Timestamp(origin)
        path = None
        if cache_root is not None:
            path = cache_root / f"origin_{_qualifying_prefix_hash(store.tables['qualifying'], origin)}.npz"
            paths[origin] = path
        loaded = False
        if path is not None and path.exists():
            with np.load(path, allow_pickle=False) as artifact:
                driver_id = artifact["driver_id"].astype("int64")
                label = artifact["label"].astype("int64")
                matrix = artifact["matrix"].astype("float64")
            if (
                np.array_equal(driver_id, group["driverId"].to_numpy(dtype="int64"))
                and np.array_equal(label, group["qualifying"].to_numpy(dtype="int64"))
                and matrix.shape == (len(group), len(DIRECT_FEATURES))
            ):
                matrices[origin] = matrix
                loaded = True
                hits += 1
        if not loaded:
            missing.append(group)
    if missing:
        missing_rows = pd.concat(missing, ignore_index=True)
        missing_rows = missing_rows.sort_values(["date", "driverId"], kind="mergesort").reset_index(drop=True)
        _, missing_features = store.window_features(missing_rows)
        missing_matrix = missing_features.to_numpy(dtype=np.float64)
        for origin, indices in missing_rows.groupby("date", sort=True).groups.items():
            origin = pd.Timestamp(origin)
            group = missing_rows.loc[list(indices)]
            matrix = missing_matrix[np.asarray(list(indices), dtype=int)]
            matrices[origin] = matrix
            path = paths.get(origin)
            if path is not None:
                temporary = path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    np.savez_compressed(
                        stream,
                        driver_id=group["driverId"].to_numpy(dtype="int64"),
                        label=group["qualifying"].to_numpy(dtype="int64"),
                        matrix=matrix,
                    )
                temporary.replace(path)
    matrix = np.vstack([matrices[pd.Timestamp(origin)] for origin in labels["date"].drop_duplicates()])
    session_dates = pd.to_datetime(store.tables["qualifying"]["date"]).drop_duplicates().sort_values()
    coverage = {
        session: sum(origin < session <= origin + pd.Timedelta(days=30) for origin in origins)
        for session in session_dates
        if origins and origins[0] < session <= origins[-1] + pd.Timedelta(days=30)
    }
    divisors = {}
    for origin in pd.to_datetime(labels["date"].drop_duplicates()):
        covered = [
            count
            for session, count in coverage.items()
            if origin < session <= origin + pd.Timedelta(days=30)
        ]
        divisors[pd.Timestamp(origin)] = float(np.mean(covered)) if covered else 1.0
    weights = np.array(
        [PSEUDO_WEIGHT / max(1.0, divisors[pd.Timestamp(origin)]) for origin in labels["date"]],
        dtype=np.float64,
    )
    diagnostics = {
        "origins": int(labels["date"].nunique()),
        "grid_origins": int(len(origins)),
        "rows": int(len(labels)),
        "positive_rate": round(float(labels["qualifying"].mean()), 6),
        "mean_overlap": round(float(np.mean([PSEUDO_WEIGHT / value for value in weights])), 6),
        "hits": int(hits),
        "misses": int(len(missing)),
        "seconds": round(time.time() - started, 3),
    }
    return labels, matrix, weights, diagnostics


# Candidate models

def _fit_window_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
    kind: str = "logistic",
    debug: bool = False,
):
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return ConstantModel(float(labels.mean()) if len(labels) else 0.17)
    if kind == "lightgbm":
        model = LGBMClassifier(
            max_depth=4,
            num_leaves=15,
            n_estimators=50 if debug else 350,
            learning_rate=0.03,
            min_child_samples=40,
            reg_lambda=15.0,
            colsample_bytree=0.8,
            random_state=1337,
            n_jobs=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
            verbosity=-1,
        )
        model.fit(matrix, labels, sample_weight=sample_weight)
        return model
    model = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(C=0.01, penalty="l2", max_iter=1000, random_state=1337),
            ),
        ]
    )
    fit_parameters = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(matrix, labels, **fit_parameters)
    return model


def _combined_training(
    official_matrix: np.ndarray,
    official_labels: np.ndarray,
    official_dates: pd.Series,
    pseudo_matrix: np.ndarray,
    pseudo_labels: np.ndarray,
    pseudo_dates: pd.Series,
    pseudo_weights: np.ndarray,
    cutoff: pd.Timestamp,
    recency: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.vstack([official_matrix, pseudo_matrix]) if len(pseudo_matrix) else official_matrix.copy()
    labels = np.concatenate([official_labels, pseudo_labels]) if len(pseudo_labels) else official_labels.copy()
    weights = np.concatenate([np.ones(len(official_labels)), pseudo_weights])
    dates = pd.concat(
        [pd.Series(pd.to_datetime(official_dates)), pd.Series(pd.to_datetime(pseudo_dates))],
        ignore_index=True,
    )
    if recency:
        age = (pd.Timestamp(cutoff) - dates).dt.total_seconds().to_numpy() / 86400.0
        weights *= np.power(0.5, np.maximum(age, 0.0) / HALF_LIFE_DAYS)
    weights /= float(np.mean(weights))
    return matrix, labels, weights


def _origin_count_frame(
    rows: pd.DataFrame,
    base: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:
    metadata = base.assign(_date=pd.to_datetime(rows["date"]).to_numpy()).groupby(
        "_date", sort=True
    ).agg(
        schedule_expected_n=("schedule_expected_n", "first"),
        cohort_size=("driverId", "size"),
        schedule_entropy=("schedule_entropy", "first"),
        schedule_phase=("schedule_phase", "first"),
    ).reset_index(names="date")
    counts = reference.groupby("date", sort=True)["qualifying"].sum()
    historical = []
    for origin in metadata["date"]:
        available = counts[counts.index <= pd.Timestamp(origin) - pd.Timedelta(days=30)]
        historical.append(float(available.mean()) if len(available) else 3.0)
    metadata["historical_union_count"] = historical
    return metadata


def _count_offset(
    predictions: np.ndarray,
    fit_rows: pd.DataFrame,
    fit_base: pd.DataFrame,
    target_rows: pd.DataFrame,
    target_base: pd.DataFrame,
) -> np.ndarray:
    columns = [
        "schedule_expected_n",
        "cohort_size",
        "schedule_entropy",
        "schedule_phase",
        "historical_union_count",
    ]
    fit_frame = _origin_count_frame(fit_rows, fit_base, fit_rows)
    targets = fit_rows.groupby("date", sort=True)["qualifying"].sum().reindex(fit_frame["date"])
    target_frame = _origin_count_frame(target_rows, target_base, fit_rows)
    if len(fit_frame) >= 8 and targets.nunique() > 1:
        model = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
                ("model", PoissonRegressor(alpha=1.0, max_iter=500)),
            ]
        )
        model.fit(fit_frame[columns].to_numpy(dtype=float), targets.to_numpy(dtype=float))
        expected = model.predict(target_frame[columns].to_numpy(dtype=float))
    else:
        expected = np.full(len(target_frame), float(targets.mean()) if len(targets) else 3.0)
    result = _logit(predictions)
    dates = pd.to_datetime(target_rows["date"]).reset_index(drop=True)
    for origin, expected_count in zip(target_frame["date"], expected):
        mask = (dates == pd.Timestamp(origin)).to_numpy()
        rate = np.clip(float(expected_count) / max(1, int(mask.sum())), 1e-6, 1.0 - 1e-6)
        target_logit = math.log(rate / (1.0 - rate))
        result[mask] += 0.5 * (target_logit - float(np.mean(result[mask])))
    return _clip_probability(1.0 / (1.0 + np.exp(-np.clip(result, -30.0, 30.0))))


# Replay selection

def _folds(dates: pd.Series) -> list[np.ndarray]:
    unique_dates = np.array(sorted(pd.to_datetime(dates).drop_duplicates()))
    start = max(8, int(len(unique_dates) * 0.36))
    return [chunk for chunk in np.array_split(unique_dates[start:], 4) if len(chunk)]


def _paired_bootstrap(
    records: pd.DataFrame,
    candidate: str,
    baseline: str = "champion",
) -> float:
    rng = np.random.default_rng(1337)
    dates = np.array(sorted(records["date"].unique()))
    differences = []
    for _ in range(100):
        sampled = rng.choice(dates, size=len(dates), replace=True)
        indices = np.concatenate(
            [np.flatnonzero(records["date"].to_numpy() == origin) for origin in sampled]
        )
        labels = records["label"].to_numpy(dtype=int)[indices]
        candidate_auc = _safe_auc(labels, records[candidate].to_numpy(dtype=float)[indices])
        baseline_auc = _safe_auc(labels, records[baseline].to_numpy(dtype=float)[indices])
        if np.isfinite(candidate_auc) and np.isfinite(baseline_auc):
            differences.append(candidate_auc - baseline_auc)
    return float(np.std(differences, ddof=1)) if len(differences) > 1 else float("inf")


def _candidate_diagnostics(records: pd.DataFrame) -> tuple[str, dict]:
    names = [
        "champion",
        "pseudo_logistic",
        "pseudo_recency",
        "pseudo_lightgbm",
        "pseudo_blend",
        "pseudo_count",
    ]
    fold_ids = sorted(records["fold"].unique())
    scores = {}
    champion_folds = []
    for fold_id in fold_ids:
        mask = records["fold"] == fold_id
        champion_folds.append(
            _safe_auc(records.loc[mask, "label"].to_numpy(), records.loc[mask, "champion"].to_numpy())
        )
    champion_pooled = _safe_auc(records["label"].to_numpy(), records["champion"].to_numpy())
    eligible = []
    for name in names:
        fold_scores = []
        for fold_id in fold_ids:
            mask = records["fold"] == fold_id
            fold_scores.append(
                _safe_auc(records.loc[mask, "label"].to_numpy(), records.loc[mask, name].to_numpy())
            )
        pooled = _safe_auc(records["label"].to_numpy(), records[name].to_numpy())
        standard_error = 0.0 if name == "champion" else _paired_bootstrap(records, name)
        wins = int(sum(value > base for value, base in zip(fold_scores, champion_folds)))
        delta = float(pooled - champion_pooled)
        passes = name != "champion" and wins >= 3 and delta > standard_error
        if passes:
            eligible.append(name)
        scores[name] = {
            "folds": [None if not np.isfinite(value) else round(float(value), 6) for value in fold_scores],
            "mean": round(float(np.nanmean(fold_scores)), 6),
            "pooled": round(float(pooled), 6),
            "delta": round(delta, 6),
            "bootstrap_se": round(float(standard_error), 6),
            "wins": wins,
            "passes": bool(passes),
        }
    selected = "champion"
    if eligible:
        best = max(eligible, key=lambda name: scores[name]["mean"])
        best_mean = scores[best]["mean"]
        for name in names[1:]:
            if name in eligible and best_mean - scores[name]["mean"] <= scores[name]["bootstrap_se"]:
                selected = name
                break
        if selected == "champion":
            selected = best
    seasons = {}
    for year in sorted(records["date"].dt.year.unique()):
        mask = records["date"].dt.year != year
        baseline = _safe_auc(records.loc[mask, "label"].to_numpy(), records.loc[mask, "champion"].to_numpy())
        candidate = _safe_auc(records.loc[mask, "label"].to_numpy(), records.loc[mask, selected].to_numpy())
        seasons[str(int(year))] = {
            "n": int(mask.sum()),
            "delta": None if not np.isfinite(candidate - baseline) else round(float(candidate - baseline), 6),
        }
    return selected, {"scores": scores, "leave_one_season_out": seasons}


def predict_pipeline(
    store: FeatureStore,
    training: pd.DataFrame,
    target: pd.DataFrame,
    debug: bool,
    event_cutoff: pd.Timestamp,
) -> tuple[np.ndarray, dict]:
    training = training.sort_values(["date", "driverId"], kind="mergesort").reset_index(drop=True)
    target = target.reset_index(drop=True)
    label_check = verify_official_labels(store.tables["qualifying"], training)
    training_base, training_features = store.window_features(training)
    target_base, target_features = store.window_features(target)
    pseudo, pseudo_matrix, pseudo_weights, pseudo_diagnostics = build_pseudo_frame(
        store, training, pd.Timestamp(event_cutoff)
    )
    labels = training["qualifying"].to_numpy(dtype=int)
    official_matrix = training_features.to_numpy(dtype=float)
    target_matrix = target_features.to_numpy(dtype=float)
    pseudo_labels = pseudo["qualifying"].to_numpy(dtype=int)
    replay_records = []
    for fold_id, fold_dates in enumerate(_folds(training["date"])):
        fold_start = pd.Timestamp(fold_dates[0])
        validation_mask = training["date"].isin(fold_dates).to_numpy()
        official_fit_mask = (training["date"] <= fold_start - pd.Timedelta(days=30)).to_numpy()
        pseudo_fit_mask = (
            pd.to_datetime(pseudo["date"]) + pd.Timedelta(days=30) <= fold_start
        ).to_numpy()
        if official_fit_mask.sum() < 100 or validation_mask.sum() == 0:
            continue
        fit_matrix = official_matrix[official_fit_mask]
        fit_labels = labels[official_fit_mask]
        validation_matrix = official_matrix[validation_mask]
        uniform_matrix, uniform_labels, uniform_weights = _combined_training(
            fit_matrix,
            fit_labels,
            training.loc[official_fit_mask, "date"],
            pseudo_matrix[pseudo_fit_mask],
            pseudo_labels[pseudo_fit_mask],
            pseudo.loc[pseudo_fit_mask, "date"],
            pseudo_weights[pseudo_fit_mask],
            fold_start,
            False,
        )
        recency_matrix, recency_labels, recency_weights = _combined_training(
            fit_matrix,
            fit_labels,
            training.loc[official_fit_mask, "date"],
            pseudo_matrix[pseudo_fit_mask],
            pseudo_labels[pseudo_fit_mask],
            pseudo.loc[pseudo_fit_mask, "date"],
            pseudo_weights[pseudo_fit_mask],
            fold_start,
            True,
        )
        champion_model = _fit_window_model(fit_matrix, fit_labels)
        uniform_model = _fit_window_model(uniform_matrix, uniform_labels, uniform_weights)
        recency_model = _fit_window_model(recency_matrix, recency_labels, recency_weights)
        tree_model = _fit_window_model(
            uniform_matrix, uniform_labels, uniform_weights, "lightgbm", debug
        )
        champion = _clip_probability(champion_model.predict_proba(validation_matrix)[:, 1])
        uniform = _clip_probability(uniform_model.predict_proba(validation_matrix)[:, 1])
        recency = _clip_probability(recency_model.predict_proba(validation_matrix)[:, 1])
        tree = _clip_probability(tree_model.predict_proba(validation_matrix)[:, 1])
        count = _count_offset(
            uniform,
            training.loc[official_fit_mask].reset_index(drop=True),
            training_base.loc[official_fit_mask].reset_index(drop=True),
            training.loc[validation_mask].reset_index(drop=True),
            training_base.loc[validation_mask].reset_index(drop=True),
        )
        record = pd.DataFrame(
            {
                "date": pd.to_datetime(training.loc[validation_mask, "date"]).to_numpy(),
                "fold": fold_id,
                "label": labels[validation_mask],
                "history": training_base.loc[validation_mask, "q_experience"].to_numpy(dtype=float),
                "schedule_expected_n": training_base.loc[
                    validation_mask, "schedule_expected_n"
                ].to_numpy(dtype=float),
                "champion": champion,
                "pseudo_logistic": uniform,
                "pseudo_recency": recency,
                "pseudo_lightgbm": tree,
                "pseudo_blend": _logit_blend(uniform, tree, 0.5),
                "pseudo_count": count,
            }
        )
        replay_records.append(record)
    if replay_records:
        records = pd.concat(replay_records, ignore_index=True)
        selected, replay_diagnostics = _candidate_diagnostics(records)
    else:
        records = pd.DataFrame()
        selected = "champion"
        replay_diagnostics = {"scores": {}, "leave_one_season_out": {}}
    uniform_matrix, uniform_labels, uniform_weights = _combined_training(
        official_matrix,
        labels,
        training["date"],
        pseudo_matrix,
        pseudo_labels,
        pseudo["date"],
        pseudo_weights,
        pd.Timestamp(event_cutoff),
        False,
    )
    recency_matrix, recency_labels, recency_weights = _combined_training(
        official_matrix,
        labels,
        training["date"],
        pseudo_matrix,
        pseudo_labels,
        pseudo["date"],
        pseudo_weights,
        pd.Timestamp(event_cutoff),
        True,
    )
    champion_final = None
    uniform_final = None
    tree_final = None
    if selected == "champion":
        model = _fit_window_model(official_matrix, labels)
        prediction = model.predict_proba(target_matrix)[:, 1]
    elif selected == "pseudo_recency":
        model = _fit_window_model(recency_matrix, recency_labels, recency_weights)
        prediction = model.predict_proba(target_matrix)[:, 1]
    else:
        model = _fit_window_model(uniform_matrix, uniform_labels, uniform_weights)
        uniform_final = _clip_probability(model.predict_proba(target_matrix)[:, 1])
        if selected == "pseudo_logistic":
            prediction = uniform_final
        elif selected == "pseudo_count":
            prediction = _count_offset(
                uniform_final, training, training_base, target, target_base
            )
        else:
            tree_model = _fit_window_model(
                uniform_matrix, uniform_labels, uniform_weights, "lightgbm", debug
            )
            tree_final = _clip_probability(tree_model.predict_proba(target_matrix)[:, 1])
            prediction = (
                tree_final
                if selected == "pseudo_lightgbm"
                else _logit_blend(uniform_final, tree_final, 0.5)
            )
    slices = {}
    if len(records):
        for name, mask in {
            "sparse": records["history"].to_numpy() < 10,
            "established": records["history"].to_numpy() >= 10,
            "one_or_two": records["schedule_expected_n"].to_numpy() < 2.5,
            "three": records["schedule_expected_n"].to_numpy() >= 2.5,
        }.items():
            score = _safe_auc(
                records.loc[mask, "label"].to_numpy(), records.loc[mask, selected].to_numpy()
            )
            slices[name] = {
                "n": int(mask.sum()),
                "auc": None if not np.isfinite(score) else round(float(score), 6),
            }
    diagnostics = {
        "selected": selected,
        "label_check": label_check,
        "pseudo": pseudo_diagnostics,
        "candidate_auc": replay_diagnostics["scores"],
        "leave_one_season_out": replay_diagnostics["leave_one_season_out"],
        "slices": slices,
        "folds": int(records["fold"].nunique()) if len(records) else 0,
    }
    return _clip_probability(prediction), diagnostics


# Loading

def load_snapshot(cache_root: Path, dataset_name: str, task_name: str):
    dataset_root = cache_root / dataset_name
    database = {
        name: pd.read_parquet(dataset_root / "db" / f"{name}.parquet")
        for name in (
            "circuits",
            "constructor_results",
            "constructor_standings",
            "constructors",
            "drivers",
            "qualifying",
            "races",
            "results",
            "standings",
        )
    }
    task_root = dataset_root / "tasks" / task_name
    splits = {}
    for split in ("train", "val", "test"):
        path = task_root / f"{split}.parquet"
        splits[split] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        if "date" in splits[split]:
            splits[split]["date"] = pd.to_datetime(splits[split]["date"])
    return database, splits


def diagnostics_json(diagnostics: dict) -> str:
    return json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
