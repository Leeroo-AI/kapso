from __future__ import annotations

import fcntl
import json
import math
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


CALENDAR_FEATURES = [
    "calendar_count_1",
    "calendar_count_2",
    "calendar_count_3",
    "calendar_count_5",
    "calendar_count_10",
    "calendar_exp_5",
    "calendar_last_flag",
    "calendar_streak",
    "calendar_gap",
    "calendar_alternating",
    "calendar_last_round",
    "calendar_expected_round",
    "calendar_round_std_5",
    "calendar_round_trend",
    "calendar_last_doy",
    "calendar_doy_std_5",
    "circuit_country_code",
    "circuit_lat",
    "circuit_lng",
    "circuit_alt",
    "circuit_alt_missing",
    "active_same_country",
    "circuit_years_since_first",
    "circuit_total_hosts",
    "calendar_persistence_score",
]


DRIVER_FEATURES = [
    "driver_age",
    "driver_nationality_code",
    "driver_career_years",
    "driver_total_starts",
    "driver_starts_1",
    "driver_starts_2",
    "driver_starts_3",
    "driver_starts_5",
    "driver_recency_days",
    "driver_points_per_start_1",
    "driver_points_per_start_3",
    "driver_position_order_1",
    "driver_position_order_3",
    "driver_finish_rate_3",
    "driver_grid_3",
    "driver_laps_3",
    "driver_status_mean_3",
    "driver_status_diversity_3",
    "driver_points_trend",
    "driver_starts_trend",
    "driver_round_mean_3",
    "driver_round_min_3",
    "driver_round_max_3",
    "driver_round_early_3",
    "driver_round_late_3",
    "driver_last_season_ratio",
    "standings_latest_position",
    "standings_latest_points",
    "standings_latest_wins",
    "standings_position_trajectory",
    "standings_points_trajectory",
    "qualifying_starts_3",
    "qualifying_position_3",
    "qualifying_recency_days",
    "latest_constructor_id",
    "constructor_tenure_years",
    "constructor_recent_starts",
    "constructor_recent_points",
    "constructor_latest_standing",
    "constructor_latest_points",
    "constructor_latest_wins",
    "constructor_viability_trend",
    "constructor_nationality_code",
    "constructor_driver_nat_match",
    "teammate_continuity",
    "episode_year",
]


PAIR_FEATURES = [
    "pair_starts_total",
    "pair_starts_1",
    "pair_starts_3",
    "pair_starts_5",
    "pair_recency_days",
    "pair_last_season_attendance",
    "pair_position_order",
    "pair_points_per_start",
    "pair_finish_rate",
    "pair_grid",
    "pair_laps",
    "pair_status_mean",
    "pair_qualifying_starts",
    "pair_qualifying_position",
    "constructor_circuit_starts",
    "constructor_circuit_recent_starts",
    "constructor_circuit_points",
    "nationality_country_match",
    "round_profile_distance",
    "round_early_affinity",
    "round_late_affinity",
]


FEATURE_NAMES = CALENDAR_FEATURES + DRIVER_FEATURES + PAIR_FEATURES


NATIONALITY_COUNTRY = {
    "american": {"usa", "united states"},
    "argentine": {"argentina"},
    "australian": {"australia"},
    "austrian": {"austria"},
    "belgian": {"belgium"},
    "brazilian": {"brazil"},
    "british": {"uk", "united kingdom"},
    "canadian": {"canada"},
    "chilean": {"chile"},
    "chinese": {"china"},
    "colombian": {"colombia"},
    "dutch": {"netherlands"},
    "finnish": {"finland"},
    "french": {"france"},
    "german": {"germany"},
    "hungarian": {"hungary"},
    "indian": {"india"},
    "irish": {"ireland"},
    "italian": {"italy"},
    "japanese": {"japan"},
    "malaysian": {"malaysia"},
    "mexican": {"mexico"},
    "monegasque": {"monaco"},
    "new zealander": {"new zealand"},
    "portuguese": {"portugal"},
    "russian": {"russia"},
    "south african": {"south africa"},
    "spanish": {"spain"},
    "swedish": {"sweden"},
    "swiss": {"switzerland"},
    "thai": {"thailand"},
    "venezuelan": {"venezuela"},
}


@dataclass
class PreparedData:
    circuits: pd.DataFrame
    drivers: pd.DataFrame
    constructors: pd.DataFrame
    races: pd.DataFrame
    results: pd.DataFrame
    standings: pd.DataFrame
    qualifying: pd.DataFrame
    constructor_results: pd.DataFrame
    constructor_standings: pd.DataFrame


@dataclass
class EpisodeMatrix:
    features: np.ndarray
    labels: np.ndarray
    host_labels: np.ndarray
    group_dates: np.ndarray
    driver_ids: np.ndarray
    label_sizes: np.ndarray

    @property
    def num_groups(self) -> int:
        return len(self.group_dates)


@dataclass
class GenerativeScores:
    p_host: np.ndarray
    p_attend: np.ndarray

    @property
    def product(self) -> np.ndarray:
        return self.p_host * self.p_attend


def prepare_data(db) -> PreparedData:
    circuits = db.table_dict["circuits"].df.copy()
    drivers = db.table_dict["drivers"].df.copy()
    constructors = db.table_dict["constructors"].df.copy()
    races = db.table_dict["races"].df.copy()
    results = db.table_dict["results"].df.copy()
    standings = db.table_dict["standings"].df.copy()
    qualifying = db.table_dict["qualifying"].df.copy()
    constructor_results = db.table_dict["constructor_results"].df.copy()
    constructor_standings = db.table_dict["constructor_standings"].df.copy()
    circuits["_country_code"] = pd.Categorical(circuits["country"].fillna("")).codes
    drivers["_nationality_code"] = pd.Categorical(drivers["nationality"].fillna("")).codes
    constructors["_nationality_code"] = pd.Categorical(constructors["nationality"].fillna("")).codes
    races["_year"] = races["date"].dt.year.astype(int)
    season_max_round = races.groupby("_year")["round"].transform("max").clip(lower=1)
    races["_round_norm"] = races["round"] / season_max_round
    races["_doy_norm"] = races["date"].dt.dayofyear / 366.0
    race_keys = races[["raceId", "circuitId", "_year", "_round_norm", "_doy_norm"]]
    results = results.merge(race_keys, on="raceId", how="left", validate="many_to_one")
    qualifying = qualifying.merge(race_keys, on="raceId", how="left", validate="many_to_one")
    constructor_results = constructor_results.merge(race_keys, on="raceId", how="left", validate="many_to_one")
    constructor_standings = constructor_standings.merge(race_keys, on="raceId", how="left", validate="many_to_one")
    standings["_year"] = standings["date"].dt.year.astype(int)
    return PreparedData(
        circuits=circuits,
        drivers=drivers,
        constructors=constructors,
        races=races,
        results=results,
        standings=standings,
        qualifying=qualifying,
        constructor_results=constructor_results,
        constructor_standings=constructor_standings,
    )


def official_episodes(table) -> pd.DataFrame:
    frame = table.df[["date", "driverId", "circuitId"]].copy()
    frame = frame.rename(columns={"circuitId": "target"})
    frame["target"] = frame["target"].map(lambda values: np.asarray(values, dtype=np.int64))
    return frame.reset_index(drop=True)


def input_rows(table) -> pd.DataFrame:
    return table.df[["date", "driverId"]].copy().reset_index(drop=True)


def reconstruct_episodes(data: PreparedData, years: list[int]) -> pd.DataFrame:
    frames = []
    for year in years:
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        end = cutoff + pd.Timedelta(days=365)
        future_races = data.races[(data.races["date"] > cutoff) & (data.races["date"] <= end)][["raceId", "circuitId"]]
        interactions = data.results[["raceId", "driverId"]].merge(future_races, on="raceId", how="inner")
        grouped = interactions.groupby("driverId", sort=True)["circuitId"].apply(lambda values: np.sort(pd.unique(values)).astype(np.int64))
        frame = grouped.reset_index().rename(columns={"circuitId": "target"})
        frame.insert(0, "date", cutoff)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["date", "driverId", "target"])
    return pd.concat(frames, ignore_index=True)


def calendar_features(cutoff: pd.Timestamp, data: PreparedData) -> pd.DataFrame:
    values = np.zeros((77, len(CALENDAR_FEATURES)), dtype=np.float64)
    column = {name: idx for idx, name in enumerate(CALENDAR_FEATURES)}
    history = data.races[data.races["date"] <= cutoff]
    cutoff_year = cutoff.year
    by_circuit = {int(key): frame for key, frame in history.groupby("circuitId", sort=False)}
    countries = data.circuits.set_index("circuitId")["country"].fillna("").astype(str)
    last_flags = np.zeros(77, dtype=np.float64)
    for circuit_id in range(77):
        frame = by_circuit.get(circuit_id)
        if frame is None or frame.empty:
            continue
        seasons = np.sort(frame["_year"].dropna().astype(int).unique())
        ages = cutoff_year - seasons
        ages = ages[ages >= 1]
        season_set = set(int(value) for value in seasons)
        for window in (1, 2, 3, 5, 10):
            values[circuit_id, column[f"calendar_count_{window}"]] = float(np.sum(ages <= window))
        exp_frequency = float(np.sum(0.5 ** ((ages[ages <= 5] - 1) / 5.0)))
        last_flag = float((cutoff_year - 1) in season_set)
        last_flags[circuit_id] = last_flag
        streak = 0
        while (cutoff_year - 1 - streak) in season_set:
            streak += 1
        gap = float(np.min(ages)) if len(ages) else 50.0
        alternating = float(
            (cutoff_year - 1) in season_set
            and (cutoff_year - 2) not in season_set
            and (cutoff_year - 3) in season_set
        )
        recent = frame[(frame["_year"] >= cutoff_year - 5) & (frame["_year"] < cutoff_year)].sort_values("date")
        latest = frame.sort_values("date").iloc[-1]
        hosted_seasons = np.sort(recent["_year"].unique())[-2:]
        recent_rounds = recent[recent["_year"].isin(hosted_seasons)]["_round_norm"]
        expected_round = float(recent_rounds.median()) if len(recent_rounds) else float(latest["_round_norm"])
        round_std = float(recent["_round_norm"].std(ddof=0)) if len(recent) > 1 else 0.0
        last_year_round = frame.loc[frame["_year"] == cutoff_year - 1, "_round_norm"]
        previous_round = frame.loc[frame["_year"] == cutoff_year - 2, "_round_norm"]
        round_trend = float(last_year_round.mean() - previous_round.mean()) if len(last_year_round) and len(previous_round) else 0.0
        doy_std = float(recent["_doy_norm"].std(ddof=0)) if len(recent) > 1 else 0.0
        persistence = exp_frequency + 0.8 * last_flag + 0.15 * min(streak, 4) + 0.08 * np.sum(ages <= 5) - 0.025 * min(gap, 20)
        values[circuit_id, column["calendar_exp_5"]] = exp_frequency
        values[circuit_id, column["calendar_last_flag"]] = last_flag
        values[circuit_id, column["calendar_streak"]] = streak
        values[circuit_id, column["calendar_gap"]] = gap
        values[circuit_id, column["calendar_alternating"]] = alternating
        values[circuit_id, column["calendar_last_round"]] = float(latest["_round_norm"])
        values[circuit_id, column["calendar_expected_round"]] = expected_round
        values[circuit_id, column["calendar_round_std_5"]] = round_std
        values[circuit_id, column["calendar_round_trend"]] = round_trend
        values[circuit_id, column["calendar_last_doy"]] = float(latest["_doy_norm"])
        values[circuit_id, column["calendar_doy_std_5"]] = doy_std
        values[circuit_id, column["circuit_years_since_first"]] = float(cutoff_year - int(seasons.min()))
        values[circuit_id, column["circuit_total_hosts"]] = float(len(frame))
        values[circuit_id, column["calendar_persistence_score"]] = persistence
    metadata = data.circuits.set_index("circuitId").reindex(range(77))
    values[:, column["circuit_country_code"]] = metadata["_country_code"].fillna(-1).to_numpy()
    values[:, column["circuit_lat"]] = metadata["lat"].fillna(0).to_numpy()
    values[:, column["circuit_lng"]] = metadata["lng"].fillna(0).to_numpy()
    values[:, column["circuit_alt"]] = metadata["alt"].fillna(0).to_numpy()
    values[:, column["circuit_alt_missing"]] = metadata["alt"].isna().astype(float).to_numpy()
    active_country = {}
    for circuit_id in range(77):
        if last_flags[circuit_id] > 0:
            country = str(countries.get(circuit_id, ""))
            active_country[country] = active_country.get(country, 0) + 1
    values[:, column["active_same_country"]] = [active_country.get(str(countries.get(circuit_id, "")), 0) for circuit_id in range(77)]
    return pd.DataFrame(values, index=np.arange(77), columns=CALENDAR_FEATURES)


def safe_mean(frame: pd.DataFrame, column: str, default: float = 0.0) -> float:
    if frame.empty or column not in frame:
        return default
    value = frame[column].mean()
    return default if pd.isna(value) else float(value)


def driver_features(cutoff: pd.Timestamp, driver_ids: np.ndarray, data: PreparedData) -> tuple[pd.DataFrame, dict[int, int]]:
    values = np.zeros((len(driver_ids), len(DRIVER_FEATURES)), dtype=np.float64)
    column = {name: idx for idx, name in enumerate(DRIVER_FEATURES)}
    cutoff_year = cutoff.year
    history = data.results[data.results["date"] <= cutoff]
    standings = data.standings[data.standings["date"] <= cutoff]
    qualifying = data.qualifying[data.qualifying["date"] <= cutoff]
    constructor_results = data.constructor_results[data.constructor_results["date"] <= cutoff]
    constructor_standings = data.constructor_standings[data.constructor_standings["date"] <= cutoff]
    driver_metadata = data.drivers.set_index("driverId")
    constructor_metadata = data.constructors.set_index("constructorId")
    latest_constructors: dict[int, int] = {}
    races_last_season = max(1, int(data.races[(data.races["date"] <= cutoff) & (data.races["_year"] == cutoff_year - 1)]["raceId"].nunique()))
    for row_index, driver_id_value in enumerate(driver_ids):
        driver_id = int(driver_id_value)
        metadata = driver_metadata.loc[driver_id]
        dob = metadata["dob"]
        age = float((cutoff - dob).days / 365.25) if pd.notna(dob) else 0.0
        values[row_index, column["driver_age"]] = age
        values[row_index, column["driver_nationality_code"]] = float(metadata["_nationality_code"])
        values[row_index, column["episode_year"]] = float(cutoff_year)
        frame = history[history["driverId"] == driver_id].sort_values("date")
        if frame.empty:
            values[row_index, column["driver_recency_days"]] = 10000.0
            values[row_index, column["qualifying_recency_days"]] = 10000.0
            values[row_index, column["standings_latest_position"]] = 100.0
            values[row_index, column["qualifying_position_3"]] = 100.0
            values[row_index, column["constructor_latest_standing"]] = 100.0
            latest_constructors[driver_id] = -1
            continue
        first_date = frame["date"].min()
        values[row_index, column["driver_career_years"]] = float((cutoff - first_date).days / 365.25)
        values[row_index, column["driver_total_starts"]] = float(len(frame))
        values[row_index, column["driver_recency_days"]] = float(min(10000, (cutoff - frame["date"].max()).days))
        recent_frames = {}
        for window in (1, 2, 3, 5):
            recent = frame[(frame["_year"] >= cutoff_year - window) & (frame["_year"] < cutoff_year)]
            recent_frames[window] = recent
            values[row_index, column[f"driver_starts_{window}"]] = float(len(recent))
        one = recent_frames[1]
        three = recent_frames[3]
        values[row_index, column["driver_points_per_start_1"]] = safe_mean(one, "points")
        values[row_index, column["driver_points_per_start_3"]] = safe_mean(three, "points")
        values[row_index, column["driver_position_order_1"]] = safe_mean(one, "positionOrder", 30.0)
        values[row_index, column["driver_position_order_3"]] = safe_mean(three, "positionOrder", 30.0)
        values[row_index, column["driver_finish_rate_3"]] = float(three["position"].notna().mean()) if len(three) else 0.0
        values[row_index, column["driver_grid_3"]] = safe_mean(three, "grid", 30.0)
        values[row_index, column["driver_laps_3"]] = safe_mean(three, "laps")
        values[row_index, column["driver_status_mean_3"]] = safe_mean(three, "statusId")
        values[row_index, column["driver_status_diversity_3"]] = float(three["statusId"].nunique()) if len(three) else 0.0
        prior = frame[frame["_year"] == cutoff_year - 2]
        values[row_index, column["driver_points_trend"]] = safe_mean(one, "points") - safe_mean(prior, "points")
        values[row_index, column["driver_starts_trend"]] = float(len(one) - len(prior))
        values[row_index, column["driver_round_mean_3"]] = safe_mean(three, "_round_norm", 0.5)
        values[row_index, column["driver_round_min_3"]] = float(three["_round_norm"].min()) if len(three) else 0.0
        values[row_index, column["driver_round_max_3"]] = float(three["_round_norm"].max()) if len(three) else 1.0
        values[row_index, column["driver_round_early_3"]] = float((three["_round_norm"] <= 0.4).mean()) if len(three) else 0.0
        values[row_index, column["driver_round_late_3"]] = float((three["_round_norm"] >= 0.7).mean()) if len(three) else 0.0
        values[row_index, column["driver_last_season_ratio"]] = float(min(1.5, len(one) / races_last_season))
        driver_standings = standings[standings["driverId"] == driver_id].sort_values("date")
        values[row_index, column["standings_latest_position"]] = 100.0
        if len(driver_standings):
            latest = driver_standings.iloc[-1]
            values[row_index, column["standings_latest_position"]] = float(latest["position"])
            values[row_index, column["standings_latest_points"]] = float(latest["points"])
            values[row_index, column["standings_latest_wins"]] = float(latest["wins"])
            previous = driver_standings[driver_standings["_year"] <= cutoff_year - 2]
            if len(previous):
                prior_latest = previous.iloc[-1]
                values[row_index, column["standings_position_trajectory"]] = float(prior_latest["position"] - latest["position"])
                values[row_index, column["standings_points_trajectory"]] = float(latest["points"] - prior_latest["points"])
        driver_qualifying = qualifying[qualifying["driverId"] == driver_id].sort_values("date")
        qualifying_recent = driver_qualifying[(driver_qualifying["_year"] >= cutoff_year - 3) & (driver_qualifying["_year"] < cutoff_year)]
        values[row_index, column["qualifying_starts_3"]] = float(len(qualifying_recent))
        values[row_index, column["qualifying_position_3"]] = safe_mean(qualifying_recent, "position", 100.0)
        values[row_index, column["qualifying_recency_days"]] = float(min(10000, (cutoff - driver_qualifying["date"].max()).days)) if len(driver_qualifying) else 10000.0
        latest_constructor = int(frame.iloc[-1]["constructorId"])
        latest_constructors[driver_id] = latest_constructor
        values[row_index, column["latest_constructor_id"]] = float(latest_constructor)
        constructor_driver = frame[frame["constructorId"] == latest_constructor]
        values[row_index, column["constructor_tenure_years"]] = float((cutoff - constructor_driver["date"].min()).days / 365.25)
        constructor_history = history[history["constructorId"] == latest_constructor]
        constructor_one = constructor_history[constructor_history["_year"] == cutoff_year - 1]
        constructor_prior = constructor_history[constructor_history["_year"] == cutoff_year - 2]
        values[row_index, column["constructor_recent_starts"]] = float(len(constructor_one))
        latest_constructor_results = constructor_results[constructor_results["constructorId"] == latest_constructor].sort_values("date")
        recent_constructor_results = latest_constructor_results[latest_constructor_results["_year"] == cutoff_year - 1]
        values[row_index, column["constructor_recent_points"]] = safe_mean(recent_constructor_results, "points")
        values[row_index, column["constructor_viability_trend"]] = float(len(constructor_one) - len(constructor_prior))
        latest_constructor_standings = constructor_standings[constructor_standings["constructorId"] == latest_constructor].sort_values("date")
        values[row_index, column["constructor_latest_standing"]] = 100.0
        if len(latest_constructor_standings):
            latest_cs = latest_constructor_standings.iloc[-1]
            values[row_index, column["constructor_latest_standing"]] = float(latest_cs["position"])
            values[row_index, column["constructor_latest_points"]] = float(latest_cs["points"])
            values[row_index, column["constructor_latest_wins"]] = float(latest_cs["wins"])
        constructor_nationality_code = -1.0
        constructor_nationality = ""
        if latest_constructor in constructor_metadata.index:
            constructor_nationality_code = float(constructor_metadata.loc[latest_constructor, "_nationality_code"])
            constructor_nationality = str(constructor_metadata.loc[latest_constructor, "nationality"]).lower()
        values[row_index, column["constructor_nationality_code"]] = constructor_nationality_code
        driver_nationality = str(metadata["nationality"]).lower()
        values[row_index, column["constructor_driver_nat_match"]] = float(driver_nationality == constructor_nationality)
        teammates_one = set(constructor_one["driverId"].astype(int)) - {driver_id}
        teammates_prior = set(constructor_prior["driverId"].astype(int)) - {driver_id}
        union = teammates_one | teammates_prior
        values[row_index, column["teammate_continuity"]] = float(len(teammates_one & teammates_prior) / max(1, len(union)))
    return pd.DataFrame(values, index=driver_ids, columns=DRIVER_FEATURES), latest_constructors


def nationality_country_match(nationality: str, country: str) -> float:
    nationality_key = nationality.strip().lower()
    country_key = country.strip().lower()
    if not nationality_key or not country_key:
        return 0.0
    return float(country_key in NATIONALITY_COUNTRY.get(nationality_key, set()))


def pair_features(
    cutoff: pd.Timestamp,
    driver_ids: np.ndarray,
    calendar: pd.DataFrame,
    drivers: pd.DataFrame,
    latest_constructors: dict[int, int],
    data: PreparedData,
) -> pd.DataFrame:
    rows = len(driver_ids) * 77
    values = np.zeros((rows, len(PAIR_FEATURES)), dtype=np.float64)
    column = {name: idx for idx, name in enumerate(PAIR_FEATURES)}
    cutoff_year = cutoff.year
    driver_position = {int(driver_id): index for index, driver_id in enumerate(driver_ids)}
    history = data.results[(data.results["date"] <= cutoff) & (data.results["driverId"].isin(driver_ids))]
    for (driver_id_value, circuit_id_value), frame in history.groupby(["driverId", "circuitId"], sort=False):
        driver_id = int(driver_id_value)
        circuit_id = int(circuit_id_value)
        index = driver_position[driver_id] * 77 + circuit_id
        recent_one = frame[frame["_year"] == cutoff_year - 1]
        recent_three = frame[(frame["_year"] >= cutoff_year - 3) & (frame["_year"] < cutoff_year)]
        recent_five = frame[(frame["_year"] >= cutoff_year - 5) & (frame["_year"] < cutoff_year)]
        values[index, column["pair_starts_total"]] = float(len(frame))
        values[index, column["pair_starts_1"]] = float(len(recent_one))
        values[index, column["pair_starts_3"]] = float(len(recent_three))
        values[index, column["pair_starts_5"]] = float(len(recent_five))
        values[index, column["pair_recency_days"]] = float(min(10000, (cutoff - frame["date"].max()).days))
        values[index, column["pair_last_season_attendance"]] = float(len(recent_one) > 0)
        values[index, column["pair_position_order"]] = safe_mean(frame, "positionOrder", 30.0)
        values[index, column["pair_points_per_start"]] = safe_mean(frame, "points")
        values[index, column["pair_finish_rate"]] = float(frame["position"].notna().mean())
        values[index, column["pair_grid"]] = safe_mean(frame, "grid", 30.0)
        values[index, column["pair_laps"]] = safe_mean(frame, "laps")
        values[index, column["pair_status_mean"]] = safe_mean(frame, "statusId")
    qualifying = data.qualifying[(data.qualifying["date"] <= cutoff) & (data.qualifying["driverId"].isin(driver_ids))]
    for (driver_id_value, circuit_id_value), frame in qualifying.groupby(["driverId", "circuitId"], sort=False):
        driver_id = int(driver_id_value)
        circuit_id = int(circuit_id_value)
        index = driver_position[driver_id] * 77 + circuit_id
        values[index, column["pair_qualifying_starts"]] = float(len(frame))
        values[index, column["pair_qualifying_position"]] = safe_mean(frame, "position", 100.0)
    constructor_ids = set(value for value in latest_constructors.values() if value >= 0)
    constructor_history = data.results[(data.results["date"] <= cutoff) & (data.results["constructorId"].isin(constructor_ids))]
    constructor_aggregates = {}
    for (constructor_id_value, circuit_id_value), frame in constructor_history.groupby(["constructorId", "circuitId"], sort=False):
        recent = frame[(frame["_year"] >= cutoff_year - 3) & (frame["_year"] < cutoff_year)]
        constructor_aggregates[(int(constructor_id_value), int(circuit_id_value))] = (
            float(len(frame)),
            float(len(recent)),
            safe_mean(frame, "points"),
        )
    driver_metadata = data.drivers.set_index("driverId")
    circuit_metadata = data.circuits.set_index("circuitId")
    expected_round = calendar["calendar_expected_round"].to_numpy()
    for driver_index, driver_id_value in enumerate(driver_ids):
        driver_id = int(driver_id_value)
        latest_constructor = latest_constructors.get(driver_id, -1)
        nationality = str(driver_metadata.loc[driver_id, "nationality"])
        driver_round_mean = float(drivers.iloc[driver_index]["driver_round_mean_3"])
        early_affinity = float(drivers.iloc[driver_index]["driver_round_early_3"])
        late_affinity = float(drivers.iloc[driver_index]["driver_round_late_3"])
        for circuit_id in range(77):
            index = driver_index * 77 + circuit_id
            constructor_values = constructor_aggregates.get((latest_constructor, circuit_id), (0.0, 0.0, 0.0))
            values[index, column["constructor_circuit_starts"]] = constructor_values[0]
            values[index, column["constructor_circuit_recent_starts"]] = constructor_values[1]
            values[index, column["constructor_circuit_points"]] = constructor_values[2]
            country = str(circuit_metadata.loc[circuit_id, "country"])
            values[index, column["nationality_country_match"]] = nationality_country_match(nationality, country)
            values[index, column["round_profile_distance"]] = abs(float(expected_round[circuit_id]) - driver_round_mean)
            values[index, column["round_early_affinity"]] = early_affinity * float(expected_round[circuit_id] <= 0.4)
            values[index, column["round_late_affinity"]] = late_affinity * float(expected_round[circuit_id] >= 0.7)
    missing_recency = values[:, column["pair_recency_days"]] == 0
    no_starts = values[:, column["pair_starts_total"]] == 0
    values[missing_recency & no_starts, column["pair_recency_days"]] = 10000.0
    missing_position = values[:, column["pair_position_order"]] == 0
    values[missing_position & no_starts, column["pair_position_order"]] = 30.0
    missing_grid = values[:, column["pair_grid"]] == 0
    values[missing_grid & no_starts, column["pair_grid"]] = 30.0
    no_qualifying = values[:, column["pair_qualifying_starts"]] == 0
    values[no_qualifying, column["pair_qualifying_position"]] = 100.0
    return pd.DataFrame(values, columns=PAIR_FEATURES)


def target_host_mask(cutoff: pd.Timestamp, data: PreparedData) -> np.ndarray:
    end = cutoff + pd.Timedelta(days=365)
    circuits = data.races.loc[(data.races["date"] > cutoff) & (data.races["date"] <= end), "circuitId"].astype(int).unique()
    mask = np.zeros(77, dtype=np.uint8)
    mask[circuits] = 1
    return mask


def build_episode_matrix(episodes: pd.DataFrame, data: PreparedData, with_labels: bool) -> EpisodeMatrix:
    group_count = len(episodes)
    features = np.empty((group_count * 77, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.zeros(group_count * 77, dtype=np.uint8)
    host_labels = np.zeros(group_count * 77, dtype=np.uint8)
    label_sizes = np.zeros(group_count, dtype=np.int16)
    group_dates = episodes["date"].to_numpy(dtype="datetime64[ns]")
    driver_ids = episodes["driverId"].to_numpy(dtype=np.int64)
    for cutoff_value, group in episodes.groupby("date", sort=True):
        cutoff = pd.Timestamp(cutoff_value)
        indices = group.index.to_numpy(dtype=int)
        drivers_for_cutoff = group["driverId"].to_numpy(dtype=np.int64)
        calendar = calendar_features(cutoff, data)
        driver_frame, latest_constructors = driver_features(cutoff, drivers_for_cutoff, data)
        pair_frame = pair_features(cutoff, drivers_for_cutoff, calendar, driver_frame, latest_constructors, data)
        combined = np.concatenate(
            [
                np.tile(calendar.to_numpy(dtype=np.float64), (len(drivers_for_cutoff), 1)),
                np.repeat(driver_frame.to_numpy(dtype=np.float64), 77, axis=0),
                pair_frame.to_numpy(dtype=np.float64),
            ],
            axis=1,
        )
        host = target_host_mask(cutoff, data) if with_labels else np.zeros(77, dtype=np.uint8)
        for local_index, episode_index in enumerate(indices):
            pair_slice = slice(episode_index * 77, (episode_index + 1) * 77)
            local_slice = slice(local_index * 77, (local_index + 1) * 77)
            features[pair_slice] = combined[local_slice]
            host_labels[pair_slice] = host
            if with_labels:
                target = np.asarray(group.iloc[local_index]["target"], dtype=np.int64)
                labels[episode_index * 77 + target] = 1
                label_sizes[episode_index] = len(target)
    features = np.nan_to_num(features, nan=0.0, posinf=10000.0, neginf=-10000.0)
    return EpisodeMatrix(
        features=features,
        labels=labels,
        host_labels=host_labels,
        group_dates=group_dates,
        driver_ids=driver_ids,
        label_sizes=label_sizes,
    )


def pair_indices(group_indices: np.ndarray) -> np.ndarray:
    return (group_indices[:, None] * 77 + np.arange(77, dtype=np.int64)[None, :]).reshape(-1)


def episode_weights(group_dates: np.ndarray, reference: np.datetime64 | None = None) -> np.ndarray:
    years = pd.DatetimeIndex(group_dates).year.to_numpy(dtype=float)
    reference_year = float(pd.Timestamp(reference).year) if reference is not None else float(years.max())
    return np.maximum(0.05, 0.5 ** ((reference_year - years) / 8.0))


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64).reshape(-1, 77)
    output = np.empty_like(matrix)
    circuit_ids = np.arange(77, dtype=np.int64)
    for row_index, row in enumerate(matrix):
        order = np.lexsort((circuit_ids, -row))
        rank = np.empty(77, dtype=np.int64)
        rank[order] = np.arange(77)
        output[row_index] = 1.0 - rank / 76.0
    return output.reshape(-1)


def fallback_generative(matrix: EpisodeMatrix) -> GenerativeScores:
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    calendar = matrix.features[:, feature_index["calendar_persistence_score"]]
    ratio = matrix.features[:, feature_index["driver_last_season_ratio"]]
    distance = matrix.features[:, feature_index["round_profile_distance"]]
    p_host = 0.08 + 0.90 * percentile_ranks(calendar)
    p_attend = np.clip(0.08 + 0.88 * ratio - 0.15 * distance, 0.03, 0.98)
    return GenerativeScores(p_host=p_host.astype(np.float32), p_attend=p_attend.astype(np.float32))


def binary_model(seed: int, trees: int, depth: int, leaves: int, min_child: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=trees,
        learning_rate=0.035,
        num_leaves=leaves,
        max_depth=depth,
        min_child_samples=min_child,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=11,
        verbosity=-1,
    )


def generative_model(regularization: float, seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=regularization,
            solver="lbfgs",
            max_iter=300,
            random_state=seed,
        ),
    )


def host_feature_indices() -> np.ndarray:
    return np.array([FEATURE_NAMES.index(name) for name in CALENDAR_FEATURES], dtype=np.int64)


def attendance_feature_indices() -> np.ndarray:
    names = DRIVER_FEATURES + [
        "calendar_expected_round",
        "calendar_round_std_5",
        "round_profile_distance",
        "round_early_affinity",
        "round_late_affinity",
    ]
    return np.array([FEATURE_NAMES.index(name) for name in names], dtype=np.int64)


def unique_host_pair_indices(matrix: EpisodeMatrix, group_indices: np.ndarray) -> np.ndarray:
    selected_dates = matrix.group_dates[group_indices]
    _, first_positions = np.unique(selected_dates, return_index=True)
    first_groups = group_indices[np.sort(first_positions)]
    return pair_indices(first_groups)


def fit_generative_models(
    matrix: EpisodeMatrix,
    train_groups: np.ndarray,
    trees: int,
    seed: int,
) -> tuple[object | None, object | None]:
    train_pairs = pair_indices(train_groups)
    host_pairs = unique_host_pair_indices(matrix, train_groups)
    weights_group = episode_weights(matrix.group_dates[train_groups])
    weight_lookup = dict(zip(train_groups.tolist(), weights_group.tolist()))
    host_groups = host_pairs // 77
    host_weights = np.array([weight_lookup.get(int(group), 0.05) for group in host_groups], dtype=np.float64)
    host_labels = matrix.host_labels[host_pairs]
    host_model = None
    if len(np.unique(host_labels)) == 2 and len(host_pairs) >= 300:
        host_model = generative_model(0.01, seed)
        host_model.fit(
            matrix.features[host_pairs][:, host_feature_indices()],
            host_labels,
            logisticregression__sample_weight=host_weights,
        )
    attend_pairs = train_pairs[matrix.host_labels[train_pairs] == 1]
    attend_labels = matrix.labels[attend_pairs]
    attend_model = None
    if len(np.unique(attend_labels)) == 2 and len(attend_pairs) >= 300:
        attend_weights = np.array([weight_lookup.get(int(group), 0.05) for group in attend_pairs // 77], dtype=np.float64)
        attend_model = generative_model(0.03, seed + 1)
        attend_model.fit(
            matrix.features[attend_pairs][:, attendance_feature_indices()],
            attend_labels,
            logisticregression__sample_weight=attend_weights,
        )
    return host_model, attend_model


def apply_generative_models(
    target: EpisodeMatrix,
    target_pairs: np.ndarray,
    host_model: object | None,
    attend_model: object | None,
    fallback: GenerativeScores,
) -> None:
    if host_model is not None:
        fallback.p_host[target_pairs] = host_model.predict_proba(target.features[target_pairs][:, host_feature_indices()])[:, 1]
    if attend_model is not None:
        fallback.p_attend[target_pairs] = attend_model.predict_proba(target.features[target_pairs][:, attendance_feature_indices()])[:, 1]


def forward_blocks(group_dates: np.ndarray) -> list[np.ndarray]:
    dates = np.sort(np.unique(group_dates))
    early = dates[pd.DatetimeIndex(dates).year < 1990]
    modern = dates[pd.DatetimeIndex(dates).year >= 1990]
    blocks = [early[index:index + 5] for index in range(0, len(early), 5)]
    blocks.extend([np.array([date]) for date in modern])
    return [block for block in blocks if len(block)]


def crossfit_generative(matrix: EpisodeMatrix, trees: int, debug: bool) -> GenerativeScores:
    output = fallback_generative(matrix)
    blocks = forward_blocks(matrix.group_dates)
    if debug:
        blocks = [block for block in blocks if pd.Timestamp(block[-1]).year >= 1988]
    for block in blocks:
        block_start = block.min()
        train_groups = np.where(matrix.group_dates + np.timedelta64(365, "D") <= block_start)[0]
        target_groups = np.where(np.isin(matrix.group_dates, block))[0]
        if len(train_groups) < 100 or not len(target_groups):
            continue
        host_model, attend_model = fit_generative_models(matrix, train_groups, trees, 17)
        apply_generative_models(matrix, pair_indices(target_groups), host_model, attend_model, output)
    return output


def fit_predict_generative(
    train: EpisodeMatrix,
    target: EpisodeMatrix,
    trees: int,
) -> GenerativeScores:
    output = fallback_generative(target)
    train_groups = np.arange(train.num_groups, dtype=np.int64)
    host_model, attend_model = fit_generative_models(train, train_groups, trees, 17)
    target_pairs = np.arange(target.num_groups * 77, dtype=np.int64)
    apply_generative_models(target, target_pairs, host_model, attend_model, output)
    return output


def ranking_features(matrix: EpisodeMatrix, generative: GenerativeScores, pair_subset: np.ndarray) -> np.ndarray:
    learned = np.column_stack(
        [
            generative.p_host[pair_subset],
            generative.p_attend[pair_subset],
            generative.product[pair_subset],
        ]
    )
    return np.concatenate([matrix.features[pair_subset], learned.astype(np.float32)], axis=1)


def ranker_model(seed: int, trees: int) -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank",
        metric="map",
        n_estimators=trees,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed,
        n_jobs=11,
        verbosity=-1,
        lambdarank_truncation_level=13,
        label_gain=[0, 1],
    )


def companion_model(seed: int, trees: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=trees,
        learning_rate=0.025,
        num_leaves=15,
        max_depth=5,
        min_child_samples=80,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=seed + 1000,
        n_jobs=11,
        verbosity=-1,
    )


def fit_predict_personalized(
    train: EpisodeMatrix,
    target: EpisodeMatrix,
    train_groups: np.ndarray,
    target_groups: np.ndarray,
    train_generative: GenerativeScores,
    target_generative: GenerativeScores,
    trees: int,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_pairs = pair_indices(train_groups)
    target_pairs = pair_indices(target_groups)
    train_x = ranking_features(train, train_generative, train_pairs)
    target_x = ranking_features(target, target_generative, target_pairs)
    train_y = train.labels[train_pairs]
    group_weights = episode_weights(train.group_dates[train_groups], target.group_dates[target_groups].min())
    sample_weights = np.repeat(group_weights, 77)
    ranker_scores = np.zeros(len(target_pairs), dtype=np.float64)
    classifier_scores = np.zeros(len(target_pairs), dtype=np.float64)
    for seed in seeds:
        ranker = ranker_model(seed, trees)
        ranker.fit(
            train_x,
            train_y,
            group=np.full(len(train_groups), 77, dtype=np.int32),
            sample_weight=sample_weights,
            eval_at=[10, 13],
            callbacks=[lgb.log_evaluation(0)],
        )
        classifier = companion_model(seed, trees)
        classifier.fit(train_x, train_y, sample_weight=sample_weights, callbacks=[lgb.log_evaluation(0)])
        ranker_scores += ranker.predict(target_x) / len(seeds)
        classifier_scores += classifier.predict_proba(target_x)[:, 1] / len(seeds)
    ranker_percentile = percentile_ranks(ranker_scores)
    classifier_percentile = percentile_ranks(classifier_scores)
    generative_percentile = percentile_ranks(target_generative.product[target_pairs])
    return ranker_percentile, classifier_percentile, generative_percentile


def gate_feature_indices() -> np.ndarray:
    names = DRIVER_FEATURES + [
        "calendar_count_1",
        "calendar_count_3",
        "calendar_count_5",
    ]
    return np.array([FEATURE_NAMES.index(name) for name in names], dtype=np.int64)


def fallback_gate(matrix: EpisodeMatrix) -> np.ndarray:
    feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
    pairs = np.arange(matrix.num_groups, dtype=np.int64) * 77
    ratio = matrix.features[pairs, feature_index["driver_last_season_ratio"]]
    recency = matrix.features[pairs, feature_index["driver_recency_days"]]
    linear = -1.2 + 4.0 * ratio - 0.0002 * np.minimum(recency, 3000)
    return 1.0 / (1.0 + np.exp(-np.clip(linear, -8, 8)))


def fit_gate_model(
    train: EpisodeMatrix,
    train_groups: np.ndarray,
    trees: int,
    reference: np.datetime64 | None = None,
) -> lgb.LGBMClassifier | None:
    train_pairs = train_groups * 77
    labels = (train.label_sizes[train_groups] >= 10).astype(np.uint8)
    if len(np.unique(labels)) < 2:
        return None
    model = binary_model(43, min(400, trees), 3, 7, 40)
    weights = episode_weights(train.group_dates[train_groups], reference)
    model.fit(train.features[train_pairs][:, gate_feature_indices()], labels, sample_weight=weights)
    return model


def predict_gate_raw(model: lgb.LGBMClassifier | None, target: EpisodeMatrix, target_groups: np.ndarray) -> np.ndarray:
    if model is None:
        return fallback_gate(target)[target_groups]
    target_pairs = target_groups * 77
    return model.predict_proba(target.features[target_pairs][:, gate_feature_indices()])[:, 1]


def crossfit_gate(matrix: EpisodeMatrix, trees: int, debug: bool) -> np.ndarray:
    output = fallback_gate(matrix)
    blocks = forward_blocks(matrix.group_dates)
    if debug:
        blocks = [block for block in blocks if pd.Timestamp(block[-1]).year >= 1988]
    for block in blocks:
        block_start = block.min()
        train_groups = np.where(matrix.group_dates + np.timedelta64(365, "D") <= block_start)[0]
        target_groups = np.where(np.isin(matrix.group_dates, block))[0]
        if len(train_groups) < 100 or not len(target_groups):
            continue
        model = fit_gate_model(matrix, train_groups, trees, block_start)
        output[target_groups] = predict_gate_raw(model, matrix, target_groups)
    return output


def fit_predict_gate(
    train: EpisodeMatrix,
    target: EpisodeMatrix,
    train_groups: np.ndarray,
    target_groups: np.ndarray,
    trees: int,
) -> np.ndarray:
    reference = target.group_dates[target_groups].min()
    model = fit_gate_model(train, train_groups, trees, reference)
    return predict_gate_raw(model, target, target_groups)


def calendar_percentile(matrix: EpisodeMatrix, groups: np.ndarray) -> np.ndarray:
    index = FEATURE_NAMES.index("calendar_persistence_score")
    return percentile_ranks(matrix.features[pair_indices(groups), index])


def stable_top_ten(scores: np.ndarray) -> np.ndarray:
    matrix = np.asarray(scores, dtype=np.float64).reshape(-1, 77)
    circuit_ids = np.arange(77, dtype=np.int64)
    output = np.empty((len(matrix), 10), dtype=np.int64)
    for row_index, row in enumerate(matrix):
        output[row_index] = np.lexsort((circuit_ids, -row))[:10]
    return output


def map_at_ten(predictions: np.ndarray, matrix: EpisodeMatrix, groups: np.ndarray) -> float:
    values = []
    for prediction, group in zip(predictions, groups):
        true_ids = set(np.where(matrix.labels[group * 77:(group + 1) * 77] == 1)[0].tolist())
        hits = np.array([int(value in true_ids) for value in prediction], dtype=np.float64)
        precision = np.cumsum(hits) / np.arange(1, 11)
        values.append(float(np.sum(precision * hits) / min(len(true_ids), 10)))
    return float(np.mean(values)) if values else 0.0


def candidate_blends() -> dict[str, tuple[float, float, float]]:
    return {
        "generative_only": (0.0, 0.0, 1.0),
        "equal": (1.0 / 3, 1.0 / 3, 1.0 / 3),
        "ranker_heavy": (0.55, 0.25, 0.20),
        "classifier_heavy": (0.25, 0.55, 0.20),
    }


def blend_score(
    components: tuple[np.ndarray, np.ndarray, np.ndarray],
    weights: tuple[float, float, float],
) -> np.ndarray:
    return weights[0] * components[0] + weights[1] * components[1] + weights[2] * components[2]


def gated_score(calendar: np.ndarray, personalized: np.ndarray, p_full: np.ndarray, alpha: float) -> np.ndarray:
    pair_gate = np.repeat(alpha * (1.0 - np.clip(p_full, 0.0, 1.0)), 77)
    return (1.0 - pair_gate) * calendar + pair_gate * personalized


def certainty_shrink(personalized: np.ndarray, calendar: np.ndarray, pool_size: int) -> np.ndarray:
    if pool_size <= 0:
        return personalized
    matrix = calendar.reshape(-1, 77)
    threshold = 1.0 - (pool_size - 1) / 76.0
    certainty = (matrix >= threshold).reshape(-1)
    return personalized * (0.10 + 0.90 * certainty.astype(np.float64))


def forward_select(
    matrix: EpisodeMatrix,
    generative: GenerativeScores,
    gate_oof: np.ndarray,
    trees: int,
    seeds: list[int],
    debug: bool,
) -> tuple[dict, list[dict]]:
    years = [2003, 2004] if debug else list(range(1994, 2005))
    available_years = set(pd.DatetimeIndex(matrix.group_dates).year.tolist())
    years = [year for year in years if year in available_years]
    records: dict[tuple[str, float, int], list[float]] = {}
    stratum_records: list[dict] = []
    blend_options = candidate_blends()
    alphas = [0.35, 0.60, 0.85]
    certainty_pools = [0, 12]
    first_fold_seconds = 0.0
    for fold_number, year in enumerate(years):
        started = time.time()
        target_groups = np.where(pd.DatetimeIndex(matrix.group_dates).year == year)[0]
        fold_date = matrix.group_dates[target_groups].min()
        train_groups = np.where(matrix.group_dates + np.timedelta64(365, "D") <= fold_date)[0]
        components = fit_predict_personalized(
            matrix,
            matrix,
            train_groups,
            target_groups,
            generative,
            generative,
            trees,
            seeds,
        )
        p_full = gate_oof[target_groups]
        calendar = calendar_percentile(matrix, target_groups)
        fold_detail = {"year": year, "count": int(len(target_groups))}
        for blend_name, weights in blend_options.items():
            base_personalized = blend_score(components, weights)
            for certainty_pool in certainty_pools:
                personalized = certainty_shrink(base_personalized, calendar, certainty_pool)
                for alpha in alphas:
                    prediction = stable_top_ten(gated_score(calendar, personalized, p_full, alpha))
                    score = map_at_ten(prediction, matrix, target_groups)
                    records.setdefault((blend_name, alpha, certainty_pool), []).append(score)
        baseline_prediction = stable_top_ten(calendar)
        fold_detail["calendar_map"] = map_at_ten(baseline_prediction, matrix, target_groups)
        full_groups = target_groups[matrix.label_sizes[target_groups] >= 10]
        partial_groups = target_groups[matrix.label_sizes[target_groups] < 10]
        best_reference = gated_score(
            calendar,
            blend_score(components, blend_options["equal"]),
            p_full,
            0.60,
        )
        prediction = stable_top_ten(best_reference)
        if len(full_groups):
            local = np.where(matrix.label_sizes[target_groups] >= 10)[0]
            fold_detail["full_count"] = int(len(local))
            fold_detail["full_map"] = map_at_ten(prediction[local], matrix, full_groups)
        if len(partial_groups):
            local = np.where(matrix.label_sizes[target_groups] < 10)[0]
            fold_detail["partial_count"] = int(len(local))
            fold_detail["partial_map"] = map_at_ten(prediction[local], matrix, partial_groups)
        fold_detail["seconds"] = round(time.time() - started, 3)
        if fold_number == 0:
            first_fold_seconds = fold_detail["seconds"]
        stratum_records.append(fold_detail)
        print(
            f"[selection] fold={year} rows={len(target_groups)} "
            f"calendar_map={fold_detail['calendar_map']:.6f} seconds={fold_detail['seconds']:.2f}"
        )
    summaries = []
    for (blend_name, alpha, certainty_pool), values in records.items():
        array = np.asarray(values, dtype=np.float64)
        mean = float(array.mean())
        se = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
        summaries.append(
            {
                "blend": blend_name,
                "alpha": alpha,
                "certainty_pool": certainty_pool,
                "mean_map": mean,
                "season_se": se,
                "criterion": mean - 0.5 * se,
                "fold_maps": array.tolist(),
            }
        )
    generative_best = max((row for row in summaries if row["blend"] == "generative_only"), key=lambda row: row["criterion"])
    learned_best = max((row for row in summaries if row["blend"] != "generative_only"), key=lambda row: row["criterion"])
    ranker_stable = learned_best["mean_map"] > generative_best["mean_map"]
    eligible = summaries if ranker_stable else [row for row in summaries if row["blend"] == "generative_only"]
    best_statistical = max(eligible, key=lambda row: row["criterion"])
    tied = [row for row in eligible if row["mean_map"] >= best_statistical["mean_map"] - max(best_statistical["season_se"], 1e-12)]
    complexity = {"generative_only": 0, "equal": 1, "ranker_heavy": 2, "classifier_heavy": 2}
    selected = min(
        tied,
        key=lambda row: (
            complexity[row["blend"]] + int(row["certainty_pool"] > 0),
            row["alpha"],
            -row["criterion"],
        ),
    )
    selected = dict(selected)
    selected["ranker_retained"] = bool(selected["blend"] != "generative_only" and ranker_stable)
    selected["first_fold_seconds"] = first_fold_seconds
    selected["generative_best_mean"] = generative_best["mean_map"]
    selected["learned_best_mean"] = learned_best["mean_map"]
    selected["candidate_summaries"] = summaries
    print(
        f"[selection] selected blend={selected['blend']} alpha={selected['alpha']:.2f} "
        f"certainty_pool={selected['certainty_pool']} mean_map={selected['mean_map']:.6f} se={selected['season_se']:.6f} "
        f"ranker_retained={selected['ranker_retained']} "
        f"learned_best_mean={selected['learned_best_mean']:.6f}"
    )
    return selected, stratum_records


def fit_final_predictions(
    train: EpisodeMatrix,
    target: EpisodeMatrix,
    train_generative: GenerativeScores,
    target_generative: GenerativeScores,
    selected: dict,
    trees: int,
    seeds: list[int],
) -> tuple[np.ndarray, dict]:
    train_groups = np.arange(train.num_groups, dtype=np.int64)
    target_groups = np.arange(target.num_groups, dtype=np.int64)
    components = fit_predict_personalized(
        train,
        target,
        train_groups,
        target_groups,
        train_generative,
        target_generative,
        trees,
        seeds,
    )
    p_full = fit_predict_gate(train, target, train_groups, target_groups, trees)
    calendar = calendar_percentile(target, target_groups)
    personalized = blend_score(components, candidate_blends()[selected["blend"]])
    personalized = certainty_shrink(personalized, calendar, int(selected["certainty_pool"]))
    score = gated_score(calendar, personalized, p_full, float(selected["alpha"]))
    predictions = stable_top_ten(score)
    diagnostics = {
        "p_full_min": float(p_full.min()),
        "p_full_mean": float(p_full.mean()),
        "p_full_max": float(p_full.max()),
        "unique_prediction_rows": int(len(np.unique(predictions, axis=0))),
    }
    return predictions, diagnostics


def validate_predictions(predictions: np.ndarray, expected_rows: int) -> None:
    if predictions.shape != (expected_rows, 10):
        raise RuntimeError(f"prediction shape {predictions.shape} does not match {(expected_rows, 10)}")
    if not np.issubdtype(predictions.dtype, np.integer):
        raise RuntimeError(f"prediction dtype {predictions.dtype} is not integer")
    if predictions.min() < 0 or predictions.max() >= 77:
        raise RuntimeError("prediction ids are outside [0, 77)")
    if any(len(np.unique(row)) != 10 for row in predictions):
        raise RuntimeError("prediction row contains duplicate ids")


def matrix_cache_key(data: PreparedData) -> str:
    return f"gclm_v4_r{len(data.races)}_x{len(data.results)}_q{len(data.qualifying)}"


def register_artifact(cache_root: Path, name: str, path: Path, content_key: str, description: str) -> None:
    registry = cache_root / "artifacts.json"
    lock_path = cache_root / "artifacts.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        entries = json.loads(registry.read_text()) if registry.exists() else []
        relative = str(path.relative_to(cache_root))
        if not any(entry.get("path") == relative and entry.get("content_key") == content_key for entry in entries):
            entries.append(
                {
                    "name": name,
                    "path": relative,
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py --debug or main.py with the same sanitized rel-f1 cache.",
                }
            )
            temporary = registry.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_or_build_matrix(
    name: str,
    episodes: pd.DataFrame,
    data: PreparedData,
    cache_root: Path,
    with_labels: bool,
) -> tuple[EpisodeMatrix, bool]:
    content_key = matrix_cache_key(data)
    directory = cache_root / "gated_calendar_lambdamart_lane0" / content_key
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.pkl"
    if path.exists():
        with path.open("rb") as handle:
            matrix = pickle.load(handle)
        expected_dates = episodes["date"].to_numpy(dtype="datetime64[ns]")
        expected_drivers = episodes["driverId"].to_numpy(dtype=np.int64)
        if (
            matrix.num_groups == len(episodes)
            and np.array_equal(matrix.group_dates, expected_dates)
            and np.array_equal(matrix.driver_ids, expected_drivers)
        ):
            return matrix, True
    matrix = build_episode_matrix(episodes.reset_index(drop=True), data, with_labels)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)
    register_artifact(
        cache_root,
        f"Gated Calendar LambdaMART {name} feature matrix",
        path,
        content_key,
        f"Temporally censored all-table pair features for {name}; {matrix.num_groups * 77} rows.",
    )
    return matrix, False
