# Imports

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Constants

VERSION = "lane0_gridpace_v2"
WINDOWS = (3, 5, 10, 20)
METRICS = ("grid", "finish", "points", "points_share", "reliability", "laps", "fast", "rank")
SEEDS = (17, 37, 73)
TEMPERATURES = (0.6, 1.0, 1.5, 2.0)
BLENDS = (0.0, 0.35, 0.65, 1.0)


# State

@dataclass
class RatingState:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    fast: float = math.nan
    slow: float = math.nan
    last_date: pd.Timestamp | None = None
    first_date: pd.Timestamp | None = None

    def update(self, value: float, date: pd.Timestamp) -> None:
        if not np.isfinite(value):
            return
        self.count += 1
        self.total += float(value)
        self.total_sq += float(value) ** 2
        self.fast = float(value) if not np.isfinite(self.fast) else 0.5 * float(value) + 0.5 * self.fast
        self.slow = float(value) if not np.isfinite(self.slow) else 0.15 * float(value) + 0.85 * self.slow
        self.last_date = date
        if self.first_date is None:
            self.first_date = date

    def estimate(self, prior: float, weight: float = 3.0) -> float:
        return float((self.total + weight * prior) / (self.count + weight))

    def uncertainty(self) -> float:
        if self.count < 2:
            return float(1.0 / math.sqrt(self.count + 1.0))
        mean = self.total / self.count
        variance = max(0.0, self.total_sq / self.count - mean * mean)
        return float(math.sqrt(variance + 0.04) / math.sqrt(self.count + 1.0))

    def trend(self) -> float:
        if not np.isfinite(self.fast) or not np.isfinite(self.slow):
            return 0.0
        return float(self.fast - self.slow)


@dataclass
class MetricHistory:
    values: dict[str, deque] = field(default_factory=lambda: {m: deque(maxlen=20) for m in METRICS})
    ewma: dict[tuple[str, int], float] = field(default_factory=dict)
    count: int = 0
    last_date: pd.Timestamp | None = None
    first_date: pd.Timestamp | None = None

    def update(self, observation: dict[str, float], date: pd.Timestamp) -> None:
        self.count += 1
        self.last_date = date
        if self.first_date is None:
            self.first_date = date
        for metric in METRICS:
            value = float(observation.get(metric, math.nan))
            if not np.isfinite(value):
                continue
            self.values[metric].append(value)
            for window in WINDOWS:
                key = (metric, window)
                alpha = 2.0 / (window + 1.0)
                previous = self.ewma.get(key, value)
                self.ewma[key] = alpha * value + (1.0 - alpha) * previous

    def snapshot(self, prefix: str, date: pd.Timestamp, full: bool = True) -> dict[str, float]:
        output: dict[str, float] = {}
        selected = METRICS if full else ("grid", "finish", "points_share", "reliability")
        for metric in selected:
            values = np.asarray(self.values[metric], dtype=np.float64)
            output[f"{prefix}_{metric}_last"] = float(values[-1]) if len(values) else math.nan
            for window in WINDOWS:
                tail = values[-window:]
                output[f"{prefix}_{metric}_mean{window}"] = float(np.mean(tail)) if len(tail) else math.nan
                output[f"{prefix}_{metric}_ewma{window}"] = float(self.ewma.get((metric, window), math.nan))
        grid = np.asarray(self.values["grid"], dtype=np.float64)
        output[f"{prefix}_grid_trend"] = float(self.ewma.get(("grid", 3), math.nan) - self.ewma.get(("grid", 10), math.nan))
        output[f"{prefix}_grid_volatility10"] = float(np.std(grid[-10:])) if len(grid) >= 2 else math.nan
        output[f"{prefix}_grid_volatility20"] = float(np.std(grid[-20:])) if len(grid) >= 2 else math.nan
        output[f"{prefix}_event_count"] = float(self.count)
        output[f"{prefix}_recency_days"] = float((date - self.last_date).total_seconds() / 86400.0) if self.last_date is not None else math.nan
        output[f"{prefix}_tenure_days"] = float((date - self.first_date).total_seconds() / 86400.0) if self.first_date is not None else 0.0
        return output


@dataclass
class StandingState:
    latest: tuple[pd.Timestamp, float, float, float, float] | None = None
    previous: tuple[pd.Timestamp, float, float, float, float] | None = None
    count: int = 0

    def update(self, date: pd.Timestamp, position: float, points: float, wins: float, round_number: float) -> None:
        self.previous = self.latest
        self.latest = (date, position, points, wins, round_number)
        self.count += 1

    def snapshot(self, prefix: str, date: pd.Timestamp) -> dict[str, float]:
        if self.latest is None:
            return {
                f"{prefix}_position": math.nan,
                f"{prefix}_points": math.nan,
                f"{prefix}_wins": math.nan,
                f"{prefix}_points_per_round": math.nan,
                f"{prefix}_points_momentum": math.nan,
                f"{prefix}_position_momentum": math.nan,
                f"{prefix}_recency_days": math.nan,
                f"{prefix}_same_year": 0.0,
                f"{prefix}_count": 0.0,
            }
        last_date, position, points, wins, round_number = self.latest
        previous_points = self.previous[2] if self.previous is not None and self.previous[0].year == last_date.year else 0.0
        previous_position = self.previous[1] if self.previous is not None and self.previous[0].year == last_date.year else position
        return {
            f"{prefix}_position": float(position),
            f"{prefix}_points": float(points),
            f"{prefix}_wins": float(wins),
            f"{prefix}_points_per_round": float(points / max(round_number, 1.0)),
            f"{prefix}_points_momentum": float(points - previous_points),
            f"{prefix}_position_momentum": float(previous_position - position),
            f"{prefix}_recency_days": float((date - last_date).total_seconds() / 86400.0),
            f"{prefix}_same_year": float(last_date.year == date.year),
            f"{prefix}_count": float(self.count),
        }


# Cache

def feature_key(context: Any) -> str:
    db = context.db
    payload = {
        "version": VERSION,
        "tables": {name: [len(table.df), list(table.df.columns)] for name, table in sorted(db.table_dict.items())},
        "splits": {
            name: [len(getattr(context, name).df), int(getattr(context, name).df["qualifyId"].sum())]
            for name in ("train", "val", "test")
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def register_artifact(cache_dir: Path, path: Path, key: str) -> None:
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            records = []
        relative = str(path.relative_to(cache_dir))
        if not any(record.get("path") == relative for record in records):
            records.append({
                "name": "lane0 F1 timestamp-safe feature matrix",
                "path": relative,
                "description": "All-table as-of features and online grid-pace rating snapshots",
                "content_key": key,
                "rebuild_hint": "Run main.py after changing VERSION or the sanitized database schema",
            })
            temporary = registry.with_suffix(".tmp.lane0")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_or_build_features(context: Any, cache_dir: Path) -> tuple[pd.DataFrame, dict[str, str], bool]:
    key = feature_key(context)
    cache_path = cache_dir / f"lane0_features_{key}.pkl"
    if cache_path.exists():
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
        return payload["features"], payload["source_max"], True
    features, source_max = build_features(context)
    temporary = cache_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({"features": features, "source_max": source_max}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, cache_path)
    register_artifact(cache_dir, cache_path, key)
    return features, source_max, False


# Feature preparation

def prepare_seeds(context: Any) -> pd.DataFrame:
    qualifying = context.db.table_dict["qualifying"].df.copy()
    frames = []
    offset = 0
    for split in ("train", "val", "test"):
        task_frame = getattr(context, split).df[["date", "qualifyId"]].copy()
        task_frame["_row_id"] = np.arange(len(task_frame), dtype=np.int64)
        task_frame["_global_id"] = np.arange(offset, offset + len(task_frame), dtype=np.int64)
        task_frame["split"] = split
        offset += len(task_frame)
        merged = task_frame.merge(
            qualifying[["qualifyId", "raceId", "driverId", "constructorId", "number", "date"]],
            on="qualifyId",
            how="left",
            suffixes=("", "_qualifying"),
            validate="one_to_one",
            sort=False,
        )
        if merged[["raceId", "driverId", "constructorId"]].isna().any().any():
            raise RuntimeError(f"qualifying join failed for {split}")
        if not (merged["date"] == merged["date_qualifying"]).all():
            raise RuntimeError(f"task and qualifying timestamps disagree for {split}")
        frames.append(merged.drop(columns="date_qualifying"))
    seeds = pd.concat(frames, ignore_index=True)
    seeds["date"] = pd.to_datetime(seeds["date"])
    return seeds


def make_observation(row: Any, field_size: int, max_points: float, max_laps: float) -> dict[str, float]:
    grid = float(row.grid)
    grid_percentile = (grid - 1.0) / max(field_size - 1.0, 1.0) if grid > 0 else math.nan
    finish = (float(row.positionOrder) - 1.0) / max(field_size - 1.0, 1.0)
    fastest_rank = float(row["rank"]) / max(field_size, 1.0) if pd.notna(row["rank"]) else math.nan
    return {
        "grid": grid_percentile,
        "finish": finish,
        "points": float(row.points),
        "points_share": float(row.points) / max(max_points, 1.0),
        "reliability": float(row.laps) / max(max_laps, 1.0),
        "laps": float(row.laps),
        "fast": float(pd.notna(row.fastestLap)),
        "rank": fastest_rank,
    }


def infer_circuit(races: pd.DataFrame, date: pd.Timestamp, year: int, round_number: int) -> tuple[int, float, pd.Timestamp | None]:
    eligible = races[(races["date"] < date) & (races["year"] < year)]
    exact = eligible[eligible["round"] == round_number]
    if len(exact):
        row = exact.sort_values(["year", "date"]).iloc[-1]
        day_delta = abs(int(row["date"].dayofyear) - int(date.dayofyear))
        confidence = 1.0 / (1.0 + (year - int(row["year"]) - 1) + day_delta / 45.0)
        return int(row["circuitId"]), float(confidence), pd.Timestamp(row["date"])
    if len(eligible):
        candidate = eligible.assign(distance=(eligible["round"] - round_number).abs() * 60 + (eligible["date"].dt.dayofyear - date.dayofyear).abs())
        row = candidate.sort_values(["year", "distance"], ascending=[False, True]).iloc[0]
        confidence = 0.25 / (1.0 + float(row["distance"]) / 90.0)
        return int(row["circuitId"]), float(confidence), pd.Timestamp(row["date"])
    return -1, 0.0, None


def history_snapshot(histories: dict[Any, MetricHistory], key: Any, prefix: str, date: pd.Timestamp, full: bool = True) -> dict[str, float]:
    history = histories.get(key)
    if history is None:
        history = MetricHistory()
    return history.snapshot(prefix, date, full)


def rating_snapshot(state: RatingState | None, prefix: str, date: pd.Timestamp, prior: float) -> dict[str, float]:
    current = state if state is not None else RatingState()
    return {
        f"{prefix}_mean": current.estimate(prior),
        f"{prefix}_uncertainty": current.uncertainty(),
        f"{prefix}_count": float(current.count),
        f"{prefix}_trend": current.trend(),
        f"{prefix}_fast": float(current.fast) if np.isfinite(current.fast) else prior,
        f"{prefix}_slow": float(current.slow) if np.isfinite(current.slow) else prior,
        f"{prefix}_recency_days": float((date - current.last_date).total_seconds() / 86400.0) if current.last_date is not None else math.nan,
    }


def build_features(context: Any) -> tuple[pd.DataFrame, dict[str, str]]:
    db = context.db.table_dict
    seeds = prepare_seeds(context)
    results = db["results"].df.copy()
    standings = db["standings"].df.copy()
    constructor_results = db["constructor_results"].df.copy()
    constructor_standings = db["constructor_standings"].df.copy()
    races = db["races"].df.copy()
    circuits = db["circuits"].df.copy().set_index("circuitId", drop=False)
    drivers = db["drivers"].df.copy().set_index("driverId", drop=False)
    constructors = db["constructors"].df.copy().set_index("constructorId", drop=False)
    for frame in (results, standings, constructor_results, constructor_standings, races):
        frame["date"] = pd.to_datetime(frame["date"])
    race_round = races.set_index("raceId")["round"].to_dict()
    race_circuit = races.set_index("raceId")["circuitId"].to_dict()
    result_events = [(pd.Timestamp(group["date"].iloc[0]), int(race_id), group.copy()) for race_id, group in results.groupby("raceId", sort=False)]
    result_events.sort(key=lambda value: (value[0], value[1]))
    standings = standings.sort_values(["date", "driverStandingsId"]).reset_index(drop=True)
    constructor_results = constructor_results.sort_values(["date", "constructorResultsId"]).reset_index(drop=True)
    constructor_standings = constructor_standings.sort_values(["date", "constructorStandingsId"]).reset_index(drop=True)
    query_groups = [(pd.Timestamp(group["date"].iloc[0]), int(race_id), group.copy()) for race_id, group in seeds.groupby("raceId", sort=False)]
    query_groups.sort(key=lambda value: (value[0], value[1]))
    qualifying_dates = db["qualifying"].df[["date"]].drop_duplicates().copy()
    qualifying_dates["date"] = pd.to_datetime(qualifying_dates["date"])
    qualifying_dates["year"] = qualifying_dates["date"].dt.year
    round_lookup: dict[pd.Timestamp, int] = {}
    for _, group in qualifying_dates.sort_values("date").groupby("year"):
        for index, date in enumerate(group["date"].sort_values(), 1):
            round_lookup[pd.Timestamp(date)] = index
    constructor_rating: dict[int, RatingState] = defaultdict(RatingState)
    constructor_season: dict[tuple[int, int], RatingState] = defaultdict(RatingState)
    season_priors: dict[tuple[int, int], tuple[float, float]] = {}
    driver_rating: dict[int, RatingState] = defaultdict(RatingState)
    teammate_rating: dict[int, RatingState] = defaultdict(RatingState)
    driver_history: dict[int, MetricHistory] = defaultdict(MetricHistory)
    constructor_history: dict[int, MetricHistory] = defaultdict(MetricHistory)
    combination_history: dict[tuple[int, int], MetricHistory] = defaultdict(MetricHistory)
    driver_circuit_history: dict[tuple[int, int], MetricHistory] = defaultdict(MetricHistory)
    constructor_circuit_history: dict[tuple[int, int], MetricHistory] = defaultdict(MetricHistory)
    driver_standing: dict[int, StandingState] = defaultdict(StandingState)
    team_standing: dict[int, StandingState] = defaultdict(StandingState)
    team_result_history: dict[int, MetricHistory] = defaultdict(MetricHistory)
    last_driver_constructor: dict[int, int] = {}
    source_max: dict[str, pd.Timestamp | None] = {
        "qualifying": None,
        "results": None,
        "standings": None,
        "constructor_results": None,
        "constructor_standings": None,
        "races_circuits": None,
    }
    result_pointer = 0
    standing_pointer = 0
    constructor_result_pointer = 0
    constructor_standing_pointer = 0
    output_rows: list[dict[str, Any]] = []

    def advance_results(limit: pd.Timestamp) -> None:
        nonlocal result_pointer
        while result_pointer < len(result_events) and result_events[result_pointer][0] < limit:
            date, race_id, group = result_events[result_pointer]
            field_size = len(group)
            max_points = float(group["points"].max())
            max_laps = float(group["laps"].max())
            circuit_id = int(race_circuit.get(race_id, -1))
            observations: dict[int, dict[str, float]] = {}
            for _, row in group.iterrows():
                driver_id = int(row.driverId)
                constructor_id = int(row.constructorId)
                observation = make_observation(row, field_size, max_points, max_laps)
                observations[driver_id] = observation
                driver_history[driver_id].update(observation, date)
                combination_history[(driver_id, constructor_id)].update(observation, date)
                driver_circuit_history[(driver_id, circuit_id)].update(observation, date)
                last_driver_constructor[driver_id] = constructor_id
                if np.isfinite(observation["grid"]):
                    driver_rating[driver_id].update(1.0 - observation["grid"], date)
            for constructor_id, team in group.groupby("constructorId"):
                constructor_id = int(constructor_id)
                team_observations = [observations[int(driver_id)] for driver_id in team["driverId"]]
                event_observation = {
                    metric: float(np.nanmean([observation[metric] for observation in team_observations]))
                    if np.isfinite([observation[metric] for observation in team_observations]).any() else math.nan
                    for metric in METRICS
                }
                constructor_history[constructor_id].update(event_observation, date)
                constructor_circuit_history[(constructor_id, circuit_id)].update(event_observation, date)
                valid_grid = [observation["grid"] for observation in team_observations if np.isfinite(observation["grid"])]
                if valid_grid:
                    prior_neutral = constructor_rating[constructor_id].estimate(0.5)
                    prior_lower = constructor_rating[constructor_id].estimate(0.42)
                    constructor_rating[constructor_id].update(1.0 - float(np.mean(valid_grid)), date)
                    season_key = (date.year, constructor_id)
                    if season_key not in season_priors:
                        season_priors[season_key] = (0.67 * prior_neutral + 0.33 * 0.5, 0.67 * prior_lower + 0.33 * 0.42)
                    constructor_season[season_key].update(1.0 - float(np.mean(valid_grid)), date)
                if len(valid_grid) >= 2:
                    team_mean = float(np.mean(valid_grid))
                    for driver_id in team["driverId"]:
                        driver_id = int(driver_id)
                        own = observations[driver_id]["grid"]
                        if np.isfinite(own):
                            teammate_rating[driver_id].update(team_mean - own, date)
            result_pointer += 1
            source_max["results"] = date

    def advance_standings(limit: pd.Timestamp) -> None:
        nonlocal standing_pointer
        while standing_pointer < len(standings) and pd.Timestamp(standings.iloc[standing_pointer]["date"]) < limit:
            row = standings.iloc[standing_pointer]
            date = pd.Timestamp(row["date"])
            driver_standing[int(row.driverId)].update(date, float(row.position), float(row.points), float(row.wins), float(race_round.get(int(row.raceId), 1)))
            standing_pointer += 1
            source_max["standings"] = date

    def advance_constructor_results(limit: pd.Timestamp) -> None:
        nonlocal constructor_result_pointer
        while constructor_result_pointer < len(constructor_results) and pd.Timestamp(constructor_results.iloc[constructor_result_pointer]["date"]) < limit:
            row = constructor_results.iloc[constructor_result_pointer]
            date = pd.Timestamp(row["date"])
            points = float(row.points)
            observation = {metric: math.nan for metric in METRICS}
            observation["points"] = points
            observation["points_share"] = points / 50.0
            team_result_history[int(row.constructorId)].update(observation, date)
            constructor_result_pointer += 1
            source_max["constructor_results"] = date

    def advance_constructor_standings(limit: pd.Timestamp) -> None:
        nonlocal constructor_standing_pointer
        while constructor_standing_pointer < len(constructor_standings) and pd.Timestamp(constructor_standings.iloc[constructor_standing_pointer]["date"]) < limit:
            row = constructor_standings.iloc[constructor_standing_pointer]
            date = pd.Timestamp(row["date"])
            team_standing[int(row.constructorId)].update(date, float(row.position), float(row.points), float(row.wins), float(race_round.get(int(row.raceId), 1)))
            constructor_standing_pointer += 1
            source_max["constructor_standings"] = date

    for date, race_id, group in query_groups:
        advance_results(date)
        advance_standings(date)
        advance_constructor_results(date)
        advance_constructor_standings(date)
        for family in ("results", "standings", "constructor_results", "constructor_standings"):
            if source_max[family] is not None and not source_max[family] < date:
                raise RuntimeError(f"temporal assertion failed for {family} at {date}")
        year = int(date.year)
        round_number = int(round_lookup[date])
        inferred_circuit, circuit_confidence, circuit_source_date = infer_circuit(races, date, year, round_number)
        if circuit_source_date is not None:
            if not circuit_source_date < date:
                raise RuntimeError(f"temporal assertion failed for race/circuit at {date}")
            current_race_max = source_max["races_circuits"]
            source_max["races_circuits"] = circuit_source_date if current_race_max is None else max(current_race_max, circuit_source_date)
        prior_races = races[races["date"] < date]
        prior_season_races = prior_races[prior_races["year"] == year]
        last_race_date = pd.Timestamp(prior_races["date"].max()) if len(prior_races) else None
        roster_size = len(group)
        event_rows: list[dict[str, Any]] = []
        for _, seed in group.iterrows():
            driver_id = int(seed.driverId)
            constructor_id = int(seed.constructorId)
            persistent = constructor_rating.get(constructor_id)
            persistent_state = persistent if persistent is not None else RatingState()
            cp_neutral = persistent_state.estimate(0.5)
            cp_lower = persistent_state.estimate(0.42)
            season_key = (year, constructor_id)
            season_state = constructor_season.get(season_key)
            neutral_reset, lower_reset = season_priors.get(season_key, (0.67 * cp_neutral + 0.33 * 0.5, 0.67 * cp_lower + 0.33 * 0.42))
            season_current = season_state if season_state is not None else RatingState()
            cs_neutral = season_current.estimate(neutral_reset)
            cs_lower = season_current.estimate(lower_reset)
            driver_state = driver_rating.get(driver_id)
            teammate_state = teammate_rating.get(driver_id)
            row: dict[str, Any] = {
                "_global_id": int(seed["_global_id"]),
                "_row_id": int(seed["_row_id"]),
                "split": seed.split,
                "date": date,
                "qualifyId": int(seed.qualifyId),
                "raceId": race_id,
                "driverId_cat": str(driver_id),
                "constructorId_cat": str(constructor_id),
                "number_cat": str(int(seed.number)),
                "qualifying_number": float(seed.number),
                "roster_size": float(roster_size),
                "year": float(year),
                "month": float(date.month),
                "day_of_year": float(date.dayofyear),
                "round_inferred": float(round_number),
                "season_progress": float(round_number / max(round_number + 2.0, 20.0)),
                "inferred_circuit_cat": str(inferred_circuit),
                "circuit_inference_confidence": circuit_confidence,
                "prior_schedule_events": float(len(prior_races)),
                "prior_season_events": float(len(prior_season_races)),
                "days_since_prior_race": float((date - last_race_date).total_seconds() / 86400.0) if last_race_date is not None else math.nan,
                "constructor_persistent_neutral": cp_neutral,
                "constructor_persistent_lower": cp_lower,
                "constructor_season_neutral": cs_neutral,
                "constructor_season_lower": cs_lower,
                "driver_field_mean": (driver_state if driver_state is not None else RatingState()).estimate(0.5),
                "driver_teammate_mean": (teammate_state if teammate_state is not None else RatingState()).estimate(0.0),
                "team_change": float(driver_id in last_driver_constructor and last_driver_constructor[driver_id] != constructor_id),
            }
            row.update(rating_snapshot(persistent, "constructor_persistent_state", date, 0.5))
            row.update(rating_snapshot(season_state, "constructor_season_state", date, neutral_reset))
            row.update(rating_snapshot(driver_state, "driver_field_state", date, 0.5))
            row.update(rating_snapshot(teammate_state, "driver_teammate_state", date, 0.0))
            row.update(history_snapshot(driver_history, driver_id, "result_driver", date, True))
            row.update(history_snapshot(constructor_history, constructor_id, "result_constructor", date, True))
            row.update(history_snapshot(combination_history, (driver_id, constructor_id), "result_combination", date, False))
            row.update(history_snapshot(driver_circuit_history, (driver_id, inferred_circuit), "result_driver_circuit", date, False))
            row.update(history_snapshot(constructor_circuit_history, (constructor_id, inferred_circuit), "result_constructor_circuit", date, False))
            row.update(history_snapshot(team_result_history, constructor_id, "constructor_result", date, False))
            row.update((driver_standing.get(driver_id) or StandingState()).snapshot("driver_standing", date))
            row.update((team_standing.get(constructor_id) or StandingState()).snapshot("constructor_standing", date))
            driver = drivers.loc[driver_id] if driver_id in drivers.index else None
            constructor = constructors.loc[constructor_id] if constructor_id in constructors.index else None
            dob = pd.Timestamp(driver.dob) if driver is not None and pd.notna(driver.dob) else None
            row["driver_age"] = float((date - dob).total_seconds() / (86400.0 * 365.25)) if dob is not None else math.nan
            row["driver_nationality_cat"] = str(driver.nationality) if driver is not None else "missing"
            row["constructor_nationality_cat"] = str(constructor.nationality) if constructor is not None else "missing"
            if inferred_circuit in circuits.index:
                circuit = circuits.loc[inferred_circuit]
                row["circuit_country_cat"] = str(circuit.country)
                row["circuit_lat"] = float(circuit.lat)
                row["circuit_lng"] = float(circuit.lng)
                row["circuit_alt"] = float(circuit.alt) if pd.notna(circuit.alt) else math.nan
            else:
                row["circuit_country_cat"] = "missing"
                row["circuit_lat"] = math.nan
                row["circuit_lng"] = math.nan
                row["circuit_alt"] = math.nan
            event_rows.append(row)
        pace_columns = [
            "constructor_persistent_neutral",
            "constructor_persistent_lower",
            "constructor_season_neutral",
            "constructor_season_lower",
            "driver_field_mean",
            "driver_teammate_mean",
            "result_driver_grid_ewma3",
            "result_driver_grid_ewma10",
            "result_constructor_grid_ewma3",
            "result_constructor_grid_ewma10",
        ]
        event_frame = pd.DataFrame(event_rows)
        for column in pace_columns:
            values = event_frame[column].astype(float)
            mean = float(values.mean())
            standard = float(values.std(ddof=0))
            event_frame[f"{column}_z"] = (values - mean) / standard if standard > 1e-9 else 0.0
            event_frame[f"{column}_rank"] = values.rank(method="average", ascending=False, pct=True).astype(float)
        relative_specs = {
            "result_driver_grid_last": -1.0,
            "result_driver_grid_ewma3": -1.0,
            "result_driver_grid_ewma5": -1.0,
            "result_driver_grid_ewma10": -1.0,
            "result_driver_finish_ewma3": -1.0,
            "result_driver_points_share_ewma3": 1.0,
            "result_driver_reliability_ewma3": 1.0,
            "result_constructor_grid_last": -1.0,
            "result_constructor_grid_ewma3": -1.0,
            "result_constructor_grid_ewma5": -1.0,
            "result_constructor_grid_ewma10": -1.0,
            "result_constructor_finish_ewma3": -1.0,
            "result_constructor_points_share_ewma3": 1.0,
            "result_constructor_reliability_ewma3": 1.0,
            "result_combination_grid_ewma3": -1.0,
            "result_driver_circuit_grid_ewma10": -1.0,
            "result_constructor_circuit_grid_ewma10": -1.0,
            "driver_standing_position": -1.0,
            "driver_standing_points_per_round": 1.0,
            "constructor_standing_position": -1.0,
            "constructor_standing_points_per_round": 1.0,
        }
        for column, direction in relative_specs.items():
            quality = event_frame[column].astype(float) * direction
            mean = float(quality.mean())
            standard = float(quality.std(ddof=0))
            event_frame[f"race_quality_z_{column}"] = (quality - mean) / standard if standard > 1e-9 else 0.0
            event_frame[f"race_quality_rank_{column}"] = 1.0 - quality.rank(method="average", ascending=False, pct=True).astype(float)
        contrast_columns = ["driver_field_mean", "result_driver_grid_ewma3", "result_driver_grid_ewma10"]
        for column in contrast_columns:
            grouped = event_frame.groupby("constructorId_cat")[column]
            sums = grouped.transform("sum")
            counts = grouped.transform("count")
            teammate = (sums - event_frame[column]) / (counts - 1).replace(0, np.nan)
            event_frame[f"teammate_contrast_{column}"] = event_frame[column] - teammate
        output_rows.extend(event_frame.to_dict("records"))
        source_max["qualifying"] = date
    features = pd.DataFrame(output_rows).sort_values("_global_id").reset_index(drop=True)
    expected = len(context.train.df) + len(context.val.df) + len(context.test.df)
    if len(features) != expected or not np.array_equal(features["_global_id"].to_numpy(), np.arange(expected)):
        raise RuntimeError("feature rows do not preserve task order")
    source_text = {family: (str(value) if value is not None else "none") for family, value in source_max.items()}
    return features, source_text


# Ratings

def add_rating_features(features: pd.DataFrame, prior_name: str, temperature: float) -> pd.DataFrame:
    frame = features.copy()
    persistent = f"constructor_persistent_{prior_name}"
    season = f"constructor_season_{prior_name}"
    rating = np.empty(len(frame), dtype=np.float64)
    pace = np.empty(len(frame), dtype=np.float64)
    channel_columns = (persistent, season, "driver_field_mean", "driver_teammate_mean")
    for _, positions in frame.groupby("raceId", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        channels = []
        for column in channel_columns:
            values = frame.iloc[positions][column].to_numpy(dtype=np.float64)
            standard = float(np.std(values))
            channels.append((values - float(np.mean(values))) / standard if standard > 1e-9 else np.zeros(len(values)))
        event_pace = 0.45 * channels[0] + 0.20 * channels[1] + 0.20 * channels[2] + 0.15 * channels[3]
        differences = (event_pace[None, :] - event_pace[:, None]) / temperature
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(differences, -30.0, 30.0)))
        event_rating = 1.0 + probabilities.sum(axis=1) - 0.5
        pace[positions] = event_pace
        rating[positions] = event_rating
    frame["pace"] = pace
    frame["r_rating"] = rating
    frame["pace_rank"] = frame.groupby("raceId")["pace"].rank(method="average", ascending=False, pct=True)
    frame["rating_rank"] = frame.groupby("raceId")["r_rating"].rank(method="average", ascending=True, pct=True)
    frame["rating_centered"] = frame["r_rating"] - (frame["roster_size"] + 1.0) / 2.0
    return frame


def project_races(frame: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    projected = np.asarray(predictions, dtype=np.float64).copy()
    for _, positions in frame.groupby("raceId", sort=False).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        values = projected[positions]
        roster_size = len(values)
        target = roster_size * (roster_size + 1.0) / 2.0
        low = -2.0 * roster_size - float(np.max(values))
        high = 2.0 * roster_size - float(np.min(values))
        for _ in range(60):
            midpoint = 0.5 * (low + high)
            total = float(np.clip(values + midpoint, 1.0, float(roster_size)).sum())
            if total < target:
                low = midpoint
            else:
                high = midpoint
        projected[positions] = np.clip(values + 0.5 * (low + high), 1.0, float(roster_size))
    return projected


# Selection

def expanding_folds(train_frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    races = train_frame[["raceId", "date"]].drop_duplicates().sort_values(["date", "raceId"])["raceId"].to_numpy()
    burn = max(1, int(math.floor(0.4 * len(races))))
    blocks = np.array_split(races[burn:], 4)
    folds = []
    for block in blocks:
        validation = train_frame["raceId"].isin(block).to_numpy()
        first_date = train_frame.loc[validation, "date"].min()
        training = (train_frame["date"] < first_date).to_numpy()
        folds.append((np.flatnonzero(training), np.flatnonzero(validation)))
    return folds


def score_prediction(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
    }


def select_rating(features: pd.DataFrame, train_y: np.ndarray, folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[str, float, list[dict[str, Any]]]:
    train_base = features[features["split"] == "train"].reset_index(drop=True)
    records = []
    validation_union = np.concatenate([validation for _, validation in folds])
    for prior in ("neutral", "lower"):
        for temperature in TEMPERATURES:
            rated = add_rating_features(train_base, prior, temperature)
            prediction = rated["r_rating"].to_numpy()
            fold_scores = [float(r2_score(train_y[validation], prediction[validation])) for _, validation in folds]
            pooled = float(r2_score(train_y[validation_union], prediction[validation_union]))
            stability = pooled + 0.08 * min(fold_scores)
            records.append({"prior": prior, "temperature": temperature, "pooled_r2": pooled, "fold_r2": fold_scores, "stability_score": stability})
    winner = max(records, key=lambda record: (record["stability_score"], record["pooled_r2"], record["prior"] == "neutral"))
    return str(winner["prior"]), float(winner["temperature"]), records


def feature_columns(frame: pd.DataFrame, group: str) -> list[str]:
    excluded = {"_global_id", "_row_id", "split", "date", "qualifyId", "raceId"}
    columns = [column for column in frame.columns if column not in excluded]
    if group == "wide":
        return columns
    if group in ("focused", "focused_identity"):
        exact = {
            "r_rating",
            "pace",
            "pace_rank",
            "rating_rank",
            "rating_centered",
            "roster_size",
            "year",
            "month",
            "round_inferred",
            "season_progress",
            "qualifying_number",
            "driver_age",
            "team_change",
            "circuit_inference_confidence",
            "days_since_prior_race",
        }
        patterns = (
            "constructor_persistent_",
            "constructor_season_",
            "driver_field_",
            "driver_teammate_",
            "driver_standing_",
            "constructor_standing_",
            "race_quality_",
            "teammate_contrast_",
        )
        history_prefixes = (
            "result_driver_",
            "result_constructor_",
            "result_combination_",
            "result_driver_circuit_",
            "result_constructor_circuit_",
            "constructor_result_",
        )
        history_suffixes = (
            "grid_last",
            "grid_mean3",
            "grid_ewma3",
            "grid_mean5",
            "grid_ewma5",
            "grid_mean10",
            "grid_ewma10",
            "grid_mean20",
            "grid_ewma20",
            "grid_trend",
            "grid_volatility10",
            "grid_volatility20",
            "finish_last",
            "finish_ewma3",
            "finish_ewma10",
            "points_share_last",
            "points_share_ewma3",
            "points_share_ewma10",
            "reliability_last",
            "reliability_ewma3",
            "reliability_ewma10",
            "fast_ewma3",
            "rank_ewma3",
            "event_count",
            "recency_days",
            "tenure_days",
        )
        focused = [
            column
            for column in columns
            if column in exact
            or any(column.startswith(pattern) for pattern in patterns)
            or any(column.startswith(prefix) and column.endswith(history_suffixes) for prefix in history_prefixes)
        ]
        if group == "focused_identity":
            focused.extend(column for column in columns if column.endswith("_cat") and column not in focused)
        return focused
    tokens = (
        "r_rating",
        "pace",
        "rating_",
        "roster",
        "year",
        "round",
        "qualifying_number",
        "constructor_persistent",
        "constructor_season",
        "driver_field",
        "driver_teammate",
        "result_driver_",
        "result_constructor_",
        "driver_standing",
        "constructor_standing",
        "team_change",
        "driver_age",
        "recency",
        "event_count",
        "uncertainty",
        "trend",
    )
    return [column for column in columns if any(token in column for token in tokens)]


def prepare_matrix(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[int]]:
    matrix = frame[columns].copy()
    categorical = []
    for index, column in enumerate(columns):
        if column.endswith("_cat"):
            matrix[column] = matrix[column].fillna("missing").astype(str)
            categorical.append(index)
        else:
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
    return matrix, categorical


def make_model(iterations: int, seed: int) -> CatBoostRegressor:
    return CatBoostRegressor(
        iterations=iterations,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=12.0,
        random_strength=0.3,
        loss_function="RMSE",
        random_seed=seed,
        thread_count=max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))),
        verbose=False,
        allow_writing_files=False,
    )


def cross_validate_residual(
    train_frame: pd.DataFrame,
    train_y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    debug: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    tree_candidates = (100,) if debug else (400, 800)
    max_trees = max(tree_candidates)
    validation_union = np.concatenate([validation for _, validation in folds])
    residual_target = train_y - train_frame["r_rating"].to_numpy()
    predictions: dict[tuple[str, int], np.ndarray] = {}
    start = time.time()
    for feature_group in ("focused", "focused_identity", "core", "wide"):
        columns = feature_columns(train_frame, feature_group)
        matrix, categorical = prepare_matrix(train_frame, columns)
        by_tree = {trees: np.full(len(train_frame), np.nan, dtype=np.float64) for trees in tree_candidates}
        for training, validation in folds:
            fold_by_tree = {trees: [] for trees in tree_candidates}
            for seed in SEEDS:
                model = make_model(max_trees, seed)
                model.fit(matrix.iloc[training], residual_target[training], cat_features=categorical)
                for trees in tree_candidates:
                    fold_by_tree[trees].append(model.predict(matrix.iloc[validation], ntree_end=trees))
            for trees in tree_candidates:
                by_tree[trees][validation] = np.mean(fold_by_tree[trees], axis=0)
        for trees in tree_candidates:
            predictions[(feature_group, trees)] = by_tree[trees]
    elapsed = time.time() - start
    records = []
    for (feature_group, trees), residual_prediction in predictions.items():
        for blend in BLENDS:
            raw = train_frame["r_rating"].to_numpy() + blend * residual_prediction
            prediction = raw.copy()
            valid_frame = train_frame.iloc[validation_union].reset_index(drop=True)
            prediction[validation_union] = project_races(valid_frame, raw[validation_union])
            fold_scores = [float(r2_score(train_y[validation], prediction[validation])) for _, validation in folds]
            pooled = float(r2_score(train_y[validation_union], prediction[validation_union]))
            stability = pooled + 0.08 * min(fold_scores)
            records.append({
                "feature_group": feature_group,
                "trees": trees,
                "blend": blend,
                "pooled_r2": pooled,
                "fold_r2": fold_scores,
                "worst_fold_r2": min(fold_scores),
                "stability_score": stability,
                "strata": stratum_scores(train_frame, train_y, prediction, validation_union),
            })
    winner = max(records, key=lambda record: (record["stability_score"], record["pooled_r2"], -record["trees"], record["feature_group"] == "core"))
    return winner, records, elapsed


def fit_ensemble(train_frame: pd.DataFrame, train_y: np.ndarray, predict_frame: pd.DataFrame, feature_group: str, trees: int) -> np.ndarray:
    columns = feature_columns(train_frame, feature_group)
    train_matrix, categorical = prepare_matrix(train_frame, columns)
    predict_matrix, _ = prepare_matrix(predict_frame, columns)
    target = train_y - train_frame["r_rating"].to_numpy()
    predictions = []
    for seed in SEEDS:
        model = make_model(trees, seed)
        model.fit(train_matrix, target, cat_features=categorical)
        predictions.append(model.predict(predict_matrix))
    return np.mean(predictions, axis=0)


def stratum_scores(train_frame: pd.DataFrame, train_y: np.ndarray, prediction: np.ndarray, validation_union: np.ndarray) -> dict[str, dict[str, float]]:
    selected = train_frame.iloc[validation_union].copy().reset_index(drop=True)
    selected["target"] = train_y[validation_union]
    selected["prediction"] = prediction[validation_union]
    selected["roster_stratum"] = pd.cut(selected["roster_size"], bins=[0, 20, 22, 100], labels=["small", "medium", "large"], include_lowest=True)
    selected["driver_experience_stratum"] = pd.cut(selected["result_driver_event_count"], bins=[-1, 5, 30, np.inf], labels=["cold", "developing", "established"])
    selected["constructor_experience_stratum"] = pd.cut(selected["result_constructor_event_count"], bins=[-1, 5, 30, np.inf], labels=["cold", "developing", "established"])
    output = {}
    for axis in ("roster_stratum", "driver_experience_stratum", "constructor_experience_stratum"):
        for name, group in selected.groupby(axis, observed=True):
            key = f"{axis}:{name}"
            output[key] = {
                "count": int(len(group)),
                "r2": float(r2_score(group["target"], group["prediction"])) if len(group) >= 2 else math.nan,
                "mae": float(mean_absolute_error(group["target"], group["prediction"])),
            }
    return output
