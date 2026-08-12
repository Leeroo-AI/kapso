from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit, prange


BASE_TIME = 946684800
STRIDE = 2_000_000_000
DAY = 86400.0
NODE_COUNTS = {
    "user": 218680,
    "beer": 751524,
    "style": 157,
    "brewer": 50013,
    "place": 86061,
    "place_type": 8,
    "availability": 117424,
    "state": 622,
    "country": 251,
}


def seconds(values) -> np.ndarray:
    return pd.to_datetime(values).values.astype("datetime64[s]").astype(np.int64)


def foreign_key(values) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(-1).to_numpy(np.int32)


def ordered_mapping(frame: pd.DataFrame, key: str, value: str, size: int) -> np.ndarray:
    result = np.full(size, -1, dtype=np.int32)
    result[frame[key].to_numpy(np.int64)] = foreign_key(frame[value])
    return result


def save_array(root: Path, name: str, value: np.ndarray) -> None:
    np.save(root / f"{name}.npy", value, allow_pickle=False)


def build_relation(
    root: Path,
    name: str,
    frame: pd.DataFrame,
    entity_col: str,
    entity_count: int,
    value_col: str | None,
) -> dict:
    valid = frame["user_id"].notna() & frame["created_at"].notna()
    user = frame.loc[valid, "user_id"].to_numpy(np.int32)
    entity = foreign_key(frame.loc[valid, entity_col])
    event_time = seconds(frame.loc[valid, "created_at"])
    if value_col is None:
        value = np.ones(len(user), dtype=np.float32)
    else:
        value = pd.to_numeric(frame.loc[valid, value_col], errors="coerce").fillna(0).to_numpy(np.float32)
    legal = (
        (user >= 0)
        & (user < NODE_COUNTS["user"])
        & (entity < entity_count)
        & (event_time >= BASE_TIME)
    )
    user = user[legal]
    entity = entity[legal]
    event_time = event_time[legal]
    value = value[legal]
    user_order = np.lexsort((event_time, user))
    user_sorted = user[user_order]
    time_by_user = event_time[user_order]
    entity_by_user = entity[user_order]
    value_by_user = value[user_order]
    key_by_user = user_sorted.astype(np.int64) * STRIDE + time_by_user - BASE_TIME
    entity_valid = entity >= 0
    entity_order = np.lexsort((event_time[entity_valid], entity[entity_valid]))
    entity_sorted = entity[entity_valid][entity_order]
    time_by_entity = event_time[entity_valid][entity_order]
    user_by_entity = user[entity_valid][entity_order]
    key_by_entity = entity_sorted.astype(np.int64) * STRIDE + time_by_entity - BASE_TIME
    first_time = np.full(entity_count, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_time, entity[entity_valid], event_time[entity_valid])
    save_array(root, f"{name}_user_key", key_by_user)
    save_array(root, f"{name}_user_time", time_by_user)
    save_array(root, f"{name}_user_entity", entity_by_user)
    save_array(root, f"{name}_user_value", value_by_user)
    save_array(root, f"{name}_entity_key", key_by_entity)
    save_array(root, f"{name}_entity_time", time_by_entity)
    save_array(root, f"{name}_entity_user", user_by_entity)
    save_array(root, f"{name}_entity_first", first_time)
    return {
        "rows": int(len(user)),
        "minimum": int(event_time.min()) if len(event_time) else None,
        "maximum": int(event_time.max()) if len(event_time) else None,
    }


def register_artifact(shared_root: Path, artifact_root: Path) -> None:
    registry = shared_root / "artifacts.json"
    lock_path = shared_root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        else:
            records = []
        relative = str(artifact_root.relative_to(shared_root))
        if not any(record.get("path") == relative for record in records):
            records.append(
                {
                    "name": "lane3 temporal heterogeneous graph indexes",
                    "path": relative,
                    "description": "Projected cutoff-safe interaction indexes and immutable foreign-key mappings for user churn",
                    "content_key": "rel-ratebeer-user-churn-lane3-temporal-v4",
                    "rebuild_hint": "Run main.py after deleting the artifact directory",
                }
            )
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def ensure_temporal_cache(shared_root: Path, database_root: Path) -> Path:
    root = shared_root / "lane3_temporal_graph_v4"
    manifest = root / "manifest.json"
    if manifest.exists():
        continent_path = root / "country_continent.npy"
        if not continent_path.exists():
            countries = pd.read_parquet(database_root / "countries.parquet", columns=["country_id", "continent"])
            continent_labels = {"North America": 0, "Europe": 1, "Asia": 2, "South America": 3, "Oceania": 4, "Africa": 5}
            country_continent = np.full(NODE_COUNTS["country"], -1, dtype=np.int32)
            for row in countries.itertuples():
                country_continent[row.country_id] = continent_labels.get(row.continent, -1)
            save_array(root, "country_continent", country_continent)
        return root
    root.mkdir(parents=True, exist_ok=True)
    start = time.time()
    beer_ratings = pd.read_parquet(
        database_root / "beer_ratings.parquet",
        columns=["user_id", "beer_id", "created_at", "total_score"],
    )
    beer_stats = build_relation(root, "beer", beer_ratings, "beer_id", NODE_COUNTS["beer"], "total_score")
    del beer_ratings
    gc.collect()
    place_ratings = pd.read_parquet(
        database_root / "place_ratings.parquet",
        columns=["user_id", "place_id", "created_at", "total_score"],
    )
    place_stats = build_relation(root, "place", place_ratings, "place_id", NODE_COUNTS["place"], "total_score")
    del place_ratings
    favorites = pd.read_parquet(
        database_root / "favorites.parquet",
        columns=["user_id", "beer_id", "created_at"],
    )
    favorite_stats = build_relation(root, "favorite", favorites, "beer_id", NODE_COUNTS["beer"], None)
    del favorites
    beers = pd.read_parquet(database_root / "beers.parquet", columns=["beer_id", "style_id", "brewer_id", "created_at"])
    brewers = pd.read_parquet(database_root / "brewers.parquet", columns=["brewer_id", "state_id", "country_id", "type_id"])
    places = pd.read_parquet(database_root / "places.parquet", columns=["place_id", "state_id", "country_id", "type_id"])
    states = pd.read_parquet(database_root / "states.parquet", columns=["state_id", "country_id"])
    users = pd.read_parquet(database_root / "users.parquet", columns=["user_id", "created_at"])
    countries = pd.read_parquet(database_root / "countries.parquet", columns=["country_id", "continent"])
    save_array(root, "beer_style", ordered_mapping(beers, "beer_id", "style_id", NODE_COUNTS["beer"]))
    save_array(root, "beer_brewer", ordered_mapping(beers, "beer_id", "brewer_id", NODE_COUNTS["beer"]))
    beer_created = np.full(NODE_COUNTS["beer"], np.iinfo(np.int64).max, dtype=np.int64)
    beer_created[beers["beer_id"].to_numpy(np.int64)] = seconds(beers["created_at"])
    save_array(root, "beer_created", beer_created)
    save_array(root, "brewer_state", ordered_mapping(brewers, "brewer_id", "state_id", NODE_COUNTS["brewer"]))
    save_array(root, "brewer_country", ordered_mapping(brewers, "brewer_id", "country_id", NODE_COUNTS["brewer"]))
    save_array(root, "brewer_type", ordered_mapping(brewers, "brewer_id", "type_id", NODE_COUNTS["brewer"]))
    save_array(root, "place_state", ordered_mapping(places, "place_id", "state_id", NODE_COUNTS["place"]))
    save_array(root, "place_country", ordered_mapping(places, "place_id", "country_id", NODE_COUNTS["place"]))
    save_array(root, "place_type", ordered_mapping(places, "place_id", "type_id", NODE_COUNTS["place"]))
    save_array(root, "state_country", ordered_mapping(states, "state_id", "country_id", NODE_COUNTS["state"]))
    continent_labels = {"North America": 0, "Europe": 1, "Asia": 2, "South America": 3, "Oceania": 4, "Africa": 5}
    country_continent = np.full(NODE_COUNTS["country"], -1, dtype=np.int32)
    for row in countries.itertuples():
        country_continent[row.country_id] = continent_labels.get(row.continent, -1)
    save_array(root, "country_continent", country_continent)
    user_created = np.full(NODE_COUNTS["user"], BASE_TIME, dtype=np.int64)
    user_created[users["user_id"].to_numpy(np.int64)] = seconds(users["created_at"])
    save_array(root, "user_created", user_created)
    record = {
        "version": 4,
        "built_seconds": time.time() - start,
        "relations": {"beer": beer_stats, "place": place_stats, "favorite": favorite_stats},
        "availability_legal_edges_v": 0,
        "availability_legal_edges_t": 0,
    }
    temporary = root / "manifest.tmp"
    temporary.write_text(json.dumps(record, indent=2))
    os.replace(temporary, manifest)
    register_artifact(shared_root, root)
    return root


@dataclass
class RelationData:
    name: str
    root: Path

    def __post_init__(self) -> None:
        load = lambda suffix: np.load(self.root / f"{self.name}_{suffix}.npy", mmap_mode="r")
        self.user_key = load("user_key")
        self.user_time = load("user_time")
        self.user_entity = load("user_entity")
        self.user_value = load("user_value")
        self.entity_key = load("entity_key")
        self.entity_time = load("entity_time")
        self.entity_user = load("entity_user")
        self.entity_first = load("entity_first")
        self.value_prefix = np.empty(len(self.user_value) + 1, dtype=np.float64)
        self.value_square_prefix = np.empty(len(self.user_value) + 1, dtype=np.float64)
        self.value_prefix[0] = 0
        self.value_square_prefix[0] = 0
        np.cumsum(self.user_value, dtype=np.float64, out=self.value_prefix[1:])
        np.cumsum(np.square(self.user_value.astype(np.float64)), dtype=np.float64, out=self.value_square_prefix[1:])
        self.gap_ready = False

    def prepare_gaps(self) -> None:
        if self.gap_ready:
            return
        event_user = np.asarray(self.user_key // STRIDE, dtype=np.int32)
        gap = np.zeros(len(self.user_time), dtype=np.float32)
        if len(gap) > 1:
            same = event_user[1:] == event_user[:-1]
            gap[1:] = np.where(same, (self.user_time[1:] - self.user_time[:-1]) / DAY, 0)
        self.gap_sum = np.empty(len(gap) + 1, dtype=np.float64)
        self.gap_square_sum = np.empty(len(gap) + 1, dtype=np.float64)
        self.gap_sum[0] = 0
        self.gap_square_sum[0] = 0
        np.cumsum(gap, dtype=np.float64, out=self.gap_sum[1:])
        np.cumsum(np.square(gap.astype(np.float64)), dtype=np.float64, out=self.gap_square_sum[1:])
        self.gap_thresholds = {}
        for threshold in [7, 30, 60, 90, 180]:
            values = np.empty(len(gap) + 1, dtype=np.int32)
            values[0] = 0
            np.cumsum(gap > threshold, dtype=np.int32, out=values[1:])
            self.gap_thresholds[threshold] = values
        self.gap_ready = True

    def sample_events(self, cutoff: int, count: int, rng: np.random.Generator):
        accepted = []
        remaining = count
        while remaining:
            candidates = rng.integers(0, len(self.user_time), size=max(remaining * 2, 1024))
            candidates = candidates[self.user_time[candidates] <= cutoff]
            if len(candidates):
                take = candidates[:remaining]
                accepted.append(take)
                remaining -= len(take)
        indices = np.concatenate(accepted)
        packed = self.user_key[indices]
        users = (packed // STRIDE).astype(np.int32)
        return users, np.asarray(self.user_entity[indices], dtype=np.int32), np.asarray(self.user_time[indices], dtype=np.int64)


class TemporalData:
    def __init__(self, root: Path):
        self.root = root
        self.beer = RelationData("beer", root)
        self.place = RelationData("place", root)
        self.favorite = RelationData("favorite", root)
        for name in [
            "beer_style",
            "beer_brewer",
            "beer_created",
            "brewer_state",
            "brewer_country",
            "brewer_type",
            "place_state",
            "place_country",
            "place_type",
            "state_country",
            "country_continent",
            "user_created",
        ]:
            setattr(self, name, np.load(root / f"{name}.npy", mmap_mode="r"))


def channel_features(
    relation: RelationData,
    users: np.ndarray,
    seed_time: np.ndarray,
    windows: list[int],
    prefix: str,
) -> tuple[list[np.ndarray], list[str], dict]:
    packed = users.astype(np.int64) * STRIDE + seed_time - BASE_TIME
    right = np.searchsorted(relation.user_key, packed, side="right")
    start = np.searchsorted(relation.user_key, users.astype(np.int64) * STRIDE, side="left")
    arrays = []
    names = []
    lefts = {}
    counts = {}
    for window in windows:
        left = np.searchsorted(relation.user_key, packed - int(window * DAY), side="left")
        count = right - left
        lefts[window] = left
        counts[window] = count
        arrays.extend([np.log1p(count), count.astype(np.float64)])
        names.extend([f"{prefix}_log_count_{window}d", f"{prefix}_count_{window}d"])
        total = relation.value_prefix[right] - relation.value_prefix[left]
        square = relation.value_square_prefix[right] - relation.value_square_prefix[left]
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        variance = np.divide(square, count, out=np.zeros_like(square), where=count > 0) - np.square(mean)
        arrays.extend([mean, np.sqrt(np.maximum(variance, 0))])
        names.extend([f"{prefix}_mean_value_{window}d", f"{prefix}_std_value_{window}d"])
    lifetime = right - start
    arrays.extend([np.log1p(lifetime), lifetime.astype(np.float64)])
    names.extend([f"{prefix}_log_lifetime_count", f"{prefix}_lifetime_count"])
    positions = []
    recencies = []
    for rank in [1, 2, 3, 5, 10, 20]:
        position = right - rank
        valid = position >= start
        safe = np.maximum(position, 0)
        event_time = np.where(valid, relation.user_time[safe], seed_time - int(10000 * DAY))
        recency = (seed_time - event_time) / DAY
        arrays.append(recency)
        names.append(f"{prefix}_recency_rank_{rank}")
        positions.append(position)
        recencies.append(recency)
    for rank in range(5):
        gap = recencies[rank + 1] - recencies[rank]
        valid = positions[rank + 1] >= start
        arrays.append(np.where(valid, gap, 10000.0))
        names.append(f"{prefix}_gap_{rank + 1}")
    first_position = np.minimum(start, np.maximum(right - 1, 0))
    has_history = right > start
    first_event = np.where(has_history, relation.user_time[first_position], seed_time)
    arrays.append(np.maximum((seed_time - first_event) / DAY, 0))
    names.append(f"{prefix}_tenure_days")
    for short, long in zip(windows[:-1], windows[1:]):
        ratio = np.divide(counts[short], np.maximum(counts[long], 1))
        arrays.append(ratio)
        names.append(f"{prefix}_count_ratio_{short}_{long}")
    return arrays, names, {"right": right, "lefts": lefts, "start": start, "counts": counts}


@njit(parallel=True, cache=True)
def diversity_features(
    entity: np.ndarray,
    event_time: np.ndarray,
    right: np.ndarray,
    left30: np.ndarray,
    left90: np.ndarray,
    left365: np.ndarray,
    first_map: np.ndarray,
    second_map: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(right), 9), dtype=np.float32)
    for i in prange(len(right)):
        b30 = set()
        b90 = set()
        b365 = set()
        f90 = set()
        s90 = set()
        d30 = set()
        d90 = set()
        d365 = set()
        lower = max(left365[i], right[i] - 2048)
        for j in range(right[i] - 1, lower - 1, -1):
            item = int(entity[j])
            b365.add(item)
            day = int(event_time[j] // 86400)
            d365.add(day)
            if j >= left90[i]:
                b90.add(item)
                d90.add(day)
                mapped = int(first_map[item])
                if mapped >= 0:
                    f90.add(mapped)
                mapped = int(second_map[item])
                if mapped >= 0:
                    s90.add(mapped)
            if j >= left30[i]:
                b30.add(item)
                d30.add(day)
        result[i, 0] = len(b30)
        result[i, 1] = len(b90)
        result[i, 2] = len(b365)
        result[i, 3] = len(f90)
        result[i, 4] = len(s90)
        result[i, 5] = max(0, right[i] - left365[i] - 2048)
        result[i, 6] = len(d30)
        result[i, 7] = len(d90)
        result[i, 8] = len(d365)
    return result


@njit(parallel=True, cache=True)
def summarize_routes(
    entities: np.ndarray,
    neighbors: np.ndarray,
    first_times: np.ndarray,
    second_times: np.ndarray,
    seed_time: np.ndarray,
    entity_key: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(seed_time), 12), dtype=np.float32)
    for i in prange(len(seed_time)):
        first_count = 0
        second_count = 0
        degree_sum = 0.0
        degree_max = 0.0
        first_age_sum = 0.0
        first_age_max = 0.0
        second_age_sum = 0.0
        second_age_max = 0.0
        neighbor_set = set()
        for j in range(entities.shape[1]):
            entity = int(entities[i, j])
            if entity < 0:
                continue
            first_count += 1
            boundary = entity * STRIDE + int(seed_time[i]) - BASE_TIME + 1
            entity_right = np.searchsorted(entity_key, boundary)
            entity_left = np.searchsorted(entity_key, entity * STRIDE)
            degree = math.log1p(entity_right - entity_left)
            degree_sum += degree
            degree_max = max(degree_max, degree)
            first_age = max(0.0, (seed_time[i] - first_times[i, j]) / DAY)
            first_age_sum += first_age
            first_age_max = max(first_age_max, first_age)
            for k in range(neighbors.shape[2]):
                neighbor = int(neighbors[i, j, k])
                if neighbor < 0:
                    continue
                second_count += 1
                neighbor_set.add(neighbor)
                second_age = max(0.0, (seed_time[i] - second_times[i, j, k]) / DAY)
                second_age_sum += second_age
                second_age_max = max(second_age_max, second_age)
        result[i, 0] = first_count
        result[i, 1] = second_count
        result[i, 2] = len(neighbor_set)
        result[i, 3] = degree_sum / max(first_count, 1)
        result[i, 4] = degree_max
        result[i, 5] = first_age_sum / max(first_count, 1)
        result[i, 6] = first_age_max
        result[i, 7] = second_age_sum / max(second_count, 1)
        result[i, 8] = second_age_max
        result[i, 9] = len(neighbor_set) / max(second_count, 1)
        result[i, 10] = second_count / max(first_count, 1)
        result[i, 11] = math.log1p(second_count)
    return result


@njit(parallel=True, cache=True)
def preference_composition(
    entity: np.ndarray,
    right: np.ndarray,
    left90: np.ndarray,
    left365: np.ndarray,
    beer_style: np.ndarray,
    beer_brewer: np.ndarray,
    brewer_country: np.ndarray,
    country_continent: np.ndarray,
    style_bucket: np.ndarray,
) -> np.ndarray:
    result = np.zeros((len(right), 50), dtype=np.float32)
    for i in prange(len(right)):
        result[i, 46] = -1
        result[i, 47] = -1
        count90 = 0
        count365 = 0
        lower = max(left365[i], right[i] - 4096)
        for j in range(right[i] - 1, lower - 1, -1):
            beer = int(entity[j])
            if beer < 0:
                continue
            style = int(beer_style[beer])
            bucket = int(style_bucket[style]) if style >= 0 else -1
            brewer = int(beer_brewer[beer])
            country = int(brewer_country[brewer]) if brewer >= 0 else -1
            continent = int(country_continent[country]) if country >= 0 else -1
            count365 += 1
            if bucket >= 0:
                result[i, 20 + bucket] += 1
            if j >= left90[i]:
                count90 += 1
                if bucket >= 0:
                    result[i, bucket] += 1
                if continent >= 0:
                    result[i, 40 + continent] += 1
        for bucket in range(20):
            result[i, bucket] /= max(count90, 1)
            result[i, 20 + bucket] /= max(count365, 1)
        for continent in range(6):
            result[i, 40 + continent] /= max(count90, 1)
        if right[i] > lower:
            beer = int(entity[right[i] - 1])
            if beer >= 0:
                result[i, 46] = beer_style[beer]
                brewer = int(beer_brewer[beer])
                country = int(brewer_country[brewer]) if brewer >= 0 else -1
                result[i, 47] = country_continent[country] if country >= 0 else -1
        result[i, 48] = count90
        result[i, 49] = count365
    return result


def make_hazard_features(data: TemporalData, users, timestamps) -> tuple[np.ndarray, list[str], dict]:
    users = np.asarray(users, dtype=np.int32)
    seed_time = seconds(timestamps)
    arrays = []
    names = []
    beer_arrays, beer_names, beer_aux = channel_features(
        data.beer, users, seed_time, [1, 2, 3, 7, 14, 30, 60, 90, 180, 365, 730], "beer"
    )
    arrays.extend(beer_arrays)
    names.extend(beer_names)
    diversity = diversity_features(
        data.beer.user_entity,
        data.beer.user_time,
        beer_aux["right"],
        beer_aux["lefts"][30],
        beer_aux["lefts"][90],
        beer_aux["lefts"][365],
        data.beer_style,
        data.beer_brewer,
    )
    for column, name in enumerate(
        ["beer_unique_30d", "beer_unique_90d", "beer_unique_365d", "style_unique_90d", "brewer_unique_90d", "diversity_events_capped", "beer_active_days_30d", "beer_active_days_90d", "beer_active_days_365d"]
    ):
        arrays.append(diversity[:, column])
        names.append(name)
    arrays.extend(
        [
            np.divide(beer_aux["counts"][30], np.maximum(diversity[:, 6], 1)),
            np.divide(beer_aux["counts"][90], np.maximum(diversity[:, 7], 1)),
            np.divide(beer_aux["counts"][365], np.maximum(diversity[:, 8], 1)),
            np.divide(beer_aux["counts"][30], np.maximum(diversity[:, 0], 1)),
            np.divide(beer_aux["counts"][90], np.maximum(diversity[:, 1], 1)),
        ]
    )
    names.extend(["beer_events_per_day_30d", "beer_events_per_day_90d", "beer_events_per_day_365d", "beer_repeat_ratio_30d", "beer_repeat_ratio_90d"])
    data.beer.prepare_gaps()
    gap_count = np.maximum(beer_aux["right"] - beer_aux["start"] - 1, 0)
    gap_sum = data.beer.gap_sum[beer_aux["right"]] - data.beer.gap_sum[beer_aux["start"]]
    gap_square = data.beer.gap_square_sum[beer_aux["right"]] - data.beer.gap_square_sum[beer_aux["start"]]
    gap_mean = np.divide(gap_sum, gap_count, out=np.zeros_like(gap_sum), where=gap_count > 0)
    gap_variance = np.divide(gap_square, gap_count, out=np.zeros_like(gap_square), where=gap_count > 0) - np.square(gap_mean)
    arrays.extend([gap_mean, np.sqrt(np.maximum(gap_variance, 0))])
    names.extend(["beer_lifetime_gap_mean", "beer_lifetime_gap_std"])
    for threshold in [7, 30, 60, 90, 180]:
        prefix = data.beer.gap_thresholds[threshold]
        count = prefix[beer_aux["right"]] - prefix[beer_aux["start"]]
        arrays.append(np.divide(count, np.maximum(gap_count, 1)))
        names.append(f"beer_lifetime_gap_over_{threshold}d")
    right = beer_aux["right"]
    start = beer_aux["start"]
    last_position = np.maximum(right - 1, 0)
    previous_position = np.maximum(right - 2, 0)
    has_last = right > start
    has_previous = right - start > 1
    last_value = np.where(has_last, data.beer.user_value[last_position], 0)
    previous_value = np.where(has_previous, data.beer.user_value[previous_position], last_value)
    arrays.extend([last_value, previous_value, last_value - previous_value])
    names.extend(["beer_last_value", "beer_previous_value", "beer_last_value_delta"])
    place_arrays, place_names, place_aux = channel_features(
        data.place, users, seed_time, [30, 90, 180, 365, 730], "place"
    )
    arrays.extend(place_arrays)
    names.extend(place_names)
    favorite_arrays, favorite_names, favorite_aux = channel_features(
        data.favorite, users, seed_time, [30, 90, 180, 365], "favorite"
    )
    arrays.extend(favorite_arrays)
    names.extend(favorite_names)
    beer_recency = arrays[names.index("beer_recency_rank_1")]
    place_recency = arrays[names.index("place_recency_rank_1")]
    favorite_recency = arrays[names.index("favorite_recency_rank_1")]
    arrays.extend(
        [
            np.minimum(beer_recency, place_recency),
            np.minimum(beer_recency, favorite_recency),
            place_aux["counts"][90] > 0,
            favorite_aux["counts"][90] > 0,
            np.log1p(beer_aux["counts"][90] + place_aux["counts"][90] + favorite_aux["counts"][90]),
        ]
    )
    names.extend(["cross_any_recency", "cross_beer_favorite_recency", "cross_has_place_90d", "cross_has_favorite_90d", "cross_log_events_90d"])
    created = np.asarray(data.user_created[users], dtype=np.int64)
    arrays.append(np.maximum(seed_time - created, 0) / DAY)
    names.append("user_account_age_days")
    calendar = pd.to_datetime(np.asarray(timestamps))
    year = calendar.year.to_numpy(np.float64)
    month_angle = 2 * math.pi * calendar.month.to_numpy(np.float64) / 12.0
    arrays.extend([year, np.sin(month_angle), np.cos(month_angle)])
    names.extend(["seed_year", "seed_month_sin", "seed_month_cos"])
    top_styles = np.asarray([15, 68, 22, 16, 87, 37, 1, 41, 9, 3, 2, 138, 10, 11, 40, 33, 18, 44, 5, 59], dtype=np.int32)
    style_bucket = np.full(NODE_COUNTS["style"], -1, dtype=np.int32)
    style_bucket[top_styles] = np.arange(len(top_styles), dtype=np.int32)
    composition = preference_composition(
        data.beer.user_entity,
        beer_aux["right"],
        beer_aux["lefts"][90],
        beer_aux["lefts"][365],
        data.beer_style,
        data.beer_brewer,
        data.brewer_country,
        data.country_continent,
        style_bucket,
    )
    route_names = [
        "first_count",
        "second_count",
        "unique_neighbors",
        "mean_log_entity_degree",
        "max_log_entity_degree",
        "mean_first_age",
        "max_first_age",
        "mean_second_age",
        "max_second_age",
        "neighbor_unique_ratio",
        "neighbors_per_entity",
        "log_second_count",
    ]
    for route_prefix, relation in [("rating_route", data.beer), ("place_route", data.place), ("favorite_route", data.favorite)]:
        route = sample_relation_routes(relation, users, seed_time, (16, 8), True)
        summary = summarize_routes(*route, seed_time, relation.entity_key)
        for column, route_name in enumerate(route_names):
            arrays.append(summary[:, column])
            names.append(f"{route_prefix}_{route_name}")
        del route, summary
    for column in range(20):
        arrays.append(composition[:, column])
        names.append(f"style_top_{column}_share_90d")
    for column in range(20):
        arrays.append(composition[:, 20 + column])
        names.append(f"style_top_{column}_share_365d")
    for column in range(6):
        arrays.append(composition[:, 40 + column])
        names.append(f"brewer_continent_{column}_share_90d")
    arrays.extend([composition[:, 46], composition[:, 47], composition[:, 48], composition[:, 49]])
    names.extend(["last_style_id", "last_brewer_continent", "composition_count_90d", "composition_count_365d"])
    matrix = np.column_stack(arrays).astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=10000.0, neginf=-10000.0)
    profile = {
        "count": int(len(users)),
        "beer_count_90_quantiles": np.quantile(beer_aux["counts"][90], [0, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
        "beer_recency_quantiles": np.quantile(beer_recency, [0, 0.25, 0.5, 0.75, 0.9, 0.99]).tolist(),
        "place_coverage": float(np.mean(place_aux["counts"][90] > 0)),
        "favorite_coverage": float(np.mean(favorite_aux["counts"][90] > 0)),
    }
    return matrix, names, profile


@njit(parallel=True, cache=True)
def sample_routes(
    seed_users: np.ndarray,
    seed_time: np.ndarray,
    user_key: np.ndarray,
    user_entity: np.ndarray,
    user_time: np.ndarray,
    entity_key: np.ndarray,
    entity_user: np.ndarray,
    entity_time: np.ndarray,
    fanout_first: int,
    fanout_second: int,
    inclusive: bool,
):
    count = len(seed_users)
    entities = np.full((count, fanout_first), -1, dtype=np.int32)
    neighbors = np.full((count, fanout_first, fanout_second), -1, dtype=np.int32)
    first_times = np.full((count, fanout_first), -1, dtype=np.int64)
    second_times = np.full((count, fanout_first, fanout_second), -1, dtype=np.int64)
    for i in prange(count):
        user = int(seed_users[i])
        timestamp = int(seed_time[i])
        packed = user * STRIDE + timestamp - BASE_TIME
        boundary = packed + 1 if inclusive else packed
        right = np.searchsorted(user_key, boundary)
        left = np.searchsorted(user_key, user * STRIDE, side="left")
        found = 0
        cursor = right - 1
        while cursor >= left and found < fanout_first:
            entity = int(user_entity[cursor])
            if entity < 0:
                cursor -= 1
                continue
            duplicate = False
            for previous in range(found):
                if entities[i, previous] == entity:
                    duplicate = True
                    break
            if not duplicate:
                entities[i, found] = entity
                first_times[i, found] = int(user_time[cursor])
                entity_packed = entity * STRIDE + timestamp - BASE_TIME
                entity_boundary = entity_packed + 1 if inclusive else entity_packed
                entity_right = np.searchsorted(entity_key, entity_boundary)
                entity_left = np.searchsorted(entity_key, entity * STRIDE, side="left")
                neighbor_count = 0
                neighbor_cursor = entity_right - 1
                while neighbor_cursor >= entity_left and neighbor_count < fanout_second:
                    neighbor = int(entity_user[neighbor_cursor])
                    repeated = neighbor == user
                    for previous_neighbor in range(neighbor_count):
                        if neighbors[i, found, previous_neighbor] == neighbor:
                            repeated = True
                            break
                    if not repeated:
                        neighbors[i, found, neighbor_count] = neighbor
                        second_times[i, found, neighbor_count] = int(entity_time[neighbor_cursor])
                        neighbor_count += 1
                    neighbor_cursor -= 1
                found += 1
            cursor -= 1
    return entities, neighbors, first_times, second_times


def sample_relation_routes(
    relation: RelationData,
    users: np.ndarray,
    seed_time: np.ndarray,
    fanout: tuple[int, int],
    inclusive: bool,
):
    return sample_routes(
        np.asarray(users, dtype=np.int32),
        np.asarray(seed_time, dtype=np.int64),
        relation.user_key,
        relation.user_entity,
        relation.user_time,
        relation.entity_key,
        relation.entity_user,
        relation.entity_time,
        fanout[0],
        fanout[1],
        inclusive,
    )


def audit_temporal_routes(data: TemporalData, users, timestamps, fanout, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    users = np.asarray(users, dtype=np.int32)
    seed_time = seconds(timestamps)
    indices = rng.choice(len(users), size=min(100, len(users)), replace=False)
    maximum = -1
    sampled = 0
    relation_counts = {}
    for name, relation in [("beer", data.beer), ("place", data.place), ("favorite", data.favorite)]:
        entities, neighbors, first_times, second_times = sample_relation_routes(
            relation, users[indices], seed_time[indices], fanout, True
        )
        legal_first = first_times >= 0
        legal_second = second_times >= 0
        if np.any(first_times[legal_first] > np.repeat(seed_time[indices], fanout[0])[legal_first.ravel()]):
            raise RuntimeError(f"temporal audit failed for {name} first-hop events")
        expanded = np.repeat(seed_time[indices], fanout[0] * fanout[1])
        if np.any(second_times.ravel()[legal_second.ravel()] > expanded[legal_second.ravel()]):
            raise RuntimeError(f"temporal audit failed for {name} second-hop events")
        count = int(legal_first.sum() + legal_second.sum())
        relation_counts[name] = count
        sampled += count
        if legal_first.any():
            maximum = max(maximum, int(first_times[legal_first].max()))
        if legal_second.any():
            maximum = max(maximum, int(second_times[legal_second].max()))
    return {"seeds": int(len(indices)), "sampled_events": sampled, "relation_events": relation_counts, "maximum_event_time": maximum}
