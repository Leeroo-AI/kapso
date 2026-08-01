from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numba import njit


DAY = 86400
V_TIME = 1535760000
T_TIME = 1577836800
DEFAULT_SCORE = 3.36
VERSION = "lane0_causal_graph_v5"


@dataclass
class Events:
    rating_id: np.ndarray
    user: np.ndarray
    beer: np.ndarray
    time: np.ndarray
    language: np.ndarray
    train_y: np.ndarray
    val_y: np.ndarray
    train_index: np.ndarray
    val_index: np.ndarray
    test_index: np.ndarray
    full_train_n: int
    full_val_n: int
    full_test_n: int

    @property
    def n_train(self) -> int:
        return len(self.train_index)

    @property
    def n_val(self) -> int:
        return len(self.val_index)

    @property
    def n_test(self) -> int:
        return len(self.test_index)

    @property
    def n_tv(self) -> int:
        return self.n_train + self.n_val


@dataclass
class Metadata:
    beer_brewer: np.ndarray
    beer_style: np.ndarray
    beer_abv: np.ndarray
    beer_ibu: np.ndarray
    beer_created: np.ndarray
    beer_updated: np.ndarray
    beer_flags: np.ndarray
    style_parent: np.ndarray
    style_category: np.ndarray
    brewer_country: np.ndarray
    brewer_state: np.ndarray
    brewer_type: np.ndarray
    brewer_created: np.ndarray
    brewer_updated: np.ndarray
    brewer_flags: np.ndarray
    country_continent: np.ndarray
    user_created: np.ndarray
    user_updated: np.ndarray
    user_type: np.ndarray


@dataclass
class FeatureBlocks:
    base: np.ndarray
    strict_tv: np.ndarray
    frozen_val: np.ndarray
    frozen_test: np.ndarray
    names: list[str]
    residual_tv: np.ndarray


def _array(table: pa.Table, name: str, fill=None, dtype=None) -> np.ndarray:
    value = table[name].combine_chunks()
    if fill is not None:
        if isinstance(fill, float) and np.isnan(fill) and not (pa.types.is_floating(value.type) or pa.types.is_decimal(value.type)):
            value = pc.cast(value, pa.float32())
        value = pc.fill_null(value, pa.scalar(fill, type=value.type))
    result = value.to_numpy(zero_copy_only=False)
    if dtype is not None:
        result = result.astype(dtype, copy=False)
    return result


def _time_array(table: pa.Table, name: str) -> np.ndarray:
    value = table[name].combine_chunks()
    if pa.types.is_timestamp(value.type):
        valid = pc.is_valid(value).to_numpy(zero_copy_only=False)
        casted = pc.cast(value, pa.int64())
        raw = pc.fill_null(casted, pa.scalar(0, type=pa.int64())).to_numpy(zero_copy_only=False)
        unit = value.type.unit
        divisor = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}[unit]
        result = (raw // divisor).astype(np.int64)
        result[~valid] = -1
        return result
    parsed = pd.to_datetime(value.to_pandas(), errors="coerce", utc=True)
    raw = parsed.astype("int64", copy=False).to_numpy()
    result = raw // 1000000000
    result[raw == np.iinfo(np.int64).min] = -1
    return result.astype(np.int64)


def _select_indices(n: int, limit: int | None) -> np.ndarray:
    if limit is None or n <= limit:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, limit, dtype=np.int64))


def load_events(debug: bool) -> Events:
    cache = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    task_dir = cache / "tasks" / os.environ["RELBENCH_TASK"]
    train = pq.read_table(task_dir / "train.parquet", columns=["rating_id", "created_at", "total_score"])
    val = pq.read_table(task_dir / "val.parquet", columns=["rating_id", "created_at", "total_score"])
    test = pq.read_table(task_dir / "test.parquet", columns=["rating_id", "created_at"])
    train_rid_full = _array(train, "rating_id", dtype=np.int64)
    val_rid_full = _array(val, "rating_id", dtype=np.int64)
    test_rid_full = _array(test, "rating_id", dtype=np.int64)
    train_t_full = _time_array(train, "created_at")
    val_t_full = _time_array(val, "created_at")
    test_t_full = _time_array(test, "created_at")
    train_y_full = _array(train, "total_score", dtype=np.float32)
    val_y_full = _array(val, "total_score", dtype=np.float32)
    train_index = _select_indices(len(train), 300000 if debug else None)
    val_index = _select_indices(len(val), 60000 if debug else None)
    test_index = _select_indices(len(test), 60000 if debug else None)
    rating_id = np.concatenate((train_rid_full[train_index], val_rid_full[val_index], test_rid_full[test_index]))
    event_time = np.concatenate((train_t_full[train_index], val_t_full[val_index], test_t_full[test_index]))
    canonical = pq.read_table(cache / "db" / "beer_ratings.parquet", columns=["rating_id", "user_id", "beer_id", "created_at", "language"])
    canonical_rid = _array(canonical, "rating_id", dtype=np.int64)
    canonical_user = _array(canonical, "user_id", fill=-1, dtype=np.int32)
    canonical_beer = _array(canonical, "beer_id", fill=-1, dtype=np.int32)
    canonical_time = _time_array(canonical, "created_at")
    languages = pa.array(["en", "pl", "fr", "de", "nl", "sv", "it", "es", "no", "da", "pt", "hu", "sk", "cs", "fi", "ca", "bs", "ru", "hr", "ro"])
    canonical_language_arrow = pc.index_in(canonical["language"].combine_chunks(), value_set=languages)
    canonical_language = pc.fill_null(canonical_language_arrow, pa.scalar(-1, type=canonical_language_arrow.type)).to_numpy(zero_copy_only=False).astype(np.int16)
    if np.array_equal(canonical_rid, np.arange(len(canonical_rid), dtype=np.int64)):
        source_index = rating_id
    else:
        lookup = np.full(int(canonical_rid.max()) + 1, -1, dtype=np.int64)
        lookup[canonical_rid] = np.arange(len(canonical_rid), dtype=np.int64)
        source_index = lookup[rating_id]
    if np.any(source_index < 0):
        raise RuntimeError("canonical event projection is missing task rating IDs")
    if not np.array_equal(canonical_time[source_index], event_time):
        raise RuntimeError("task and canonical event timestamps disagree")
    user = canonical_user[source_index]
    beer = canonical_beer[source_index]
    language = canonical_language[source_index]
    return Events(
        rating_id=rating_id,
        user=user,
        beer=beer,
        time=event_time,
        language=language,
        train_y=train_y_full[train_index],
        val_y=val_y_full[val_index],
        train_index=train_index,
        val_index=val_index,
        test_index=test_index,
        full_train_n=len(train),
        full_val_n=len(val),
        full_test_n=len(test),
    )


def _dense_table(path: Path, columns: list[str], key: str) -> tuple[pa.Table, np.ndarray]:
    table = pq.read_table(path, columns=columns)
    ids = _array(table, key, dtype=np.int64)
    return table, ids


def _scatter(values: np.ndarray, ids: np.ndarray, size: int, fill, dtype) -> np.ndarray:
    result = np.full(size, fill, dtype=dtype)
    result[ids] = values.astype(dtype, copy=False)
    return result


def load_metadata() -> Metadata:
    db = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"
    beer_cols = ["beer_id", "brewer_id", "style_id", "alcohol_pct", "ibu", "created_at", "updated_at", "is_seasonal", "is_one_off", "is_alias", "is_verified", "is_retired"]
    beers, beer_ids = _dense_table(db / "beers.parquet", beer_cols, "beer_id")
    beer_n = int(beer_ids.max()) + 2
    beer_brewer = _scatter(_array(beers, "brewer_id", fill=-1), beer_ids, beer_n, -1, np.int32)
    beer_style = _scatter(_array(beers, "style_id", fill=-1), beer_ids, beer_n, -1, np.int16)
    beer_abv = _scatter(_array(beers, "alcohol_pct", fill=np.nan), beer_ids, beer_n, np.nan, np.float32)
    beer_ibu = _scatter(_array(beers, "ibu", fill=np.nan), beer_ids, beer_n, np.nan, np.float32)
    beer_created = _scatter(_time_array(beers, "created_at"), beer_ids, beer_n, -1, np.int64)
    beer_updated = _scatter(_time_array(beers, "updated_at"), beer_ids, beer_n, -1, np.int64)
    beer_flags = np.column_stack([_scatter(_array(beers, c, fill=False), beer_ids, beer_n, 0, np.int8) for c in beer_cols[-5:]])
    styles, style_ids = _dense_table(db / "beer_styles.parquet", ["style_id", "parent_style_id", "category"], "style_id")
    style_n = max(158, int(style_ids.max()) + 2)
    style_parent = _scatter(_array(styles, "parent_style_id", fill=-1), style_ids, style_n, -1, np.int16)
    style_category = _scatter(_array(styles, "category", fill=-1), style_ids, style_n, -1, np.int16)
    brewer_cols = ["brewer_id", "country_id", "state_id", "type_id", "created_at", "updated_at", "is_out_of_business", "is_retired", "has_logo"]
    brewers, brewer_ids = _dense_table(db / "brewers.parquet", brewer_cols, "brewer_id")
    brewer_n = int(brewer_ids.max()) + 2
    brewer_country = _scatter(_array(brewers, "country_id", fill=-1), brewer_ids, brewer_n, -1, np.int16)
    brewer_state = _scatter(_array(brewers, "state_id", fill=-1), brewer_ids, brewer_n, -1, np.int16)
    brewer_type = _scatter(_array(brewers, "type_id", fill=-1), brewer_ids, brewer_n, -1, np.int8)
    brewer_created = _scatter(_time_array(brewers, "created_at"), brewer_ids, brewer_n, -1, np.int64)
    brewer_updated = _scatter(_time_array(brewers, "updated_at"), brewer_ids, brewer_n, -1, np.int64)
    brewer_flags = np.column_stack([_scatter(_array(brewers, c, fill=False), brewer_ids, brewer_n, 0, np.int8) for c in brewer_cols[-3:]])
    countries, country_ids = _dense_table(db / "countries.parquet", ["country_id", "continent"], "country_id")
    continent_values = pa.array(["North America", "South America", "Europe", "Asia", "Africa", "Oceania", "Antarctica"])
    continent_arrow = pc.index_in(countries["continent"].combine_chunks(), value_set=continent_values)
    continent = pc.fill_null(continent_arrow, pa.scalar(-1, type=continent_arrow.type)).to_numpy(zero_copy_only=False)
    country_continent = _scatter(continent, country_ids, 252, -1, np.int8)
    users, user_ids = _dense_table(db / "users.parquet", ["user_id", "created_at", "updated_at", "user_type"], "user_id")
    user_n = int(user_ids.max()) + 2
    user_created = _scatter(_time_array(users, "created_at"), user_ids, user_n, -1, np.int64)
    user_updated = _scatter(_time_array(users, "updated_at"), user_ids, user_n, -1, np.int64)
    user_types = pa.array(["Member", "Premium", "Admin", "Brewer", "Place", "Industry"])
    user_type_arrow = pc.index_in(users["user_type"].combine_chunks(), value_set=user_types)
    user_type_values = pc.fill_null(user_type_arrow, pa.scalar(-1, type=user_type_arrow.type)).to_numpy(zero_copy_only=False)
    user_type = _scatter(user_type_values, user_ids, user_n, -1, np.int8)
    return Metadata(
        beer_brewer=beer_brewer,
        beer_style=beer_style,
        beer_abv=beer_abv,
        beer_ibu=beer_ibu,
        beer_created=beer_created,
        beer_updated=beer_updated,
        beer_flags=beer_flags,
        style_parent=style_parent,
        style_category=style_category,
        brewer_country=brewer_country,
        brewer_state=brewer_state,
        brewer_type=brewer_type,
        brewer_created=brewer_created,
        brewer_updated=brewer_updated,
        brewer_flags=brewer_flags,
        country_continent=country_continent,
        user_created=user_created,
        user_updated=user_updated,
        user_type=user_type,
    )


def entity_keys(events: Events, metadata: Metadata) -> dict[str, np.ndarray]:
    beer_index = np.where(events.beer >= 0, events.beer, len(metadata.beer_style) - 1)
    user_index = np.where(events.user >= 0, events.user, len(metadata.user_created) - 1)
    style = metadata.beer_style[beer_index].astype(np.int32)
    style_index = np.where(style >= 0, style, len(metadata.style_parent) - 1)
    parent = metadata.style_parent[style_index].astype(np.int32)
    brewer = metadata.beer_brewer[beer_index].astype(np.int32)
    brewer_index = np.where(brewer >= 0, brewer, len(metadata.brewer_country) - 1)
    country = metadata.brewer_country[brewer_index].astype(np.int32)
    abv = metadata.beer_abv[beer_index]
    abv_band = np.where(np.isfinite(abv), np.clip(np.floor(abv * 2), 0, 80), 81).astype(np.int32)
    return {
        "beer_index": beer_index.astype(np.int32),
        "user_index": user_index.astype(np.int32),
        "style": style,
        "parent": parent,
        "brewer": brewer,
        "country": country,
        "abv_band": abv_band,
    }


@njit(cache=True)
def _behavior_user(order, user, beer, style, brewer, times, out, beer_n, style_n, brewer_n):
    seen_beer = np.full(beer_n, -2, np.int32)
    seen_style = np.full(style_n, -2, np.int32)
    seen_brewer = np.full(brewer_n, -2, np.int32)
    start = 0
    while start < len(order):
        end = start + 1
        uid = user[order[start]]
        while end < len(order) and user[order[end]] == uid:
            end += 1
        p90 = start
        p365 = start
        distinct_beer = 0
        distinct_style = 0
        distinct_brewer = 0
        session = 0
        last = -1
        for pos in range(start, end):
            row = order[pos]
            now = times[row]
            while p90 < pos and times[order[p90]] < now - 90 * DAY:
                p90 += 1
            while p365 < pos and times[order[p365]] < now - 365 * DAY:
                p365 += 1
            count = pos - start
            gap = (now - last) / DAY if last >= 0 else np.nan
            if last >= 0 and now - last <= 6 * 3600:
                session += 1
            else:
                session = 1
            bid = beer[row] + 1
            sid = style[row] + 1
            brid = brewer[row] + 1
            new_beer = seen_beer[bid] != uid
            new_style = seen_style[sid] != uid
            new_brewer = seen_brewer[brid] != uid
            out[row, 0] = count
            out[row, 1] = np.log1p(count)
            out[row, 2] = np.log1p(pos - p90)
            out[row, 3] = np.log1p(pos - p365)
            out[row, 4] = np.log1p(gap) if gap >= 0 else np.nan
            out[row, 5] = session
            out[row, 6] = np.log1p(distinct_beer)
            out[row, 7] = np.log1p(distinct_style)
            out[row, 8] = np.log1p(distinct_brewer)
            out[row, 9] = distinct_beer / count if count else 1.0
            out[row, 10] = new_beer
            out[row, 11] = new_style
            out[row, 12] = new_brewer
            if new_beer:
                distinct_beer += 1
                seen_beer[bid] = uid
            if new_style:
                distinct_style += 1
                seen_style[sid] = uid
            if new_brewer:
                distinct_brewer += 1
                seen_brewer[brid] = uid
            last = now
        start = end


@njit(cache=True)
def _behavior_beer(order, user, beer, times, out, user_n):
    seen_user = np.full(user_n, -2, np.int32)
    start = 0
    while start < len(order):
        end = start + 1
        bid = beer[order[start]]
        while end < len(order) and beer[order[end]] == bid:
            end += 1
        p90 = start
        p365 = start
        distinct = 0
        last = -1
        for pos in range(start, end):
            row = order[pos]
            now = times[row]
            while p90 < pos and times[order[p90]] < now - 90 * DAY:
                p90 += 1
            while p365 < pos and times[order[p365]] < now - 365 * DAY:
                p365 += 1
            count = pos - start
            gap = (now - last) / DAY if last >= 0 else np.nan
            uid = user[row] + 1
            new_user = seen_user[uid] != bid
            recent90 = pos - p90
            recent365 = pos - p365
            out[row, 0] = count
            out[row, 1] = np.log1p(count)
            out[row, 2] = np.log1p(recent90)
            out[row, 3] = np.log1p(recent365)
            out[row, 4] = np.log1p(recent90 / 90.0)
            out[row, 5] = np.log1p(recent365 / 365.0)
            out[row, 6] = np.log1p(gap) if gap >= 0 else np.nan
            out[row, 7] = np.log1p(distinct)
            out[row, 8] = new_user
            if new_user:
                distinct += 1
                seen_user[uid] = bid
            last = now
        start = end


@njit(cache=True)
def _behavior_pair(order, keys, times, out):
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        p90 = start
        p365 = start
        last = -1
        for pos in range(start, end):
            row = order[pos]
            now = times[row]
            while p90 < pos and times[order[p90]] < now - 90 * DAY:
                p90 += 1
            while p365 < pos and times[order[p365]] < now - 365 * DAY:
                p365 += 1
            count = pos - start
            gap = (now - last) / DAY if last >= 0 else np.nan
            out[row, 0] = count
            out[row, 1] = np.log1p(count)
            out[row, 2] = np.log1p(pos - p90)
            out[row, 3] = np.log1p(pos - p365)
            out[row, 4] = np.log1p(gap) if gap >= 0 else np.nan
            last = now
        start = end


def _pair_behavior_features(events: Events, keys: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    groups = {
        "user_beer": _pair(keys["user_index"], keys["beer_index"], 1191665),
        "user_style": _pair(keys["user_index"], keys["style"], 160),
        "user_parent": _pair(keys["user_index"], keys["parent"], 160),
        "user_brewer": _pair(keys["user_index"], keys["brewer"], 50020),
        "user_abv": _pair(keys["user_index"], keys["abv_band"], 83),
    }
    blocks = []
    names = []
    for name, group_key in groups.items():
        order = np.lexsort((events.rating_id, events.time, group_key)).astype(np.int64)
        block = np.empty((len(events.time), 5), np.float32)
        _behavior_pair(order, group_key, events.time, block)
        blocks.append(block)
        names.extend([f"event_{name}_count", f"event_{name}_count_log", f"event_{name}_recent90_log", f"event_{name}_recent365_log", f"event_{name}_gap_log_days"])
    return np.column_stack(blocks).astype(np.float32), names


@njit(cache=True)
def _created_counts(event_order, event_time, event_brewer, event_style, event_country, created_order, created_time, created_brewer, created_style, created_country, out, brewer_n, style_n, country_n):
    brewer_count = np.zeros(brewer_n, np.int32)
    style_count = np.zeros(style_n, np.int32)
    country_count = np.zeros(country_n, np.int32)
    pointer = 0
    total = 0
    for pos in range(len(event_order)):
        row = event_order[pos]
        now = event_time[row]
        while pointer < len(created_order) and created_time[created_order[pointer]] <= now:
            beer_row = created_order[pointer]
            br = created_brewer[beer_row] + 1
            st = created_style[beer_row] + 1
            co = created_country[beer_row] + 1
            brewer_count[br] += 1
            style_count[st] += 1
            country_count[co] += 1
            total += 1
            pointer += 1
        out[row, 0] = np.log1p(total)
        out[row, 1] = np.log1p(brewer_count[event_brewer[row] + 1])
        out[row, 2] = np.log1p(style_count[event_style[row] + 1])
        out[row, 3] = np.log1p(country_count[event_country[row] + 1])


def _read_place_features(events: Events, event_country: np.ndarray) -> tuple[np.ndarray, list[str]]:
    db = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"
    score_cols = ["total_score", "ambiance", "service", "selection", "food", "value", "overall"]
    table = pq.read_table(db / "place_ratings.parquet", columns=["user_id", "place_id", "created_at"] + score_cols)
    user = _array(table, "user_id", fill=-1, dtype=np.int32)
    times = _time_array(table, "created_at")
    values = np.column_stack([_array(table, c, fill=np.nan, dtype=np.float32) for c in score_cols])
    places = pq.read_table(db / "places.parquet", columns=["place_id", "type_id", "country_id"])
    place_ids = _array(places, "place_id", dtype=np.int64)
    place_n = int(place_ids.max()) + 2
    place_type_lookup = _scatter(_array(places, "type_id", fill=-1), place_ids, place_n, -1, np.int16)
    place_country_lookup = _scatter(_array(places, "country_id", fill=-1), place_ids, place_n, -1, np.int16)
    place_id = _array(table, "place_id", fill=-1, dtype=np.int32)
    place_index = np.where(place_id >= 0, place_id, place_n - 1)
    place_type = place_type_lookup[place_index].astype(np.int32)
    place_country = place_country_lookup[place_index].astype(np.int32)
    event_order = np.lexsort((events.rating_id, events.time)).astype(np.int64)
    place_order = np.argsort(times, kind="stable").astype(np.int64)
    out = np.full((len(events.time), 29), np.nan, np.float32)
    _place_history(event_order, events.time, events.user, event_country, place_order, times, user, place_type, place_country, values, out, max(int(events.user.max()) + 2, 237001))
    names = ["place_count", "place_count_log", "place_recency_log_days", "place_total_mean", "place_ambiance_mean", "place_service_mean", "place_selection_mean", "place_food_mean", "place_value_mean", "place_overall_mean"]
    for place_type_id in range(8):
        names.extend([f"place_type_{place_type_id}_count_log", f"place_type_{place_type_id}_mean"])
    names.extend(["place_same_country_count", "place_same_country_count_log", "place_same_country_mean"])
    return out, names


@njit(cache=True)
def _place_history(event_order, event_time, event_user, event_country, place_order, place_time, place_user, place_type, place_country, values, out, user_n):
    count = np.zeros(user_n, np.int32)
    sums = np.zeros((user_n, values.shape[1]), np.float64)
    type_count = np.zeros((user_n, 9), np.int32)
    type_sum = np.zeros((user_n, 9), np.float32)
    country_count = np.zeros((user_n, 252), np.int32)
    country_sum = np.zeros((user_n, 252), np.float32)
    last = np.full(user_n, -1, np.int64)
    pointer = 0
    for pos in range(len(event_order)):
        row = event_order[pos]
        now = event_time[row]
        while pointer < len(place_order) and place_time[place_order[pointer]] <= now:
            source = place_order[pointer]
            uid = place_user[source] + 1
            if uid >= 0 and uid < user_n:
                count[uid] += 1
                for j in range(values.shape[1]):
                    if np.isfinite(values[source, j]):
                        sums[uid, j] += values[source, j]
                type_id = place_type[source] + 1
                country_id = place_country[source] + 1
                if type_id >= 0 and type_id < 9:
                    type_count[uid, type_id] += 1
                    if np.isfinite(values[source, 0]):
                        type_sum[uid, type_id] += values[source, 0]
                if country_id >= 0 and country_id < 252:
                    country_count[uid, country_id] += 1
                    if np.isfinite(values[source, 0]):
                        country_sum[uid, country_id] += values[source, 0]
                last[uid] = place_time[source]
            pointer += 1
        uid = event_user[row] + 1
        c = count[uid]
        out[row, 0] = c
        out[row, 1] = np.log1p(c)
        out[row, 2] = np.log1p((now - last[uid]) / DAY) if c and last[uid] >= 0 else np.nan
        for j in range(values.shape[1]):
            out[row, j + 3] = sums[uid, j] / c if c else np.nan
        output_column = 10
        for type_id in range(1, 9):
            tc = type_count[uid, type_id]
            out[row, output_column] = np.log1p(tc)
            out[row, output_column + 1] = type_sum[uid, type_id] / tc if tc else np.nan
            output_column += 2
        country_id = event_country[row] + 1
        cc = country_count[uid, country_id]
        out[row, 26] = cc
        out[row, 27] = np.log1p(cc)
        out[row, 28] = country_sum[uid, country_id] / cc if cc else np.nan


def _read_favorite_features(events: Events, metadata: Metadata, keys: dict[str, np.ndarray], beer_n: int) -> tuple[np.ndarray, list[str]]:
    path = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db" / "favorites.parquet"
    table = pq.read_table(path, columns=["user_id", "beer_id", "created_at"])
    user = _array(table, "user_id", fill=-1, dtype=np.int32)
    beer = _array(table, "beer_id", fill=-1, dtype=np.int32)
    times = _time_array(table, "created_at")
    event_order = np.lexsort((events.rating_id, events.time)).astype(np.int64)
    favorite_order = np.argsort(times, kind="stable").astype(np.int64)
    out = np.zeros((len(events.time), 8), np.float32)
    _favorite_counts(event_order, events.time, events.user, events.beer, favorite_order, times, user, beer, out, max(int(events.user.max()) + 2, 237001), beer_n)
    multiplier = np.int64(beer_n)
    favorite_keys = (user.astype(np.int64) + 1) * multiplier + beer.astype(np.int64) + 1
    order = np.lexsort((times, favorite_keys))
    sorted_keys = favorite_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique_keys = sorted_keys[starts]
    first_times = np.minimum.reduceat(times[order], starts)
    event_keys = (events.user.astype(np.int64) + 1) * multiplier + events.beer.astype(np.int64) + 1
    positions = np.searchsorted(unique_keys, event_keys)
    valid = positions < len(unique_keys)
    matched = np.zeros(len(events.time), dtype=bool)
    matched[valid] = unique_keys[positions[valid]] == event_keys[valid]
    prior = np.zeros(len(events.time), dtype=np.float32)
    valid_positions = np.flatnonzero(matched)
    prior[valid_positions] = (first_times[positions[valid_positions]] <= events.time[valid_positions]).astype(np.float32)
    out[:, 3] = prior
    favorite_beer_index = np.where(beer >= 0, beer, len(metadata.beer_style) - 1)
    favorite_style = metadata.beer_style[favorite_beer_index].astype(np.int32)
    favorite_brewer = metadata.beer_brewer[favorite_beer_index].astype(np.int32)
    event_style_key = _pair(events.user, keys["style"], 160)
    favorite_style_key = _pair(user, favorite_style, 160)
    event_brewer_key = _pair(events.user, keys["brewer"], 50020)
    favorite_brewer_key = _pair(user, favorite_brewer, 50020)
    style_count = _prior_temporal_pair_counts(favorite_style_key, times, event_style_key, events.time)
    brewer_count = _prior_temporal_pair_counts(favorite_brewer_key, times, event_brewer_key, events.time)
    out[:, 4] = np.log1p(style_count)
    out[:, 5] = np.log1p(brewer_count)
    out[:, 6] = style_count > 0
    out[:, 7] = brewer_count > 0
    names = ["favorite_user_count_log", "favorite_beer_count_log", "favorite_user_any", "favorite_current_beer_prior", "favorite_current_style_count_log", "favorite_current_brewer_count_log", "favorite_current_style_prior", "favorite_current_brewer_prior"]
    return out, names


@njit(cache=True)
def _pair_time_lookup(sorted_keys, sorted_times, event_keys, event_times, out):
    for row in range(len(event_keys)):
        key = event_keys[row]
        left = np.searchsorted(sorted_keys, key, side="left")
        right = np.searchsorted(sorted_keys, key, side="right")
        if right > left:
            out[row] = np.searchsorted(sorted_times[left:right], event_times[row], side="right")


def _prior_temporal_pair_counts(source_keys: np.ndarray, source_times: np.ndarray, event_keys: np.ndarray, event_times: np.ndarray) -> np.ndarray:
    order = np.lexsort((source_times, source_keys))
    sorted_keys = source_keys[order]
    sorted_times = source_times[order]
    out = np.zeros(len(event_keys), np.int32)
    _pair_time_lookup(sorted_keys, sorted_times, event_keys, event_times, out)
    return out


@njit(cache=True)
def _favorite_counts(event_order, event_time, event_user, event_beer, favorite_order, favorite_time, favorite_user, favorite_beer, out, user_n, beer_n):
    user_count = np.zeros(user_n, np.int32)
    beer_count = np.zeros(beer_n, np.int32)
    pointer = 0
    for pos in range(len(event_order)):
        row = event_order[pos]
        now = event_time[row]
        while pointer < len(favorite_order) and favorite_time[favorite_order[pointer]] <= now:
            source = favorite_order[pointer]
            user_count[favorite_user[source] + 1] += 1
            beer_count[favorite_beer[source] + 1] += 1
            pointer += 1
        uc = user_count[event_user[row] + 1]
        bc = beer_count[event_beer[row] + 1]
        out[row, 0] = np.log1p(uc)
        out[row, 1] = np.log1p(bc)
        out[row, 2] = uc > 0


def _metadata_features(events: Events, metadata: Metadata, keys: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    beer_index = keys["beer_index"]
    user_index = keys["user_index"]
    brewer_index = np.where(keys["brewer"] >= 0, keys["brewer"], len(metadata.brewer_country) - 1)
    style_index = np.where(keys["style"] >= 0, keys["style"], len(metadata.style_parent) - 1)
    country_index = np.where(keys["country"] >= 0, keys["country"], len(metadata.country_continent) - 1)
    beer_created = metadata.beer_created[beer_index]
    beer_updated = metadata.beer_updated[beer_index]
    brewer_created = metadata.brewer_created[brewer_index]
    brewer_updated = metadata.brewer_updated[brewer_index]
    user_created = metadata.user_created[user_index]
    user_updated = metadata.user_updated[user_index]
    beer_safe = (beer_updated < 0) | (beer_updated <= events.time)
    brewer_safe = (brewer_updated < 0) | (brewer_updated <= events.time)
    user_safe = (user_updated < 0) | (user_updated <= events.time)
    abv = metadata.beer_abv[beer_index].copy()
    ibu = metadata.beer_ibu[beer_index].copy()
    abv[~beer_safe] = np.nan
    ibu[~beer_safe] = np.nan
    beer_flags = metadata.beer_flags[beer_index].astype(np.float32)
    beer_flags[~beer_safe] = np.nan
    brewer_flags = metadata.brewer_flags[brewer_index].astype(np.float32)
    brewer_flags[~brewer_safe] = np.nan
    user_type = metadata.user_type[user_index].astype(np.float32)
    user_type[~user_safe] = np.nan
    beer_age = np.where(beer_created >= 0, np.log1p(np.maximum(events.time - beer_created, 0) / DAY), np.nan)
    brewer_age = np.where(brewer_created >= 0, np.log1p(np.maximum(events.time - brewer_created, 0) / DAY), np.nan)
    user_age = np.where(user_created >= 0, np.log1p(np.maximum(events.time - user_created, 0) / DAY), np.nan)
    values = [
        keys["style"], keys["parent"], metadata.style_category[style_index], keys["brewer"], keys["country"], metadata.brewer_state[brewer_index], metadata.brewer_type[brewer_index], metadata.country_continent[country_index], keys["abv_band"], abv, ibu, ~np.isfinite(abv), ~np.isfinite(ibu), beer_age, brewer_age, user_age, beer_safe, brewer_safe, user_safe, user_type,
    ]
    names = ["style_id", "parent_style_id", "style_category", "brewer_id", "brewer_country", "brewer_state", "brewer_type", "continent", "abv_band", "abv", "ibu", "abv_missing", "ibu_missing", "beer_age_log_days", "brewer_age_log_days", "user_age_log_days", "beer_metadata_safe", "brewer_metadata_safe", "user_metadata_safe", "user_type"]
    for i in range(beer_flags.shape[1]):
        values.append(beer_flags[:, i])
        names.append(["beer_seasonal", "beer_one_off", "beer_alias", "beer_verified", "beer_retired"][i])
    for i in range(brewer_flags.shape[1]):
        values.append(brewer_flags[:, i])
        names.append(["brewer_out_of_business", "brewer_retired", "brewer_has_logo"][i])
    result = np.column_stack(values).astype(np.float32)
    return result, names


def _calendar_features(events: Events) -> tuple[np.ndarray, list[str]]:
    month_index = events.time.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    month = month_index % 12
    year = month_index // 12 + 1970
    day_index = events.time // DAY
    weekday = (day_index + 3) % 7
    hour = (events.time % DAY) / 3600
    result = np.column_stack([
        year - 2010,
        np.sin(2 * np.pi * month / 12),
        np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * weekday / 7),
        np.cos(2 * np.pi * weekday / 7),
        np.sin(2 * np.pi * hour / 24),
        np.cos(2 * np.pi * hour / 24),
        (events.time - V_TIME) / (365.25 * DAY),
        events.language,
        events.language < 0,
    ]).astype(np.float32)
    names = ["calendar_year", "month_sin", "month_cos", "weekday_sin", "weekday_cos", "hour_sin", "hour_cos", "years_since_v", "language", "language_other"]
    return result, names


def build_base_features(events: Events, metadata: Metadata, keys: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    n = len(events.time)
    user_behavior = np.empty((n, 13), np.float32)
    user_order = np.lexsort((events.rating_id, events.time, events.user)).astype(np.int64)
    _behavior_user(user_order, events.user, events.beer, keys["style"], keys["brewer"], events.time, user_behavior, len(metadata.beer_style) + 1, len(metadata.style_parent) + 1, len(metadata.brewer_country) + 1)
    del user_order
    beer_behavior = np.empty((n, 9), np.float32)
    beer_order = np.lexsort((events.rating_id, events.time, events.beer)).astype(np.int64)
    _behavior_beer(beer_order, events.user, events.beer, events.time, beer_behavior, len(metadata.user_created) + 1)
    del beer_order
    pair_behavior, pair_names = _pair_behavior_features(events, keys)
    event_order = np.lexsort((events.rating_id, events.time)).astype(np.int64)
    valid_created = np.flatnonzero(metadata.beer_created >= 0)
    created_order = valid_created[np.argsort(metadata.beer_created[valid_created], kind="stable")].astype(np.int64)
    created_brewer = metadata.beer_brewer
    created_style = metadata.beer_style.astype(np.int32)
    created_brewer_index = np.where(created_brewer >= 0, created_brewer, len(metadata.brewer_country) - 1)
    created_country = metadata.brewer_country[created_brewer_index].astype(np.int32)
    created = np.empty((n, 4), np.float32)
    _created_counts(event_order, events.time, keys["brewer"], keys["style"], keys["country"], created_order, metadata.beer_created, created_brewer, created_style, created_country, created, len(metadata.brewer_country) + 1, len(metadata.style_parent) + 1, len(metadata.country_continent) + 1)
    metadata_features, metadata_names = _metadata_features(events, metadata, keys)
    place, place_names = _read_place_features(events, keys["country"])
    favorite, favorite_names = _read_favorite_features(events, metadata, keys, len(metadata.beer_style) + 1)
    calendar, calendar_names = _calendar_features(events)
    blocks = [user_behavior, beer_behavior, pair_behavior, created, place, favorite, metadata_features, calendar]
    names = [
        "event_user_count", "event_user_count_log", "event_user_recent90_log", "event_user_recent365_log", "event_user_gap_log_days", "event_user_session_position", "event_user_distinct_beers_log", "event_user_distinct_styles_log", "event_user_distinct_brewers_log", "event_user_exploration_rate", "event_user_new_beer", "event_user_new_style", "event_user_new_brewer",
        "event_beer_count", "event_beer_count_log", "event_beer_recent90_log", "event_beer_recent365_log", "event_beer_velocity90_log", "event_beer_velocity365_log", "event_beer_gap_log_days", "event_beer_distinct_raters_log", "event_beer_new_rater",
    ] + pair_names + [
        "created_beers_global_log", "created_beers_brewer_log", "created_beers_style_log", "created_beers_country_log",
    ] + place_names + favorite_names + metadata_names + calendar_names
    return np.column_stack(blocks).astype(np.float32), names


@njit(cache=True)
def _prior_history(order, keys, times, values, out):
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        p90 = start
        p365 = start
        sum90 = 0.0
        sum365 = 0.0
        total = 0.0
        square = 0.0
        sum_time = 0.0
        sum_time2 = 0.0
        sum_time_y = 0.0
        first_time = times[order[start]]
        last_time = -1
        for pos in range(start, end):
            row = order[pos]
            now = times[row]
            while p90 < pos and times[order[p90]] < now - 90 * DAY:
                sum90 -= values[order[p90]]
                p90 += 1
            while p365 < pos and times[order[p365]] < now - 365 * DAY:
                sum365 -= values[order[p365]]
                p365 += 1
            count = pos - start
            mean = total / count if count else np.nan
            variance = max(square / count - mean * mean, 0.0) if count else np.nan
            last5 = 0.0
            last10 = 0.0
            last25 = 0.0
            for back in range(1, 26):
                if pos - back < start:
                    break
                val = values[order[pos - back]]
                last25 += val
                if back <= 10:
                    last10 += val
                if back <= 5:
                    last5 += val
            n5 = min(count, 5)
            n10 = min(count, 10)
            n25 = min(count, 25)
            denom = count * sum_time2 - sum_time * sum_time
            slope = (count * sum_time_y - sum_time * total) / denom * 365.25 if count > 1 and denom > 0 else 0.0
            out[row, 0] = count
            out[row, 1] = np.log1p(count)
            out[row, 2] = mean
            out[row, 3] = variance
            out[row, 4] = last5 / n5 if n5 else np.nan
            out[row, 5] = last10 / n10 if n10 else np.nan
            out[row, 6] = last25 / n25 if n25 else np.nan
            out[row, 7] = sum90 / (pos - p90) if pos > p90 else np.nan
            out[row, 8] = sum365 / (pos - p365) if pos > p365 else np.nan
            out[row, 9] = max(-1.0, min(1.0, slope))
            out[row, 10] = np.log1p(max(now - first_time, 0) / DAY) if count else 0.0
            out[row, 11] = np.log1p(max(now - last_time, 0) / DAY) if count else np.nan
            out[row, 12] = count == 0
            val = values[row]
            day = (now - first_time) / DAY
            total += val
            square += val * val
            sum_time += day
            sum_time2 += day * day
            sum_time_y += day * val
            sum90 += val
            sum365 += val
            last_time = now
        start = end


def prior_history(keys: np.ndarray, times: np.ndarray, rating_id: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.lexsort((rating_id, times, keys)).astype(np.int64)
    out = np.empty((len(keys), 13), np.float32)
    _prior_history(order, keys, times, values, out)
    return out


@njit(cache=True)
def _prior_stats(order, keys, values, out):
    start = 0
    while start < len(order):
        end = start + 1
        key = keys[order[start]]
        while end < len(order) and keys[order[end]] == key:
            end += 1
        total = 0.0
        square = 0.0
        for pos in range(start, end):
            row = order[pos]
            count = pos - start
            out[row, 0] = count
            out[row, 1] = total
            out[row, 2] = square
            value = values[row]
            total += value
            square += value * value
        start = end


def prior_stats(keys: np.ndarray, times: np.ndarray, rating_id: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.lexsort((rating_id, times, keys)).astype(np.int64)
    out = np.empty((len(keys), 3), np.float32)
    _prior_stats(order, keys, values.astype(np.float32), out)
    return out


def frozen_stats(label_keys: np.ndarray, label_values: np.ndarray, serve_keys: np.ndarray) -> np.ndarray:
    order = np.argsort(label_keys, kind="stable")
    sorted_keys = label_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique = sorted_keys[starts]
    count = np.diff(np.r_[starts, len(order)]).astype(np.float32)
    values = label_values[order].astype(np.float64)
    sums = np.add.reduceat(values, starts)
    squares = np.add.reduceat(values * values, starts)
    position = np.searchsorted(unique, serve_keys)
    valid = position < len(unique)
    matched = np.zeros(len(serve_keys), dtype=bool)
    matched[valid] = unique[position[valid]] == serve_keys[valid]
    out = np.zeros((len(serve_keys), 3), np.float32)
    rows = np.flatnonzero(matched)
    out[rows, 0] = count[position[rows]]
    out[rows, 1] = sums[position[rows]]
    out[rows, 2] = squares[position[rows]]
    return out


def frozen_history(label_keys: np.ndarray, label_times: np.ndarray, label_values: np.ndarray, serve_keys: np.ndarray, serve_times: np.ndarray, cutoff: int) -> np.ndarray:
    order = np.lexsort((label_times, label_keys))
    sorted_keys = label_keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    unique = sorted_keys[starts]
    state = np.empty((len(unique), 11), np.float32)
    for group_index, start in enumerate(starts):
        end = starts[group_index + 1] if group_index + 1 < len(starts) else len(order)
        rows = order[start:end]
        vals = label_values[rows].astype(np.float64)
        ts = label_times[rows]
        count = len(rows)
        centered = (ts - ts[0]) / DAY
        denom = count * np.dot(centered, centered) - centered.sum() ** 2
        slope = (count * np.dot(centered, vals) - centered.sum() * vals.sum()) / denom * 365.25 if count > 1 and denom > 0 else 0.0
        state[group_index] = [count, np.log1p(count), vals.mean(), vals.var(), vals[-5:].mean(), vals[-10:].mean(), vals[-25:].mean(), vals[ts >= cutoff - 90 * DAY].mean() if np.any(ts >= cutoff - 90 * DAY) else np.nan, vals[ts >= cutoff - 365 * DAY].mean() if np.any(ts >= cutoff - 365 * DAY) else np.nan, np.clip(slope, -1, 1), ts[0]]
    position = np.searchsorted(unique, serve_keys)
    valid = position < len(unique)
    matched = np.zeros(len(serve_keys), dtype=bool)
    matched[valid] = unique[position[valid]] == serve_keys[valid]
    out = np.full((len(serve_keys), 13), np.nan, np.float32)
    rows = np.flatnonzero(matched)
    out[rows, :10] = state[position[rows], :10]
    first = state[position[rows], 10].astype(np.int64)
    last_indices = np.searchsorted(sorted_keys, serve_keys[rows], side="right") - 1
    last = label_times[order[last_indices]]
    out[rows, 10] = np.log1p(np.maximum(serve_times[rows] - first, 0) / DAY)
    out[rows, 11] = np.log1p(np.maximum(serve_times[rows] - last, 0) / DAY)
    out[:, 12] = (~matched).astype(np.float32)
    out[~matched, 0:2] = 0
    return out


def _pair(a: np.ndarray, b: np.ndarray, cardinality: int) -> np.ndarray:
    return (a.astype(np.int64) + 1) * np.int64(cardinality) + b.astype(np.int64) + 1


def _hierarchy_features(stats: dict[str, np.ndarray], global_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
    parent = (stats["parent"][:, 1] + 80 * global_mean) / (stats["parent"][:, 0] + 80)
    style = (stats["style"][:, 1] + 80 * parent) / (stats["style"][:, 0] + 80)
    country = (stats["country"][:, 1] + 80 * global_mean) / (stats["country"][:, 0] + 80)
    brewer_base = 0.5 * style + 0.5 * country
    brewer = (stats["brewer"][:, 1] + 80 * brewer_base) / (stats["brewer"][:, 0] + 80)
    beer_base = 0.5 * style + 0.5 * brewer
    beer = (stats["beer"][:, 1] + 40 * beer_base) / (stats["beer"][:, 0] + 40)
    abv = (stats["abv"][:, 1] + 80 * global_mean) / (stats["abv"][:, 0] + 80)
    values = [global_mean]
    names = ["label_global_mean"]
    for name, estimate in [("parent", parent), ("style", style), ("country", country), ("brewer", brewer), ("beer", beer), ("abv_band", abv)]:
        source = stats["abv" if name == "abv_band" else name]
        values.extend([np.log1p(source[:, 0]), estimate, source[:, 0] == 0])
        names.extend([f"label_{name}_count_log", f"label_{name}_eb_mean", f"label_{name}_cold"])
    return np.column_stack(values).astype(np.float32), beer.astype(np.float32), names


def _strict_label_features(events: Events, keys: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = events.n_tv
    times = events.time[:n]
    rid = events.rating_id[:n]
    y = np.concatenate((events.train_y, events.val_y)).astype(np.float32)
    user = keys["user_index"][:n]
    beer = keys["beer_index"][:n]
    user_history = prior_history(user, times, rid, y)
    beer_history = prior_history(beer, times, rid, y)
    chronological = np.lexsort((rid, times))
    global_mean = np.empty(n, np.float32)
    total = 0.0
    for position, row in enumerate(chronological):
        global_mean[row] = total / position if position else DEFAULT_SCORE
        total += float(y[row])
    stats = {}
    for name in ["parent", "style", "country", "brewer", "beer", "abv_band"]:
        source = beer if name == "beer" else keys[name][:n]
        stats["abv" if name == "abv_band" else name] = prior_stats(source, times, rid, y)
    hierarchy, beer_prior, hierarchy_names = _hierarchy_features(stats, global_mean)
    residual = y - beer_prior
    affinity_blocks = []
    affinity_names = []
    affinity_keys = {
        "user_generosity": user.astype(np.int64),
        "user_style": _pair(user, keys["style"][:n], 160),
        "user_parent": _pair(user, keys["parent"][:n], 160),
        "user_brewer": _pair(user, keys["brewer"][:n], 50020),
        "user_abv": _pair(user, keys["abv_band"][:n], 83),
    }
    user_affinity = None
    for name, group_key in affinity_keys.items():
        source = prior_stats(group_key, times, rid, residual)
        mean = source[:, 1] / (source[:, 0] + 100)
        block = np.column_stack((np.log1p(source[:, 0]), mean, source[:, 0] == 0)).astype(np.float32)
        affinity_blocks.append(block)
        affinity_names.extend([f"{name}_count_log", f"{name}_residual", f"{name}_cold"])
        if name == "user_generosity":
            user_affinity = mean.astype(np.float32)
    cohort_stats = prior_stats(beer, times, rid, user_affinity)
    cohort_mean = cohort_stats[:, 1] / np.maximum(cohort_stats[:, 0], 1)
    cohort_var = np.maximum(cohort_stats[:, 2] / np.maximum(cohort_stats[:, 0], 1) - cohort_mean * cohort_mean, 0)
    cohort = np.column_stack((np.log1p(cohort_stats[:, 0]), cohort_mean, np.sqrt(cohort_var), cohort_stats[:, 0] == 0)).astype(np.float32)
    history_names = ["count", "count_log", "mean", "variance", "last5_mean", "last10_mean", "last25_mean", "trailing90_mean", "trailing365_mean", "slope_year", "tenure_log_days", "staleness_log_days", "cold"]
    names = [f"label_user_{name}" for name in history_names] + [f"label_beer_{name}" for name in history_names] + hierarchy_names + affinity_names + ["cohort_count_log", "cohort_generosity_mean", "cohort_generosity_std", "cohort_cold"]
    matrix = np.column_stack([user_history, beer_history, hierarchy] + affinity_blocks + [cohort]).astype(np.float32)
    return matrix, residual.astype(np.float32), names


def _frozen_label_features_indices(events: Events, keys: dict[str, np.ndarray], label_index: np.ndarray, serve_index: np.ndarray, history_index: np.ndarray, residual: np.ndarray, cutoff: int) -> np.ndarray:
    all_y = np.concatenate((events.train_y, events.val_y)).astype(np.float32)
    label_times = events.time[label_index]
    label_y = all_y[label_index]
    serve_times = events.time[serve_index]
    label_user = keys["user_index"][label_index]
    serve_user = keys["user_index"][serve_index]
    label_beer = keys["beer_index"][label_index]
    serve_beer = keys["beer_index"][serve_index]
    label_residual = residual[label_index]
    user_history = frozen_history(label_user, label_times, label_y, serve_user, serve_times, cutoff)
    beer_history = frozen_history(label_beer, label_times, label_y, serve_beer, serve_times, cutoff)
    global_mean = np.full(len(serve_times), float(label_y.mean()), np.float32)
    stats = {}
    for name in ["parent", "style", "country", "brewer", "beer", "abv_band"]:
        label_key = label_beer if name == "beer" else keys[name][label_index]
        serve_key = serve_beer if name == "beer" else keys[name][serve_index]
        stats["abv" if name == "abv_band" else name] = frozen_stats(label_key, label_y, serve_key)
    hierarchy, _, _ = _hierarchy_features(stats, global_mean)
    affinity_blocks = []
    label_affinity_keys = {
        "user_generosity": label_user.astype(np.int64),
        "user_style": _pair(label_user, keys["style"][label_index], 160),
        "user_parent": _pair(label_user, keys["parent"][label_index], 160),
        "user_brewer": _pair(label_user, keys["brewer"][label_index], 50020),
        "user_abv": _pair(label_user, keys["abv_band"][label_index], 83),
    }
    serve_affinity_keys = {
        "user_generosity": serve_user.astype(np.int64),
        "user_style": _pair(serve_user, keys["style"][serve_index], 160),
        "user_parent": _pair(serve_user, keys["parent"][serve_index], 160),
        "user_brewer": _pair(serve_user, keys["brewer"][serve_index], 50020),
        "user_abv": _pair(serve_user, keys["abv_band"][serve_index], 83),
    }
    for name in label_affinity_keys:
        source = frozen_stats(label_affinity_keys[name], label_residual, serve_affinity_keys[name])
        mean = source[:, 1] / (source[:, 0] + 100)
        affinity_blocks.append(np.column_stack((np.log1p(source[:, 0]), mean, source[:, 0] == 0)).astype(np.float32))
    history_user = keys["user_index"][history_index]
    history_beer = keys["beer_index"][history_index]
    history_generosity_stats = frozen_stats(label_user, label_residual, history_user)
    history_generosity = history_generosity_stats[:, 1] / (history_generosity_stats[:, 0] + 100)
    cohort_stats_all = prior_stats(history_beer, events.time[history_index], events.rating_id[history_index], history_generosity.astype(np.float32))
    history_position = np.searchsorted(history_index, serve_index)
    if np.any(history_position >= len(history_index)) or not np.array_equal(history_index[history_position], serve_index):
        raise RuntimeError("serving rows are absent from target-free cohort history")
    cohort_stats = cohort_stats_all[history_position]
    cohort_mean = cohort_stats[:, 1] / np.maximum(cohort_stats[:, 0], 1)
    cohort_var = np.maximum(cohort_stats[:, 2] / np.maximum(cohort_stats[:, 0], 1) - cohort_mean * cohort_mean, 0)
    cohort = np.column_stack((np.log1p(cohort_stats[:, 0]), cohort_mean, np.sqrt(cohort_var), cohort_stats[:, 0] == 0)).astype(np.float32)
    return np.column_stack([user_history, beer_history, hierarchy] + affinity_blocks + [cohort]).astype(np.float32)


def _frozen_label_features(events: Events, keys: dict[str, np.ndarray], label_n: int, serve_start: int, serve_end: int, residual: np.ndarray, cutoff: int) -> np.ndarray:
    label_index = np.arange(label_n, dtype=np.int64)
    serve_index = np.arange(serve_start, serve_end, dtype=np.int64)
    history_index = np.arange(serve_end, dtype=np.int64)
    return _frozen_label_features_indices(events, keys, label_index, serve_index, history_index, residual, cutoff)


def build_internal_frozen_features(events: Events, cutoff: int, serve_index: np.ndarray, residual: np.ndarray) -> np.ndarray:
    metadata = load_metadata()
    keys = entity_keys(events, metadata)
    label_index = np.flatnonzero((np.arange(len(events.time)) < events.n_train) & (events.time <= cutoff)).astype(np.int64)
    history_index = np.arange(events.n_train, dtype=np.int64)
    return _frozen_label_features_indices(events, keys, label_index, serve_index.astype(np.int64), history_index, residual, cutoff)


def _content_key(events: Events, debug: bool) -> str:
    payload = f"{VERSION}|{debug}|{events.full_train_n}|{events.full_val_n}|{events.full_test_n}|{events.rating_id[0]}|{events.rating_id[-1]}"
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _register_artifact(cache: Path, directory: Path, key: str) -> None:
    import fcntl
    registry = cache / "artifacts.json"
    lock_path = cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(registry.read_text()) if registry.exists() else []
            name = f"ratebeer causal feature blocks lane0 {key}"
            if not any(item.get("name") == name for item in data):
                data.append({"name": name, "path": str(directory.relative_to(cache)), "description": "Strict causal relational base and A/B label feature blocks", "content_key": key, "rebuild_hint": "Run main.py with the matching debug fidelity and feature_factory VERSION"})
                registry.write_text(json.dumps(data, indent=2))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def build_or_load_features(events: Events, debug: bool, log) -> FeatureBlocks:
    cache = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    key = _content_key(events, debug)
    directory = cache / f"lane0_features_{key}"
    paths = {name: directory / f"{name}.npy" for name in ["base", "strict_tv", "frozen_val", "frozen_test", "residual_tv"]}
    names_path = directory / "feature_names.json"
    if all(path.exists() for path in paths.values()) and names_path.exists():
        log(f"feature cache hit key={key}")
        return FeatureBlocks(base=np.load(paths["base"], mmap_mode="r"), strict_tv=np.load(paths["strict_tv"], mmap_mode="r"), frozen_val=np.load(paths["frozen_val"], mmap_mode="r"), frozen_test=np.load(paths["frozen_test"], mmap_mode="r"), names=json.loads(names_path.read_text()), residual_tv=np.load(paths["residual_tv"], mmap_mode="r"))
    directory.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata()
    keys = entity_keys(events, metadata)
    start = time.time()
    base, base_names = build_base_features(events, metadata, keys)
    log(f"base feature block rows={len(base)} cols={base.shape[1]} seconds={time.time() - start:.1f}")
    start = time.time()
    strict_tv, residual_tv, label_names = _strict_label_features(events, keys)
    log(f"strict label block rows={len(strict_tv)} cols={strict_tv.shape[1]} seconds={time.time() - start:.1f}")
    start = time.time()
    frozen_val = _frozen_label_features(events, keys, events.n_train, events.n_train, events.n_tv, residual_tv, V_TIME)
    frozen_test = _frozen_label_features(events, keys, events.n_tv, events.n_tv, len(events.time), residual_tv, T_TIME)
    log(f"frozen A/B blocks val={len(frozen_val)} test={len(frozen_test)} seconds={time.time() - start:.1f}")
    np.save(paths["base"], base)
    np.save(paths["strict_tv"], strict_tv)
    np.save(paths["frozen_val"], frozen_val)
    np.save(paths["frozen_test"], frozen_test)
    np.save(paths["residual_tv"], residual_tv)
    names = base_names + label_names
    names_path.write_text(json.dumps(names))
    _register_artifact(cache, directory, key)
    return FeatureBlocks(base=np.load(paths["base"], mmap_mode="r"), strict_tv=np.load(paths["strict_tv"], mmap_mode="r"), frozen_val=np.load(paths["frozen_val"], mmap_mode="r"), frozen_test=np.load(paths["frozen_test"], mmap_mode="r"), names=names, residual_tv=np.load(paths["residual_tv"], mmap_mode="r"))
