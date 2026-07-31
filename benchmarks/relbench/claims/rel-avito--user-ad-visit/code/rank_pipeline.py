from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
import zlib
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


warnings.filterwarnings("ignore")


N_ADS = 5_960_558
N_USERS = 98_250
DAY_NS = 86_400_000_000_000
HOUR_NS = 3_600_000_000_000
SEED = 2026
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def elapsed(start: float, phase: str) -> None:
    print(f"[lane0] phase={phase} elapsed_seconds={time.time() - start:.1f}", flush=True)


def datetime_ns(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)


def integer_array(series: pd.Series, dtype=np.int32) -> np.ndarray:
    return series.to_numpy(dtype=dtype, na_value=-1)


def float_array(series: pd.Series, dtype=np.float32) -> np.ndarray:
    return series.to_numpy(dtype=dtype, na_value=np.nan)


def token_mask(value: object) -> np.uint64:
    if not isinstance(value, str) or not value:
        return np.uint64(0)
    result = 0
    for token in TOKEN_RE.findall(value.lower())[:24]:
        result |= 1 << (zlib.crc32(token.encode("utf-8")) & 63)
    return np.uint64(result)


@dataclass
class PairAggregate:
    keys: np.ndarray
    users: np.ndarray
    ads: np.ndarray
    values: dict[str, np.ndarray]
    top_ads: np.ndarray
    top_scores: np.ndarray
    top_offsets: np.ndarray


@dataclass
class CandidateSet:
    seed_users: np.ndarray
    offsets: np.ndarray
    ads: np.ndarray
    retrieval: np.ndarray


@dataclass
class OriginData:
    timestamp: pd.Timestamp
    candidates: CandidateSet
    features: np.ndarray
    feature_names: list[str]
    train_indices: np.ndarray
    train_labels: np.ndarray
    train_groups: np.ndarray
    truths: list[np.ndarray] | None


@dataclass
class StaticData:
    n_ads: int
    n_users: int
    ad_cat: np.ndarray
    ad_loc: np.ndarray
    ad_price: np.ndarray
    ad_context: np.ndarray
    ad_title: np.ndarray
    cat_parent: np.ndarray
    loc_region: np.ndarray
    loc_city: np.ndarray
    ad_parent: np.ndarray
    ad_region: np.ndarray
    ad_city: np.ndarray
    user_agent: np.ndarray
    user_os: np.ndarray
    user_device: np.ndarray
    user_family: np.ndarray
    visits_u: np.ndarray
    visits_a: np.ndarray
    visits_ip: np.ndarray
    visits_t: np.ndarray
    phones_u: np.ndarray
    phones_a: np.ndarray
    phones_ip: np.ndarray
    phones_t: np.ndarray
    exposure_u: np.ndarray
    exposure_a: np.ndarray
    exposure_t: np.ndarray
    exposure_position: np.ndarray
    exposure_object: np.ndarray
    exposure_ctr: np.ndarray
    exposure_click: np.ndarray
    searches_u: np.ndarray
    searches_t: np.ndarray
    searches_ip: np.ndarray
    searches_cat: np.ndarray
    searches_loc: np.ndarray
    searches_query: np.ndarray
    transition_src: np.ndarray
    transition_dst: np.ndarray
    transition_t: np.ndarray
    transition_type: np.ndarray
    title_masks: np.ndarray
    title_ready: np.ndarray


@dataclass
class OriginState:
    cutoff_ns: int
    visit: PairAggregate
    search: PairAggregate
    phone: PairAggregate
    ad_features: dict[str, np.ndarray]
    user_features: dict[str, np.ndarray]
    preferred: dict[str, np.ndarray]
    query_mask: np.ndarray
    pop_cat_city: dict[int, np.ndarray]
    pop_cat_region: dict[int, np.ndarray]
    pop_parent_loc: dict[int, np.ndarray]
    pop_family: dict[int, np.ndarray]
    pop_global: np.ndarray
    family_pair_keys: np.ndarray
    family_pair_counts: np.ndarray
    neighbor_ads: np.ndarray
    neighbor_scores: np.ndarray
    neighbor_types: np.ndarray
    neighbor_offsets: np.ndarray
    user_ip: np.ndarray
    ip_lookup: dict[int, np.ndarray]


def make_user_top(
    users: np.ndarray,
    ads: np.ndarray,
    scores: np.ndarray,
    n_users: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(users) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.zeros(n_users + 1, dtype=np.int64),
        )
    order = np.lexsort((-scores, users))
    sorted_users = users[order].astype(np.int32, copy=False)
    sorted_ads = ads[order].astype(np.int32, copy=False)
    sorted_scores = scores[order].astype(np.float32, copy=False)
    counts = np.bincount(sorted_users, minlength=n_users)
    offsets = np.empty(n_users + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return sorted_ads, sorted_scores, offsets


def rank_within_user(
    users: np.ndarray,
    ads: np.ndarray,
    primary: np.ndarray,
) -> np.ndarray:
    order = np.lexsort((ads, primary, users))
    sorted_users = users[order]
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ranks_sorted = np.empty(len(users), dtype=np.float32)
    ends = np.r_[starts[1:], len(users)]
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = np.arange(1, end - start + 1)
    ranks = np.empty(len(users), dtype=np.float32)
    ranks[order] = ranks_sorted
    return ranks


def dense_rank_within_user(
    users: np.ndarray,
    ads: np.ndarray,
    primary: np.ndarray,
) -> np.ndarray:
    order = np.lexsort((ads, primary, users))
    sorted_users = users[order]
    sorted_primary = primary[order]
    ranks_sorted = np.ones(len(users), dtype=np.float32)
    starts = np.r_[0, np.flatnonzero(sorted_users[1:] != sorted_users[:-1]) + 1]
    ends = np.r_[starts[1:], len(users)]
    for start, end in zip(starts, ends):
        if end - start > 1:
            ranks_sorted[start + 1 : end] = 1 + np.cumsum(
                sorted_primary[start + 1 : end]
                != sorted_primary[start : end - 1]
            )
    ranks = np.empty(len(users), dtype=np.float32)
    ranks[order] = ranks_sorted
    return ranks


def aggregate_visits(
    users: np.ndarray,
    ads: np.ndarray,
    times: np.ndarray,
    cutoff_ns: int,
    n_ads: int,
    n_users: int,
) -> PairAggregate:
    age = ((cutoff_ns - times) / HOUR_NS).astype(np.float32)
    frame = pd.DataFrame(
        {
            "key": users.astype(np.int64) * n_ads + ads.astype(np.int64),
            "day": times // DAY_NS,
            "age": age,
            "w6": age <= 6,
            "w24": age <= 24,
            "w72": age <= 72,
            "w168": age <= 168,
        }
    )
    grouped = frame.groupby("key", sort=True, observed=True)
    result = grouped.agg(
        count=("key", "size"),
        days=("day", "nunique"),
        recency=("age", "min"),
        first_age=("age", "max"),
        count_6h=("w6", "sum"),
        count_1d=("w24", "sum"),
        count_3d=("w72", "sum"),
        count_7d=("w168", "sum"),
    ).reset_index()
    keys = result.pop("key").to_numpy(np.int64, copy=False)
    pair_users = (keys // n_ads).astype(np.int32)
    pair_ads = (keys % n_ads).astype(np.int32)
    values = {name: result[name].to_numpy(np.float32, copy=False) for name in result.columns}
    count = values["count"]
    interval = np.divide(
        values["first_age"] - values["recency"],
        np.maximum(count - 1, 1),
        dtype=np.float32,
    )
    values["mean_interval"] = interval
    values["recency_rank"] = rank_within_user(
        pair_users,
        pair_ads,
        values["recency"],
    )
    values["frequency_rank"] = rank_within_user(
        pair_users,
        pair_ads,
        -values["count"],
    )
    values["distinct_days_rank"] = rank_within_user(
        pair_users,
        pair_ads,
        -values["days"],
    )
    values["count_1d_rank"] = dense_rank_within_user(
        pair_users,
        pair_ads,
        -values["count_1d"],
    )
    values["count_3d_rank"] = dense_rank_within_user(
        pair_users,
        pair_ads,
        -values["count_3d"],
    )
    values["count_7d_rank"] = dense_rank_within_user(
        pair_users,
        pair_ads,
        -values["count_7d"],
    )
    interval_for_rank = np.where(count > 1, interval, 100_000.0)
    values["recurrence_interval_rank"] = rank_within_user(
        pair_users,
        pair_ads,
        interval_for_rank,
    )
    score = (
        5.0 * np.exp(-values["recency"] / 6.0)
        + 2.5 * np.exp(-values["recency"] / 24.0)
        + 1.2 * np.exp(-values["recency"] / 72.0)
        + 1.3 * np.log1p(count)
        + 0.25 * values["days"]
        + 0.15 * np.exp(-interval / 72.0)
    ).astype(np.float32)
    top_ads, top_scores, top_offsets = make_user_top(pair_users, pair_ads, score, n_users)
    return PairAggregate(keys, pair_users, pair_ads, values, top_ads, top_scores, top_offsets)


def aggregate_search(
    users: np.ndarray,
    ads: np.ndarray,
    times: np.ndarray,
    position: np.ndarray,
    object_type: np.ndarray,
    ctr: np.ndarray,
    click: np.ndarray,
    cutoff_ns: int,
    n_ads: int,
    n_users: int,
) -> PairAggregate:
    age = ((cutoff_ns - times) / HOUR_NS).astype(np.float32)
    missing = np.isnan(click)
    clicked = click == 1
    nonclicked = click == 0
    frame = pd.DataFrame(
        {
            "key": users.astype(np.int64) * n_ads + ads.astype(np.int64),
            "age": age,
            "w6": age <= 6,
            "w24": age <= 24,
            "w72": age <= 72,
            "w168": age <= 168,
            "clicked": clicked,
            "nonclicked": nonclicked,
            "missing": missing,
            "position": np.nan_to_num(position, nan=100.0),
            "ctr": ctr,
            "object": np.nan_to_num(object_type, nan=-1.0),
        }
    )
    grouped = frame.groupby("key", sort=True, observed=True)
    result = grouped.agg(
        count=("key", "size"),
        recency=("age", "min"),
        count_6h=("w6", "sum"),
        count_1d=("w24", "sum"),
        count_3d=("w72", "sum"),
        count_7d=("w168", "sum"),
        clicks=("clicked", "sum"),
        nonclicks=("nonclicked", "sum"),
        missing_clicks=("missing", "sum"),
        min_position=("position", "min"),
        mean_position=("position", "mean"),
        mean_ctr=("ctr", "mean"),
        mean_object=("object", "mean"),
    ).reset_index()
    keys = result.pop("key").to_numpy(np.int64, copy=False)
    pair_users = (keys // n_ads).astype(np.int32)
    pair_ads = (keys % n_ads).astype(np.int32)
    values = {
        name: np.nan_to_num(result[name].to_numpy(np.float32, copy=False), nan=-1.0)
        for name in result.columns
    }
    score = (
        5.0 * values["clicks"]
        + 3.0 * np.exp(-values["recency"] / 6.0)
        + 1.5 * np.exp(-values["recency"] / 24.0)
        + 0.7 * np.log1p(values["count"])
        + 2.0 / (1.0 + values["min_position"])
        + np.maximum(values["mean_ctr"], 0.0)
        - 0.03 * values["nonclicks"]
    ).astype(np.float32)
    top_ads, top_scores, top_offsets = make_user_top(pair_users, pair_ads, score, n_users)
    return PairAggregate(keys, pair_users, pair_ads, values, top_ads, top_scores, top_offsets)


def aggregate_phones(
    users: np.ndarray,
    ads: np.ndarray,
    times: np.ndarray,
    cutoff_ns: int,
    n_ads: int,
    n_users: int,
) -> PairAggregate:
    age = ((cutoff_ns - times) / HOUR_NS).astype(np.float32)
    frame = pd.DataFrame(
        {
            "key": users.astype(np.int64) * n_ads + ads.astype(np.int64),
            "age": age,
            "w6": age <= 6,
            "w24": age <= 24,
            "w72": age <= 72,
            "w168": age <= 168,
        }
    )
    grouped = frame.groupby("key", sort=True, observed=True)
    result = grouped.agg(
        count=("key", "size"),
        recency=("age", "min"),
        count_6h=("w6", "sum"),
        count_1d=("w24", "sum"),
        count_3d=("w72", "sum"),
        count_7d=("w168", "sum"),
    ).reset_index()
    keys = result.pop("key").to_numpy(np.int64, copy=False)
    pair_users = (keys // n_ads).astype(np.int32)
    pair_ads = (keys % n_ads).astype(np.int32)
    values = {name: result[name].to_numpy(np.float32, copy=False) for name in result.columns}
    score = (
        6.0 * np.exp(-values["recency"] / 6.0)
        + 3.0 * np.exp(-values["recency"] / 24.0)
        + 1.5 * np.exp(-values["recency"] / 72.0)
        + 1.5 * np.log1p(values["count"])
    ).astype(np.float32)
    top_ads, top_scores, top_offsets = make_user_top(pair_users, pair_ads, score, n_users)
    return PairAggregate(keys, pair_users, pair_ads, values, top_ads, top_scores, top_offsets)


def top_lookup(
    keys: np.ndarray,
    ads: np.ndarray,
    scores: np.ndarray,
    quota: int,
) -> dict[int, np.ndarray]:
    if len(keys) == 0:
        return {}
    valid = keys >= 0
    keys = keys[valid].astype(np.int64, copy=False)
    ads = ads[valid].astype(np.int32, copy=False)
    scores = scores[valid].astype(np.float32, copy=False)
    order = np.lexsort((-scores, keys))
    sorted_keys = keys[order]
    sorted_ads = ads[order]
    starts = np.r_[0, np.flatnonzero(sorted_keys[1:] != sorted_keys[:-1]) + 1]
    ends = np.r_[starts[1:], len(sorted_keys)]
    result: dict[int, np.ndarray] = {}
    for start, end in zip(starts, ends):
        result[int(sorted_keys[start])] = sorted_ads[start : min(end, start + quota)].copy()
    return result


def top_preference(
    users: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    n_users: int,
    width: int = 3,
) -> np.ndarray:
    output = np.full((n_users, width), -1, dtype=np.int32)
    valid = (users >= 0) & (values >= 0)
    if not np.any(valid):
        return output
    frame = pd.DataFrame(
        {
            "user": users[valid].astype(np.int32),
            "value": values[valid].astype(np.int32),
            "weight": weights[valid].astype(np.float32),
        }
    )
    agg = frame.groupby(["user", "value"], sort=False, observed=True)["weight"].sum().reset_index()
    order = np.lexsort(
        (
            agg["value"].to_numpy(np.int32),
            -agg["weight"].to_numpy(np.float32),
            agg["user"].to_numpy(np.int32),
        )
    )
    su = agg["user"].to_numpy(np.int32)[order]
    sv = agg["value"].to_numpy(np.int32)[order]
    rank = np.zeros(len(su), dtype=np.int16)
    if len(su):
        starts = np.r_[0, np.flatnonzero(su[1:] != su[:-1]) + 1]
        for start, end in zip(starts, np.r_[starts[1:], len(su)]):
            take = min(width, end - start)
            output[su[start], :take] = sv[start : start + take]
    return output


def make_transitions(
    src_u: np.ndarray,
    src_a: np.ndarray,
    src_t: np.ndarray,
    dst_u: np.ndarray,
    dst_a: np.ndarray,
    dst_t: np.ndarray,
    transition_type: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_u = np.concatenate([src_u.astype(np.int32), dst_u.astype(np.int32)])
    all_a = np.concatenate([src_a.astype(np.int32), dst_a.astype(np.int32)])
    all_t = np.concatenate([src_t.astype(np.int64), dst_t.astype(np.int64)])
    all_kind = np.concatenate(
        [np.zeros(len(src_u), dtype=np.int8), np.ones(len(dst_u), dtype=np.int8)]
    )
    order = np.lexsort((all_kind, all_t, all_u))
    u = all_u[order]
    a = all_a[order]
    t = all_t[order]
    kind = all_kind[order]
    valid = (
        (kind[:-1] == 0)
        & (kind[1:] == 1)
        & (u[:-1] == u[1:])
        & (t[1:] > t[:-1])
        & ((t[1:] - t[:-1]) <= 30 * 60 * 1_000_000_000)
        & (a[:-1] != a[1:])
    )
    idx = np.flatnonzero(valid)
    return (
        a[idx].astype(np.int32),
        a[idx + 1].astype(np.int32),
        t[idx + 1].astype(np.int64),
        np.full(len(idx), transition_type, dtype=np.int8),
    )


def load_static(ctx, start: float) -> StaticData:
    db = ctx.db.table_dict
    ads = db["AdsInfo"].df
    categories = db["Category"].df
    locations = db["Location"].df
    users = db["UserInfo"].df
    visits = db["VisitStream"].df
    phones = db["PhoneRequestsStream"].df
    search_info = db["SearchInfo"].df
    search_stream = db["SearchStream"].df
    n_ads = len(ads)
    n_users = len(users)
    if n_ads != N_ADS or n_users != N_USERS:
        raise RuntimeError(f"unexpected entity cardinality ads={n_ads} users={n_users}")
    ad_cat = integer_array(ads["CategoryID"])
    ad_loc = integer_array(ads["LocationID"])
    ad_price = np.nan_to_num(float_array(ads["Price"]), nan=-1.0)
    ad_context = np.nan_to_num(float_array(ads["IsContext"]), nan=-1.0)
    ad_title = ads["Title"].to_numpy(copy=False)
    cat_parent_map = np.full(max(int(categories["CategoryID"].max()) + 1, 1), -1, dtype=np.int32)
    category_ids = integer_array(categories["CategoryID"])
    cat_parent_map[category_ids] = integer_array(categories["ParentCategoryID"])
    loc_region_map = np.full(max(int(locations["LocationID"].max()) + 1, 1), -1, dtype=np.int32)
    loc_city_map = np.full_like(loc_region_map, -1)
    loc_ids = integer_array(locations["LocationID"])
    loc_region_map[loc_ids] = np.nan_to_num(
        locations["RegionID"].to_numpy(np.float64), nan=-1
    ).astype(np.int32)
    loc_city_map[loc_ids] = np.nan_to_num(
        locations["CityID"].to_numpy(np.float64), nan=-1
    ).astype(np.int32)
    ad_parent = np.full(n_ads, -1, dtype=np.int32)
    ad_region = np.full(n_ads, -1, dtype=np.int32)
    ad_city = np.full(n_ads, -1, dtype=np.int32)
    valid_cat = ad_cat >= 0
    valid_loc = ad_loc >= 0
    ad_parent[valid_cat] = cat_parent_map[ad_cat[valid_cat]]
    ad_region[valid_loc] = loc_region_map[ad_loc[valid_loc]]
    ad_city[valid_loc] = loc_city_map[ad_loc[valid_loc]]
    user_agent = np.nan_to_num(float_array(users["UserAgentID"], np.float64), nan=-1).astype(np.int32)
    user_os = np.nan_to_num(float_array(users["UserAgentOSID"], np.float64), nan=-1).astype(np.int32)
    user_device = np.nan_to_num(float_array(users["UserDeviceID"], np.float64), nan=-1).astype(np.int32)
    user_family = np.nan_to_num(
        float_array(users["UserAgentFamilyID"], np.float64), nan=-1
    ).astype(np.int32)
    visits_u = integer_array(visits["UserID"])
    visits_a = integer_array(visits["AdID"])
    visits_ip = np.nan_to_num(visits["IPID"].to_numpy(np.float64), nan=-1).astype(np.int64)
    visits_t = datetime_ns(visits["ViewDate"])
    valid_visits = (
        (visits_u >= 0)
        & (visits_u < n_users)
        & (visits_a >= 0)
        & (visits_a < n_ads)
    )
    visits_u = visits_u[valid_visits]
    visits_a = visits_a[valid_visits]
    visits_ip = visits_ip[valid_visits]
    visits_t = visits_t[valid_visits]
    phones_u = integer_array(phones["UserID"])
    phones_a = integer_array(phones["AdID"])
    phones_ip = np.nan_to_num(phones["IPID"].to_numpy(np.float64), nan=-1).astype(np.int64)
    phones_t = datetime_ns(phones["PhoneRequestDate"])
    valid_phones = (
        (phones_u >= 0)
        & (phones_u < n_users)
        & (phones_a >= 0)
        & (phones_a < n_ads)
    )
    phones_u = phones_u[valid_phones]
    phones_a = phones_a[valid_phones]
    phones_ip = phones_ip[valid_phones]
    phones_t = phones_t[valid_phones]
    searches_u_float = float_array(search_info["UserID"], np.float64)
    searches_valid = np.isfinite(searches_u_float)
    search_id = integer_array(search_info["SearchID"], np.int64)
    max_search_id = int(search_id.max())
    search_user_map = np.full(max_search_id + 1, -1, dtype=np.int32)
    search_user_map[search_id[searches_valid]] = searches_u_float[searches_valid].astype(np.int32)
    stream_search_id = integer_array(search_stream["SearchID"], np.int64)
    exposure_u_all = search_user_map[stream_search_id]
    stream_ads = integer_array(search_stream["AdID"])
    exposure_valid = (
        (exposure_u_all >= 0)
        & (exposure_u_all < n_users)
        & (stream_ads >= 0)
        & (stream_ads < n_ads)
    )
    exposure_u = exposure_u_all[exposure_valid]
    exposure_a = stream_ads[exposure_valid]
    exposure_t = datetime_ns(search_stream["SearchDate"])[exposure_valid]
    exposure_position = search_stream["Position"].to_numpy(np.float32, copy=False)[exposure_valid]
    exposure_object = search_stream["ObjectType"].to_numpy(np.float32, copy=False)[exposure_valid]
    exposure_ctr = search_stream["HistCTR"].to_numpy(np.float32, copy=False)[exposure_valid]
    exposure_click = search_stream["IsClick"].to_numpy(np.float32, copy=False)[exposure_valid]
    searches_u = searches_u_float[searches_valid].astype(np.int32)
    searches_t = datetime_ns(search_info["SearchDate"])[searches_valid]
    searches_ip = np.nan_to_num(
        search_info["IPID"].to_numpy(np.float64, copy=False)[searches_valid], nan=-1
    ).astype(np.int64)
    searches_cat = integer_array(search_info["CategoryID"])[searches_valid]
    searches_loc = integer_array(search_info["LocationID"])[searches_valid]
    searches_query = search_info["SearchQuery"].to_numpy(copy=False)[searches_valid]
    order = np.lexsort((visits_t, visits_u))
    vu = visits_u[order]
    va = visits_a[order]
    vt = visits_t[order]
    visit_valid = (
        (vu[:-1] == vu[1:])
        & (vt[1:] > vt[:-1])
        & ((vt[1:] - vt[:-1]) <= 30 * 60 * 1_000_000_000)
        & (va[:-1] != va[1:])
    )
    idx = np.flatnonzero(visit_valid)
    tr_parts = [
        (
            va[idx].astype(np.int32),
            va[idx + 1].astype(np.int32),
            vt[idx + 1].astype(np.int64),
            np.ones(len(idx), dtype=np.int8),
        )
    ]
    search_transition_mask = (exposure_click == 1) | (
        (np.nan_to_num(exposure_position, nan=100.0) <= 3) & np.isnan(exposure_click)
    )
    tr_parts.append(
        make_transitions(
            exposure_u[search_transition_mask],
            exposure_a[search_transition_mask],
            exposure_t[search_transition_mask],
            visits_u,
            visits_a,
            visits_t,
            2,
        )
    )
    tr_parts.append(
        make_transitions(phones_u, phones_a, phones_t, visits_u, visits_a, visits_t, 3)
    )
    transition_src = np.concatenate([x[0] for x in tr_parts])
    transition_dst = np.concatenate([x[1] for x in tr_parts])
    transition_t = np.concatenate([x[2] for x in tr_parts])
    transition_type = np.concatenate([x[3] for x in tr_parts])
    elapsed(start, "load_static_and_transitions")
    return StaticData(
        n_ads,
        n_users,
        ad_cat,
        ad_loc,
        ad_price,
        ad_context,
        ad_title,
        cat_parent_map,
        loc_region_map,
        loc_city_map,
        ad_parent,
        ad_region,
        ad_city,
        user_agent,
        user_os,
        user_device,
        user_family,
        visits_u,
        visits_a,
        visits_ip,
        visits_t,
        phones_u,
        phones_a,
        phones_ip,
        phones_t,
        exposure_u,
        exposure_a,
        exposure_t,
        exposure_position,
        exposure_object,
        exposure_ctr,
        exposure_click,
        searches_u,
        searches_t,
        searches_ip,
        searches_cat,
        searches_loc,
        searches_query,
        transition_src,
        transition_dst,
        transition_t,
        transition_type,
        np.zeros(n_ads, dtype=np.uint64),
        np.zeros(n_ads, dtype=bool),
    )


def dense_count(ads: np.ndarray, mask: np.ndarray, n_ads: int) -> np.ndarray:
    return np.bincount(ads[mask], minlength=n_ads).astype(np.float32)


def map_pair(keys: np.ndarray, aggregate: PairAggregate) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(aggregate.keys, keys)
    valid = positions < len(aggregate.keys)
    if np.any(valid):
        valid_idx = np.flatnonzero(valid)
        valid[valid_idx] = aggregate.keys[positions[valid_idx]] == keys[valid_idx]
    return positions, valid


def build_neighbor_state(
    static: StaticData,
    cutoff_ns: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mask = static.transition_t <= cutoff_ns
    src = static.transition_src[mask]
    dst = static.transition_dst[mask]
    t = static.transition_t[mask]
    typ = static.transition_type[mask]
    if len(src) == 0:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int8),
            np.zeros(static.n_ads + 1, dtype=np.int64),
        )
    age = ((cutoff_ns - t) / HOUR_NS).astype(np.float32)
    frame = pd.DataFrame(
        {
            "key": src.astype(np.int64) * static.n_ads + dst.astype(np.int64),
            "age": age,
            "type": typ,
        }
    )
    result = (
        frame.groupby("key", sort=True, observed=True)
        .agg(count=("key", "size"), recency=("age", "min"), type=("type", "max"))
        .reset_index()
    )
    keys = result["key"].to_numpy(np.int64)
    sources = (keys // static.n_ads).astype(np.int32)
    destinations = (keys % static.n_ads).astype(np.int32)
    counts = result["count"].to_numpy(np.float32)
    recency = result["recency"].to_numpy(np.float32)
    types = result["type"].to_numpy(np.int8)
    score = (
        np.log1p(counts) + 2.0 * np.exp(-recency / 24.0) + (types == 2) + 1.5 * (types == 3)
    ).astype(np.float32)
    order = np.lexsort((-score, sources))
    sources = sources[order]
    destinations = destinations[order]
    score = score[order]
    types = types[order]
    counts_by = np.bincount(sources, minlength=static.n_ads)
    offsets = np.empty(static.n_ads + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts_by, out=offsets[1:])
    return destinations, score, types, offsets


def build_ip_state(
    static: StaticData,
    visit_mask: np.ndarray,
    cutoff_ns: int,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    recent = visit_mask & (static.visits_t >= cutoff_ns - 7 * DAY_NS) & (static.visits_ip >= 0)
    u = static.visits_u[recent]
    a = static.visits_a[recent]
    ip = static.visits_ip[recent]
    age = ((cutoff_ns - static.visits_t[recent]) / HOUR_NS).astype(np.float32)
    user_ip = top_preference(u, ip.astype(np.int64), np.exp(-age / 72.0), static.n_users, 3)
    frame = pd.DataFrame(
        {
            "ip": ip.astype(np.int64),
            "ad": a.astype(np.int32),
            "score": np.exp(-age / 24.0).astype(np.float32),
        }
    )
    agg = frame.groupby(["ip", "ad"], sort=False, observed=True)["score"].sum().reset_index()
    counts = frame.groupby("ip", sort=False, observed=True)["ad"].size()
    shared_ips = counts[counts >= 2].index.to_numpy(np.int64)
    keep = np.isin(agg["ip"].to_numpy(np.int64), shared_ips)
    lookup = top_lookup(
        agg["ip"].to_numpy(np.int64)[keep],
        agg["ad"].to_numpy(np.int32)[keep],
        agg["score"].to_numpy(np.float32)[keep],
        25,
    )
    return user_ip, lookup


def build_origin_state(static: StaticData, timestamp: pd.Timestamp, start: float) -> OriginState:
    cutoff_ns = int(timestamp.value)
    visit_mask = static.visits_t <= cutoff_ns
    search_mask = static.exposure_t <= cutoff_ns
    phone_mask = static.phones_t <= cutoff_ns
    visits = aggregate_visits(
        static.visits_u[visit_mask],
        static.visits_a[visit_mask],
        static.visits_t[visit_mask],
        cutoff_ns,
        static.n_ads,
        static.n_users,
    )
    searches = aggregate_search(
        static.exposure_u[search_mask],
        static.exposure_a[search_mask],
        static.exposure_t[search_mask],
        static.exposure_position[search_mask],
        static.exposure_object[search_mask],
        static.exposure_ctr[search_mask],
        static.exposure_click[search_mask],
        cutoff_ns,
        static.n_ads,
        static.n_users,
    )
    phones = aggregate_phones(
        static.phones_u[phone_mask],
        static.phones_a[phone_mask],
        static.phones_t[phone_mask],
        cutoff_ns,
        static.n_ads,
        static.n_users,
    )
    elapsed(start, f"pair_aggregates_{timestamp.date()}")
    visit_age = (cutoff_ns - static.visits_t) / HOUR_NS
    search_age = (cutoff_ns - static.exposure_t) / HOUR_NS
    phone_age = (cutoff_ns - static.phones_t) / HOUR_NS
    ad_features = {
        "ad_visit_1d": dense_count(
            static.visits_a, visit_mask & (visit_age <= 24), static.n_ads
        ),
        "ad_visit_3d": dense_count(
            static.visits_a, visit_mask & (visit_age <= 72), static.n_ads
        ),
        "ad_visit_7d": dense_count(
            static.visits_a, visit_mask & (visit_age <= 168), static.n_ads
        ),
        "ad_impression_1d": dense_count(
            static.exposure_a, search_mask & (search_age <= 24), static.n_ads
        ),
        "ad_impression_7d": dense_count(
            static.exposure_a, search_mask & (search_age <= 168), static.n_ads
        ),
        "ad_phone_1d": dense_count(
            static.phones_a, phone_mask & (phone_age <= 24), static.n_ads
        ),
        "ad_phone_7d": dense_count(
            static.phones_a, phone_mask & (phone_age <= 168), static.n_ads
        ),
    }
    click_base = search_mask & (static.exposure_click == 1)
    ad_features["ad_click_1d"] = dense_count(
        static.exposure_a, click_base & (search_age <= 24), static.n_ads
    )
    ad_features["ad_click_7d"] = dense_count(
        static.exposure_a, click_base & (search_age <= 168), static.n_ads
    )
    active_last = np.full(static.n_ads, np.float32(9999.0), dtype=np.float32)
    np.minimum.at(
        active_last,
        static.visits_a[visit_mask],
        visit_age[visit_mask].astype(np.float32),
    )
    np.minimum.at(
        active_last,
        static.exposure_a[search_mask],
        search_age[search_mask].astype(np.float32),
    )
    np.minimum.at(
        active_last,
        static.phones_a[phone_mask],
        phone_age[phone_mask].astype(np.float32),
    )
    ad_features["ad_last_activity"] = active_last
    unique_visitors = np.bincount(
        visits.ads[visits.values["count_7d"] > 0], minlength=static.n_ads
    ).astype(np.float32)
    ad_features["ad_visitors_7d"] = unique_visitors
    ad_features["ad_visit_velocity"] = ad_features["ad_visit_1d"] / (
        1.0 + np.maximum(ad_features["ad_visit_7d"] - ad_features["ad_visit_1d"], 0.0) / 6.0
    )
    pop_score = (
        2.0 * np.log1p(ad_features["ad_visit_1d"])
        + np.log1p(ad_features["ad_visit_7d"])
        + 0.7 * np.log1p(ad_features["ad_click_7d"])
        + 0.8 * np.log1p(ad_features["ad_phone_7d"])
        + 0.3 * np.log1p(ad_features["ad_impression_7d"])
        + np.exp(-ad_features["ad_last_activity"] / 24.0)
    ).astype(np.float32)
    active = np.flatnonzero(pop_score > 0).astype(np.int32)
    cat_city_key = static.ad_cat[active].astype(np.int64) * 10_000 + (
        static.ad_city[active].astype(np.int64) + 1
    )
    cat_region_key = static.ad_cat[active].astype(np.int64) * 1_000 + (
        static.ad_region[active].astype(np.int64) + 1
    )
    parent_loc_key = static.ad_parent[active].astype(np.int64) * 10_000 + (
        static.ad_loc[active].astype(np.int64) + 1
    )
    pop_cat_city = top_lookup(cat_city_key, active, pop_score[active], 40)
    pop_cat_region = top_lookup(cat_region_key, active, pop_score[active], 40)
    pop_parent_loc = top_lookup(parent_loc_key, active, pop_score[active], 30)
    global_order = active[np.argsort(-pop_score[active], kind="stable")[:500]]
    recent_pair = visits.values["count_7d"] > 0
    family = static.user_family[visits.users[recent_pair]]
    family_keys = family.astype(np.int64) * static.n_ads + visits.ads[recent_pair].astype(np.int64)
    family_frame = pd.DataFrame(
        {
            "key": family_keys,
            "count": visits.values["count_7d"][recent_pair],
        }
    )
    family_agg = family_frame.groupby("key", sort=True, observed=True)["count"].sum().reset_index()
    family_pair_keys = family_agg["key"].to_numpy(np.int64)
    family_pair_counts = family_agg["count"].to_numpy(np.float32)
    family_ids = family_pair_keys // static.n_ads
    family_ads = (family_pair_keys % static.n_ads).astype(np.int32)
    pop_family = top_lookup(family_ids, family_ads, family_pair_counts, 30)
    user_visit_count = np.bincount(
        visits.users, weights=visits.values["count"], minlength=static.n_users
    ).astype(np.float32)
    user_unique_ads = np.bincount(visits.users, minlength=static.n_users).astype(np.float32)
    user_repeat_ads = np.bincount(
        visits.users,
        weights=(visits.values["count"] > 1).astype(np.float32),
        minlength=static.n_users,
    ).astype(np.float32)
    user_search_count = np.bincount(
        searches.users, weights=searches.values["count"], minlength=static.n_users
    ).astype(np.float32)
    user_phone_count = np.bincount(
        phones.users, weights=phones.values["count"], minlength=static.n_users
    ).astype(np.float32)
    user_features = {
        "user_visit_count": user_visit_count,
        "user_unique_ads": user_unique_ads,
        "user_repeat_propensity": user_repeat_ads / np.maximum(user_unique_ads, 1.0),
        "user_search_count": user_search_count,
        "user_phone_count": user_phone_count,
    }
    recent_visit_pair = visits.values["count_7d"] > 0
    pref_users = visits.users[recent_visit_pair]
    pref_ads = visits.ads[recent_visit_pair]
    pref_weights = (
        visits.values["count_7d"][recent_visit_pair]
        + 2.0 * np.exp(-visits.values["recency"][recent_visit_pair] / 72.0)
    )
    raw_search_mask = (static.searches_t <= cutoff_ns) & (
        static.searches_t >= cutoff_ns - 7 * DAY_NS
    )
    search_pref_u = static.searches_u[raw_search_mask]
    search_pref_cat = static.searches_cat[raw_search_mask]
    search_pref_loc = static.searches_loc[raw_search_mask]
    search_pref_age = (
        (cutoff_ns - static.searches_t[raw_search_mask]) / HOUR_NS
    ).astype(np.float32)
    search_pref_weight = np.exp(-search_pref_age / 72.0).astype(np.float32)
    combined_u = np.concatenate([pref_users, search_pref_u])
    combined_cat = np.concatenate([static.ad_cat[pref_ads], search_pref_cat])
    combined_loc = np.concatenate([static.ad_loc[pref_ads], search_pref_loc])
    combined_parent = np.concatenate(
        [
            static.ad_parent[pref_ads],
            static.cat_parent[np.clip(search_pref_cat, 0, len(static.cat_parent) - 1)],
        ]
    )
    combined_city = np.concatenate(
        [
            static.ad_city[pref_ads],
            static.loc_city[np.clip(search_pref_loc, 0, len(static.loc_city) - 1)],
        ]
    )
    combined_region = np.concatenate(
        [
            static.ad_region[pref_ads],
            static.loc_region[np.clip(search_pref_loc, 0, len(static.loc_region) - 1)],
        ]
    )
    combined_weights = np.concatenate([pref_weights, search_pref_weight])
    preferred = {
        "cat": top_preference(combined_u, combined_cat, combined_weights, static.n_users),
        "loc": top_preference(combined_u, combined_loc, combined_weights, static.n_users),
        "parent": top_preference(
            combined_u, combined_parent, combined_weights, static.n_users
        ),
        "city": top_preference(combined_u, combined_city, combined_weights, static.n_users),
        "region": top_preference(
            combined_u, combined_region, combined_weights, static.n_users
        ),
    }
    query_mask = np.zeros(static.n_users, dtype=np.uint64)
    query_rows = np.flatnonzero(raw_search_mask)
    query_values = np.fromiter(
        (token_mask(static.searches_query[i]) for i in query_rows),
        dtype=np.uint64,
        count=len(query_rows),
    )
    np.bitwise_or.at(query_mask, static.searches_u[query_rows], query_values)
    price_mask = recent_visit_pair & (static.ad_price[visits.ads] >= 0)
    price_u = visits.users[price_mask]
    price_v = np.log1p(static.ad_price[visits.ads[price_mask]]).astype(np.float64)
    price_w = visits.values["count_7d"][price_mask].astype(np.float64)
    weight_sum = np.bincount(price_u, weights=price_w, minlength=static.n_users)
    price_sum = np.bincount(price_u, weights=price_v * price_w, minlength=static.n_users)
    price_sq = np.bincount(price_u, weights=price_v * price_v * price_w, minlength=static.n_users)
    price_mean = price_sum / np.maximum(weight_sum, 1.0)
    price_var = np.maximum(price_sq / np.maximum(weight_sum, 1.0) - price_mean**2, 0.0)
    user_features["user_log_price_mean"] = price_mean.astype(np.float32)
    user_features["user_log_price_std"] = np.sqrt(price_var).astype(np.float32)
    neighbor_ads, neighbor_scores, neighbor_types, neighbor_offsets = build_neighbor_state(
        static, cutoff_ns
    )
    user_ip, ip_lookup = build_ip_state(static, visit_mask, cutoff_ns)
    elapsed(start, f"origin_state_{timestamp.date()}")
    return OriginState(
        cutoff_ns,
        visits,
        searches,
        phones,
        ad_features,
        user_features,
        preferred,
        query_mask,
        pop_cat_city,
        pop_cat_region,
        pop_parent_loc,
        pop_family,
        global_order.astype(np.int32),
        family_pair_keys,
        family_pair_counts,
        neighbor_ads,
        neighbor_scores,
        neighbor_types,
        neighbor_offsets,
        user_ip,
        ip_lookup,
    )


def csr_slice(aggregate: PairAggregate, user: int, quota: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(aggregate.top_offsets[user])
    end = min(int(aggregate.top_offsets[user + 1]), start + quota)
    return aggregate.top_ads[start:end], aggregate.top_scores[start:end]


def build_candidates(
    static: StaticData,
    state: OriginState,
    seed_users: np.ndarray,
    cap: int = 400,
) -> CandidateSet:
    max_rows = len(seed_users) * cap
    output_ads = np.empty(max_rows, dtype=np.int32)
    output_ret = np.empty((max_rows, 18), dtype=np.float32)
    offsets = np.empty(len(seed_users) + 1, dtype=np.int64)
    offsets[0] = 0
    cursor = 0
    channel_quotas = (80, 120, 30, 100, 120)
    for row_index, user_value in enumerate(seed_users):
        user = int(user_value)
        channels: list[list[tuple[int, float, int]]] = [[], [], [], [], []]
        ads, scores = csr_slice(state.visit, user, 80)
        channels[0] = [(int(a), float(s), 0) for a, s in zip(ads, scores)]
        ads, scores = csr_slice(state.search, user, 120)
        channels[1] = [(int(a), float(s), 0) for a, s in zip(ads, scores)]
        ads, scores = csr_slice(state.phone, user, 30)
        channels[2] = [(int(a), float(s), 0) for a, s in zip(ads, scores)]
        causal: dict[int, tuple[float, int]] = {}
        source_ads = []
        source_ads.extend(channels[0][:8])
        source_ads.extend(channels[1][:5])
        source_ads.extend(channels[2][:5])
        for source, source_score, _ in source_ads:
            begin = int(state.neighbor_offsets[source])
            end = min(int(state.neighbor_offsets[source + 1]), begin + 50)
            for pos in range(begin, end):
                destination = int(state.neighbor_ads[pos])
                score = float(state.neighbor_scores[pos]) + 0.05 * source_score
                typ = int(state.neighbor_types[pos])
                previous = causal.get(destination)
                if previous is None or score > previous[0]:
                    causal[destination] = (score, typ)
        for ip in state.user_ip[user]:
            if ip < 0:
                continue
            for rank, destination in enumerate(state.ip_lookup.get(int(ip), ())):
                score = 1.0 / (1.0 + rank)
                previous = causal.get(int(destination))
                if previous is None or score > previous[0]:
                    causal[int(destination)] = (score, 4)
        causal_ranked = sorted(causal.items(), key=lambda item: (-item[1][0], item[0]))[:100]
        channels[3] = [(ad, value[0], value[1]) for ad, value in causal_ranked]
        popularity: list[int] = []
        seen_pop: set[int] = set()
        for cat in state.preferred["cat"][user]:
            if cat < 0:
                continue
            for city in state.preferred["city"][user]:
                if city < 0:
                    continue
                key = int(cat) * 10_000 + int(city) + 1
                for ad in state.pop_cat_city.get(key, ()):
                    if int(ad) not in seen_pop:
                        seen_pop.add(int(ad))
                        popularity.append(int(ad))
                if len(popularity) >= 55:
                    break
            if len(popularity) >= 55:
                break
        for cat in state.preferred["cat"][user]:
            if cat < 0:
                continue
            for region in state.preferred["region"][user]:
                if region < 0:
                    continue
                key = int(cat) * 1_000 + int(region) + 1
                for ad in state.pop_cat_region.get(key, ()):
                    if int(ad) not in seen_pop:
                        seen_pop.add(int(ad))
                        popularity.append(int(ad))
                if len(popularity) >= 80:
                    break
            if len(popularity) >= 80:
                break
        for parent in state.preferred["parent"][user]:
            if parent < 0:
                continue
            for location in state.preferred["loc"][user]:
                if location < 0:
                    continue
                key = int(parent) * 10_000 + int(location) + 1
                for ad in state.pop_parent_loc.get(key, ()):
                    if int(ad) not in seen_pop:
                        seen_pop.add(int(ad))
                        popularity.append(int(ad))
                if len(popularity) >= 100:
                    break
            if len(popularity) >= 100:
                break
        for ad in state.pop_family.get(int(static.user_family[user]), ()):
            if int(ad) not in seen_pop:
                seen_pop.add(int(ad))
                popularity.append(int(ad))
            if len(popularity) >= 115:
                break
        for ad in state.pop_global:
            if int(ad) not in seen_pop:
                seen_pop.add(int(ad))
                popularity.append(int(ad))
            if len(popularity) >= 120:
                break
        channels[4] = [
            (ad, 1.0 / (1.0 + rank), 0) for rank, ad in enumerate(popularity[:120])
        ]
        accumulator: dict[int, list[float]] = {}
        for channel, items in enumerate(channels):
            for rank, (ad, score, causal_type) in enumerate(items[: channel_quotas[channel]], 1):
                values = accumulator.get(ad)
                if values is None:
                    values = [0.0] * 18
                    values[5:10] = [999.0] * 5
                    accumulator[ad] = values
                values[channel] = score
                values[5 + channel] = float(rank)
                values[10 + channel] = 1.0
                values[15] += 1.0
                values[16] += 1.0 / (40.0 + rank)
                if causal_type:
                    values[17] = max(values[17], float(causal_type))
        ranked = sorted(
            accumulator.items(),
            key=lambda item: (-item[1][16], -item[1][15], item[0]),
        )[:cap]
        count = len(ranked)
        if count:
            output_ads[cursor : cursor + count] = [item[0] for item in ranked]
            output_ret[cursor : cursor + count] = np.asarray(
                [item[1] for item in ranked], dtype=np.float32
            )
        cursor += count
        offsets[row_index + 1] = cursor
    return CandidateSet(
        seed_users.astype(np.int32, copy=False),
        offsets,
        output_ads[:cursor].copy(),
        output_ret[:cursor].copy(),
    )


def ensure_title_masks(static: StaticData, candidate_ads: np.ndarray) -> None:
    unique_ads = np.unique(candidate_ads)
    pending = unique_ads[~static.title_ready[unique_ads]]
    if len(pending) == 0:
        return
    values = np.fromiter(
        (token_mask(static.ad_title[int(ad)]) for ad in pending),
        dtype=np.uint64,
        count=len(pending),
    )
    static.title_masks[pending] = values
    static.title_ready[pending] = True


def matched_preference(candidate: np.ndarray, users: np.ndarray, pref: np.ndarray) -> np.ndarray:
    return np.any(pref[users] == candidate[:, None], axis=1).astype(np.float32)


def extract_features(
    static: StaticData,
    state: OriginState,
    candidates: CandidateSet,
) -> tuple[np.ndarray, list[str]]:
    lengths = np.diff(candidates.offsets)
    candidate_users = np.repeat(candidates.seed_users, lengths)
    candidate_ads = candidates.ads
    keys = candidate_users.astype(np.int64) * static.n_ads + candidate_ads.astype(np.int64)
    columns: list[np.ndarray] = []
    names: list[str] = []
    for index, name in enumerate(
        [
            "repeat_score",
            "search_score",
            "phone_score",
            "causal_score",
            "popularity_score",
            "repeat_rank",
            "search_rank",
            "phone_rank",
            "causal_rank",
            "popularity_rank",
            "repeat_flag",
            "search_flag",
            "phone_flag",
            "causal_flag",
            "popularity_flag",
            "channel_agreement",
            "rrf_score",
            "causal_type",
        ]
    ):
        columns.append(candidates.retrieval[:, index].astype(np.float32, copy=False))
        names.append(name)
    for prefix, aggregate in [
        ("visit", state.visit),
        ("impression", state.search),
        ("phone", state.phone),
    ]:
        positions, valid = map_pair(keys, aggregate)
        for name, source in aggregate.values.items():
            default = 9999.0 if name in {"recency", "first_age"} else 0.0
            result = np.full(len(keys), default, dtype=np.float32)
            result[valid] = source[positions[valid]]
            columns.append(result)
            names.append(f"ua_{prefix}_{name}")
    for name, source in state.ad_features.items():
        columns.append(source[candidate_ads].astype(np.float32, copy=False))
        names.append(name)
    visit_count = columns[names.index("ua_visit_count")]
    user_total = state.user_features["user_visit_count"][candidate_users]
    columns.append(visit_count / np.maximum(user_total, 1.0))
    names.append("ua_visit_share")
    for name, source in state.user_features.items():
        columns.append(source[candidate_users].astype(np.float32, copy=False))
        names.append(name)
    cat = static.ad_cat[candidate_ads]
    loc = static.ad_loc[candidate_ads]
    parent = static.ad_parent[candidate_ads]
    city = static.ad_city[candidate_ads]
    region = static.ad_region[candidate_ads]
    metadata = [
        ("category", cat),
        ("location", loc),
        ("parent_category", parent),
        ("city", city),
        ("region", region),
        ("is_context", static.ad_context[candidate_ads]),
    ]
    for name, value in metadata:
        columns.append(value.astype(np.float32, copy=False))
        names.append(name)
    columns.extend(
        [
            matched_preference(cat, candidate_users, state.preferred["cat"]),
            matched_preference(loc, candidate_users, state.preferred["loc"]),
            matched_preference(parent, candidate_users, state.preferred["parent"]),
            matched_preference(city, candidate_users, state.preferred["city"]),
            matched_preference(region, candidate_users, state.preferred["region"]),
        ]
    )
    names.extend(
        [
            "intent_category_match",
            "intent_location_match",
            "intent_parent_match",
            "intent_city_match",
            "intent_region_match",
        ]
    )
    log_price = np.log1p(np.maximum(static.ad_price[candidate_ads], 0.0)).astype(np.float32)
    user_price_mean = state.user_features["user_log_price_mean"][candidate_users]
    user_price_std = state.user_features["user_log_price_std"][candidate_users]
    columns.extend(
        [
            log_price,
            np.abs(log_price - user_price_mean),
            (log_price - user_price_mean) / np.maximum(user_price_std, 0.25),
        ]
    )
    names.extend(["log_price", "price_abs_deviation", "price_z_deviation"])
    ensure_title_masks(static, candidate_ads)
    overlap = np.bitwise_count(
        static.title_masks[candidate_ads] & state.query_mask[candidate_users]
    ).astype(np.float32)
    columns.append(overlap)
    names.append("title_query_token_overlap")
    for name, source in [
        ("user_agent", static.user_agent),
        ("user_os", static.user_os),
        ("user_device", static.user_device),
        ("user_family", static.user_family),
    ]:
        columns.append(source[candidate_users].astype(np.float32, copy=False))
        names.append(name)
    family_keys = (
        static.user_family[candidate_users].astype(np.int64) * static.n_ads
        + candidate_ads.astype(np.int64)
    )
    pos = np.searchsorted(state.family_pair_keys, family_keys)
    valid = pos < len(state.family_pair_keys)
    if np.any(valid):
        idx = np.flatnonzero(valid)
        valid[idx] = state.family_pair_keys[pos[idx]] == family_keys[idx]
    family_count = np.zeros(len(keys), dtype=np.float32)
    family_count[valid] = state.family_pair_counts[pos[valid]]
    columns.append(family_count)
    names.append("device_family_ad_prior")
    matrix = np.column_stack(columns).astype(np.float32, copy=False)
    matrix = np.nan_to_num(matrix, nan=-1.0, posinf=9999.0, neginf=-9999.0)
    return matrix, names


def training_subset(
    candidates: CandidateSet,
    truths: list[np.ndarray] | None,
    maximum: int = 160,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if truths is None:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[int] = []
    for row, truth in enumerate(truths):
        start = int(candidates.offsets[row])
        end = int(candidates.offsets[row + 1])
        ads = candidates.ads[start:end]
        hit = np.isin(ads, np.asarray(truth, dtype=np.int32), assume_unique=False)
        if not np.any(hit):
            continue
        positive = np.flatnonzero(hit)
        negative = np.flatnonzero(~hit)
        selected = np.sort(
            np.concatenate([positive, negative[: max(0, maximum - len(positive))]])
        )
        indices.append(start + selected)
        labels.append(hit[selected].astype(np.float32))
        groups.append(len(selected))
    if not indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.concatenate(indices).astype(np.int64),
        np.concatenate(labels).astype(np.float32),
        np.asarray(groups, dtype=np.int32),
    )


def repeat_training_subset(
    origin: OriginData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if origin.truths is None:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    indices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    groups: list[int] = []
    for row, truth in enumerate(origin.truths):
        start = int(origin.candidates.offsets[row])
        end = int(origin.candidates.offsets[row + 1])
        repeat_local = np.flatnonzero(origin.candidates.retrieval[start:end, 10] > 0)
        if len(repeat_local) == 0:
            continue
        ads = origin.candidates.ads[start:end][repeat_local]
        hit = np.isin(ads, np.asarray(truth, dtype=np.int32), assume_unique=False)
        if not np.any(hit):
            continue
        indices.append(start + repeat_local)
        labels.append(hit.astype(np.float32))
        groups.append(len(repeat_local))
    if not indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
        )
    return (
        np.concatenate(indices).astype(np.int64),
        np.concatenate(labels).astype(np.float32),
        np.asarray(groups, dtype=np.int32),
    )


def fallback_predictions(candidates: CandidateSet, global_ads: np.ndarray) -> np.ndarray:
    result = np.empty((len(candidates.seed_users), 12), dtype=np.int64)
    for row in range(len(candidates.seed_users)):
        start = int(candidates.offsets[row])
        end = int(candidates.offsets[row + 1])
        selected: list[int] = []
        seen: set[int] = set()
        for ad in candidates.ads[start:end]:
            value = int(ad)
            if value not in seen:
                seen.add(value)
                selected.append(value)
            if len(selected) >= 12:
                break
        if len(selected) < 12:
            for ad in global_ads:
                value = int(ad)
                if value not in seen:
                    seen.add(value)
                    selected.append(value)
                if len(selected) >= 12:
                    break
        filler = 0
        while len(selected) < 12:
            if filler not in seen:
                seen.add(filler)
                selected.append(filler)
            filler += 1
        result[row] = selected
    return result


def scored_predictions(
    model: lgb.Booster,
    origin: OriginData,
    global_ads: np.ndarray,
) -> np.ndarray:
    scores = model.predict(origin.features, num_iteration=model.best_iteration or model.current_iteration())
    result = np.empty((len(origin.candidates.seed_users), 12), dtype=np.int64)
    for row in range(len(origin.candidates.seed_users)):
        start = int(origin.candidates.offsets[row])
        end = int(origin.candidates.offsets[row + 1])
        local_order = np.argsort(-scores[start:end], kind="stable")
        selected: list[int] = []
        seen: set[int] = set()
        for index in local_order:
            ad = int(origin.candidates.ads[start + index])
            if ad not in seen:
                seen.add(ad)
                selected.append(ad)
            if len(selected) >= 12:
                break
        if len(selected) < 12:
            for ad in global_ads:
                value = int(ad)
                if value not in seen:
                    seen.add(value)
                    selected.append(value)
                if len(selected) >= 12:
                    break
        result[row] = selected
    return result


def hybrid_predictions(
    model: lgb.Booster,
    origin: OriginData,
    global_ads: np.ndarray,
    model_weight: float,
    model_scores: np.ndarray | None = None,
    rerank_nonrepeat: bool = True,
) -> np.ndarray:
    if model_scores is None:
        model_scores = model.predict(
            origin.features, num_iteration=model.best_iteration or model.current_iteration()
        )
    result = np.empty((len(origin.candidates.seed_users), 12), dtype=np.int64)
    for row in range(len(origin.candidates.seed_users)):
        start = int(origin.candidates.offsets[row])
        end = int(origin.candidates.offsets[row + 1])
        retrieval = origin.candidates.retrieval[start:end]
        base_scores = (
            100.0 * retrieval[:, 10] + retrieval[:, 0] + retrieval[:, 16]
        )
        combined = np.empty(end - start, dtype=np.float64)
        repeat_flag = retrieval[:, 10] > 0
        for flag in [False, True]:
            local = np.flatnonzero(repeat_flag == flag)
            if len(local) == 0:
                continue
            base_order = local[np.argsort(-base_scores[local], kind="stable")]
            model_order = local[
                np.argsort(-model_scores[start:end][local], kind="stable")
            ]
            base_rank_full = np.empty(end - start, dtype=np.int32)
            model_rank_full = np.empty(end - start, dtype=np.int32)
            base_rank_full[base_order] = np.arange(1, len(local) + 1)
            model_rank_full[model_order] = np.arange(1, len(local) + 1)
            effective_weight = model_weight if flag or rerank_nonrepeat else 0.0
            combined[local] = (
                (1.0 - effective_weight) / (40.0 + base_rank_full[local])
                + effective_weight / (40.0 + model_rank_full[local])
                + 100.0 * float(flag)
            )
        local_order = np.argsort(-combined, kind="stable")
        selected: list[int] = []
        seen: set[int] = set()
        for index in local_order:
            ad = int(origin.candidates.ads[start + index])
            if ad not in seen:
                seen.add(ad)
                selected.append(ad)
            if len(selected) >= 12:
                break
        if len(selected) < 12:
            for ad in global_ads:
                value = int(ad)
                if value not in seen:
                    seen.add(value)
                    selected.append(value)
                if len(selected) >= 12:
                    break
        result[row] = selected
    return result


def repeat_first_predictions(
    origin: OriginData,
    global_ads: np.ndarray,
) -> np.ndarray:
    class RetrievalModel:
        best_iteration = 1

        @staticmethod
        def current_iteration() -> int:
            return 1

        @staticmethod
        def predict(features, num_iteration=None):
            return np.zeros(len(features), dtype=np.float32)

    return hybrid_predictions(RetrievalModel(), origin, global_ads, 0.0)


def fuse_model_ranks(
    origin: OriginData,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    first_weight: float,
) -> np.ndarray:
    result = np.zeros(len(first_scores), dtype=np.float32)
    for row in range(len(origin.candidates.seed_users)):
        start = int(origin.candidates.offsets[row])
        end = int(origin.candidates.offsets[row + 1])
        repeat_local = np.flatnonzero(
            origin.candidates.retrieval[start:end, 10] > 0
        )
        if len(repeat_local) == 0:
            continue
        first_order = repeat_local[
            np.argsort(-first_scores[start:end][repeat_local], kind="stable")
        ]
        second_order = repeat_local[
            np.argsort(-second_scores[start:end][repeat_local], kind="stable")
        ]
        first_rank = np.empty(end - start, dtype=np.int32)
        second_rank = np.empty(end - start, dtype=np.int32)
        first_rank[first_order] = np.arange(1, len(repeat_local) + 1)
        second_rank[second_order] = np.arange(1, len(repeat_local) + 1)
        result[start:end][repeat_local] = (
            first_weight / (40.0 + first_rank[repeat_local])
            + (1.0 - first_weight) / (40.0 + second_rank[repeat_local])
        )
    return result


def map_at_12(predictions: np.ndarray, truths: list[np.ndarray]) -> float:
    values = np.zeros(len(truths), dtype=np.float64)
    for row, truth in enumerate(truths):
        truth_set = set(map(int, truth))
        hits = np.fromiter(
            (1.0 if int(ad) in truth_set else 0.0 for ad in predictions[row]),
            dtype=np.float64,
            count=12,
        )
        values[row] = np.sum(np.cumsum(hits) / np.arange(1, 13) * hits) / min(
            len(truth_set), 12
        )
    return float(values.mean())


def candidate_diagnostics(
    origin: OriginData,
    history_counts: np.ndarray,
) -> dict[str, object]:
    candidates = origin.candidates
    truths = origin.truths
    if truths is None:
        return {}
    channel_names = ["repeat", "search", "phone", "causal", "popularity"]
    channel_hits = {name: 0 for name in channel_names}
    truth_total = 0
    rows_covered = 0
    union_hits = {50: 0, 100: 0, 200: 0, 400: 0}
    union_oracle = 0.0
    strata = {
        "zero": [0, 0, 0.0],
        "shallow_1_10": [0, 0, 0.0],
        "medium_11_50": [0, 0, 0.0],
        "deep_51_plus": [0, 0, 0.0],
    }
    incremental = {name: 0 for name in channel_names}
    for row, truth in enumerate(truths):
        truth_set = set(map(int, truth))
        truth_total += len(truth_set)
        start = int(candidates.offsets[row])
        end = int(candidates.offsets[row + 1])
        ads = candidates.ads[start:end]
        ret = candidates.retrieval[start:end]
        hit_any = np.isin(ads, np.fromiter(truth_set, dtype=np.int32))
        rows_covered += int(np.any(hit_any))
        for k in union_hits:
            union_hits[k] += int(np.sum(hit_any[:k]))
        oracle = min(int(np.sum(hit_any)), 12) / min(len(truth_set), 12)
        union_oracle += oracle
        cumulative: set[int] = set()
        for channel, name in enumerate(channel_names):
            channel_ads = set(map(int, ads[ret[:, 10 + channel] > 0]))
            hits = truth_set & channel_ads
            channel_hits[name] += len(hits)
            incremental[name] += len(hits - cumulative)
            cumulative |= hits
        depth = history_counts[int(candidates.seed_users[row])]
        if depth == 0:
            key = "zero"
        elif depth <= 10:
            key = "shallow_1_10"
        elif depth <= 50:
            key = "medium_11_50"
        else:
            key = "deep_51_plus"
        strata[key][0] += 1
        strata[key][1] += int(np.sum(hit_any))
        strata[key][2] += oracle
    return {
        "timestamp": str(origin.timestamp),
        "rows": len(truths),
        "truth_items": truth_total,
        "row_coverage": rows_covered / len(truths),
        "recall": {str(k): value / truth_total for k, value in union_hits.items()},
        "channel_recall": {name: value / truth_total for name, value in channel_hits.items()},
        "incremental_recall": {name: value / truth_total for name, value in incremental.items()},
        "oracle_map12": union_oracle / len(truths),
        "strata": {
            key: {
                "count": value[0],
                "retrieved_items": value[1],
                "oracle_map12": value[2] / max(value[0], 1),
            }
            for key, value in strata.items()
        },
    }


def lgb_parameters(objective: str = "lambdarank") -> dict[str, object]:
    parameters: dict[str, object] = {
        "objective": objective,
        "learning_rate": 0.035,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 6.0,
        "max_bin": 127,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "verbosity": -1,
    }
    if objective == "lambdarank":
        parameters.update(
            {
                "metric": "ndcg",
                "ndcg_eval_at": [12],
                "lambdarank_truncation_level": 12,
                "label_gain": [0, 1],
            }
        )
    else:
        parameters["metric"] = "binary_logloss"
    return parameters


def train_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    rounds: int,
    x_valid: np.ndarray | None = None,
    y_valid: np.ndarray | None = None,
    valid_groups: np.ndarray | None = None,
    objective: str = "lambdarank",
) -> lgb.Booster:
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        group=groups if objective == "lambdarank" else None,
        free_raw_data=False,
    )
    valid_sets = None
    callbacks = [lgb.log_evaluation(0)]
    if x_valid is not None and y_valid is not None:
        valid_set = lgb.Dataset(
            x_valid,
            label=y_valid,
            group=valid_groups if objective == "lambdarank" else None,
            reference=train_set,
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        callbacks.append(lgb.early_stopping(80, verbose=False))
    return lgb.train(
        lgb_parameters(objective),
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )


def origin_from_table(
    static: StaticData,
    table: pd.DataFrame,
    timestamp: pd.Timestamp,
    start: float,
    with_truth: bool,
) -> tuple[OriginData, OriginState]:
    rows = table["timestamp"] == timestamp
    frame = table.loc[rows]
    seed_users = frame["UserID"].to_numpy(np.int32, copy=False)
    truths = (
        [np.asarray(value, dtype=np.int32) for value in frame["AdID"].to_numpy()]
        if with_truth
        else None
    )
    state = build_origin_state(static, timestamp, start)
    candidates = build_candidates(static, state, seed_users)
    elapsed(start, f"candidates_{timestamp.date()}_{len(candidates.ads)}")
    features, feature_names = extract_features(static, state, candidates)
    train_indices, train_labels, train_groups = training_subset(candidates, truths)
    elapsed(start, f"features_{timestamp.date()}_{features.shape}")
    return (
        OriginData(
            timestamp,
            candidates,
            features,
            feature_names,
            train_indices,
            train_labels,
            train_groups,
            truths,
        ),
        state,
    )


def validate_predictions(
    val_predictions: np.ndarray,
    test_predictions: np.ndarray,
    n_val: int,
    n_test: int,
    n_ads: int,
) -> None:
    for name, values, expected in [
        ("val", val_predictions, (n_val, 12)),
        ("test", test_predictions, (n_test, 12)),
    ]:
        if values.shape != expected:
            raise RuntimeError(f"{name} shape {values.shape} != {expected}")
        if values.dtype != np.int64:
            raise RuntimeError(f"{name} dtype {values.dtype} != int64")
        if np.any(values < 0) or np.any(values >= n_ads):
            raise RuntimeError(f"{name} has out-of-range destination IDs")
        if np.any(np.apply_along_axis(lambda row: len(np.unique(row)) != 12, 1, values)):
            raise RuntimeError(f"{name} has duplicate destination IDs")


def debug_predictions(ctx, static: StaticData, start: float) -> tuple[np.ndarray, np.ndarray]:
    outputs = []
    for frame in [ctx.val.df, ctx.test.df]:
        timestamp = pd.Timestamp(frame["timestamp"].iloc[0])
        cutoff_ns = int(timestamp.value)
        mask = static.visits_t <= cutoff_ns
        aggregate = aggregate_visits(
            static.visits_u[mask],
            static.visits_a[mask],
            static.visits_t[mask],
            cutoff_ns,
            static.n_ads,
            static.n_users,
        )
        global_counts = np.bincount(static.visits_a[mask], minlength=static.n_ads)
        global_ads = np.argsort(-global_counts, kind="stable")[:1000].astype(np.int32)
        result = np.empty((len(frame), 12), dtype=np.int64)
        for row, user_value in enumerate(frame["UserID"].to_numpy(np.int32)):
            user = int(user_value)
            ads, _ = csr_slice(aggregate, user, 80)
            chosen: list[int] = []
            seen: set[int] = set()
            for ad in np.concatenate([ads, global_ads]):
                value = int(ad)
                if value not in seen:
                    seen.add(value)
                    chosen.append(value)
                if len(chosen) == 12:
                    break
            result[row] = chosen
        outputs.append(result)
        elapsed(start, f"debug_{timestamp.date()}")
    return outputs[0], outputs[1]


def append_artifact(shared: Path, record: dict[str, str]) -> None:
    path = shared / "artifacts.json"
    if path.exists():
        try:
            values = json.loads(path.read_text())
        except Exception:
            values = []
    else:
        values = []
    if not any(value.get("content_key") == record["content_key"] for value in values):
        values.append(record)
        temporary = path.with_suffix(".lane0.tmp")
        temporary.write_text(json.dumps(values, indent=2))
        os.replace(temporary, path)
