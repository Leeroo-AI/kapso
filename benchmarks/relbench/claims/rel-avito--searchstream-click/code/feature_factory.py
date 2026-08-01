from __future__ import annotations

import gc
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kapso_datasets.common import load_task, shared_cache_dir


V_TIMESTAMP = pd.Timestamp("2015-05-08")
T_TIMESTAMP = pd.Timestamp("2015-05-14")
ORIGIN_TIMESTAMP = pd.Timestamp("2015-04-25")


@dataclass
class FeatureBundle:
    context: object
    spine: pd.DataFrame
    labels_train: np.ndarray
    labels_val: np.ndarray
    n_train: int
    n_val: int
    n_test: int
    tables: dict[str, pd.DataFrame]


LABEL_GRAINS = [
    ("ad", ["AdID"], None),
    ("user", ["UserID"], None),
    ("ip", ["IPID"], None),
    ("position", ["Position"], None),
    ("object", ["ObjectType"], None),
    ("ad_category", ["AdCategoryID"], None),
    ("search_category", ["SearchCategoryID"], None),
    ("ad_city", ["AdCityID"], None),
    ("search_city", ["SearchCityID"], None),
    ("ad_region", ["AdRegionID"], None),
    ("search_region", ["SearchRegionID"], None),
    ("user_ad", ["UserID", "AdID"], ("user", "ad")),
    ("ip_ad", ["IPID", "AdID"], ("ip", "ad")),
    ("user_category", ["UserID", "AdCategoryID"], ("user", "ad_category")),
    ("ad_position", ["AdID", "Position"], ("ad", "position")),
    ("category_position", ["AdCategoryID", "Position"], ("ad_category", "position")),
    ("category_object", ["AdCategoryID", "ObjectType"], ("ad_category", "object")),
]

def phase(message: str, started: float) -> None:
    print(f"[candidate] {message}: {time.time() - started:.1f}s", flush=True)


def assert_order(frame: pd.DataFrame, size: int) -> None:
    if len(frame) != size:
        raise RuntimeError(f"row count changed from {size} to {len(frame)}")
    actual = frame["_global_idx"].to_numpy(dtype=np.int64, na_value=-1)
    if not np.array_equal(actual, np.arange(size, dtype=np.int64)):
        raise RuntimeError("row order changed during relational join")


def join_task_split(task_frame: pd.DataFrame, source: pd.DataFrame, split: int) -> pd.DataFrame:
    target = task_frame[["SearchDate", "primary_key"]].copy()
    target["split"] = np.int8(split)
    target["row_idx"] = np.arange(len(target), dtype=np.int32)
    joined = target.merge(
        source,
        on="primary_key",
        how="left",
        sort=False,
        suffixes=("_task", ""),
        validate="one_to_one",
    )
    if len(joined) != len(target):
        raise RuntimeError("SearchStream join changed task row count")
    if not np.array_equal(joined["row_idx"].to_numpy(), np.arange(len(target), dtype=np.int32)):
        raise RuntimeError("SearchStream join changed task row order")
    if joined["SearchDate"].isna().any():
        raise RuntimeError("SearchStream join has missing rows")
    if not joined["SearchDate_task"].equals(joined["SearchDate"]):
        bad = int((joined["SearchDate_task"] != joined["SearchDate"]).sum())
        raise RuntimeError(f"SearchStream exact-time assertion failed for {bad} rows")
    return joined.drop(columns="SearchDate_task")


def build_spine() -> FeatureBundle:
    started = time.time()
    context = load_task(upto_test_timestamp=False)
    tables = {name: table.df for name, table in context.db.table_dict.items()}
    search_stream = tables["SearchStream"]
    grouped = search_stream.groupby("SearchID", sort=False, dropna=False)["HistCTR"]
    search_stream["SlateSize"] = search_stream.groupby("SearchID", sort=False, dropna=False)["AdID"].transform("size").astype(np.float32)
    search_stream["HistCTRRank"] = grouped.rank(method="average", ascending=False).astype(np.float32)
    search_stream["HistCTRPct"] = (search_stream["HistCTRRank"] / grouped.transform("count").clip(lower=1)).astype(np.float32)
    search_stream["HistCTRMax"] = grouped.transform("max").astype(np.float32)
    search_stream["HistCTRMean"] = grouped.transform("mean").astype(np.float32)
    source_columns = [
        "primary_key",
        "SearchID",
        "AdID",
        "Position",
        "ObjectType",
        "HistCTR",
        "SearchDate",
        "SlateSize",
        "HistCTRRank",
        "HistCTRPct",
        "HistCTRMax",
        "HistCTRMean",
    ]
    full_source = search_stream[source_columns]
    pre_source = full_source.loc[full_source["SearchDate"] < T_TIMESTAMP].drop(columns="primary_key").reset_index(drop=True)
    pre_source["primary_key"] = np.arange(len(pre_source), dtype=np.int64)
    train = join_task_split(context.train.df, pre_source, 0)
    val = join_task_split(context.val.df, pre_source, 1)
    test = join_task_split(context.test.df, full_source, 2)
    expected = (2_212_750, 1_177_380, 924_990)
    observed = (len(train), len(val), len(test))
    if observed != expected:
        raise RuntimeError(f"unexpected task counts {observed}, expected {expected}")
    spine = pd.concat([train, val, test], ignore_index=True)
    spine["_global_idx"] = np.arange(len(spine), dtype=np.int64)
    search_info = tables["SearchInfo"][
        ["SearchID", "UserID", "IPID", "IsUserLoggedOn", "SearchQuery", "LocationID", "CategoryID"]
    ].rename(columns={"LocationID": "SearchLocationID", "CategoryID": "SearchCategoryID"})
    spine = spine.merge(search_info, on="SearchID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    query_present = spine["SearchQuery"].notna().to_numpy()
    query_hash = pd.util.hash_pandas_object(spine["SearchQuery"].fillna(""), index=False).to_numpy(dtype=np.uint64)
    query_code = np.full(len(spine), np.nan, dtype=np.float64)
    query_code[query_present] = (query_hash[query_present] % np.uint64(2_147_483_647)).astype(np.float64)
    spine["QueryCode"] = query_code
    ads_info = tables["AdsInfo"][
        ["AdID", "LocationID", "CategoryID", "Price", "Title", "IsContext"]
    ].rename(columns={"LocationID": "AdLocationID", "CategoryID": "AdCategoryID"})
    spine = spine.merge(ads_info, on="AdID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    user_info = tables["UserInfo"]
    spine = spine.merge(user_info, on="UserID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    category = tables["Category"]
    search_category = category.rename(
        columns={
            "CategoryID": "SearchCategoryID",
            "Level": "SearchCategoryLevel",
            "ParentCategoryID": "SearchParentCategoryID",
            "SubcategoryID": "SearchSubcategoryID",
        }
    )
    ad_category = category.rename(
        columns={
            "CategoryID": "AdCategoryID",
            "Level": "AdCategoryLevel",
            "ParentCategoryID": "AdParentCategoryID",
            "SubcategoryID": "AdSubcategoryID",
        }
    )
    spine = spine.merge(search_category, on="SearchCategoryID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    spine = spine.merge(ad_category, on="AdCategoryID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    location = tables["Location"]
    search_location = location.rename(
        columns={
            "LocationID": "SearchLocationID",
            "Level": "SearchLocationLevel",
            "RegionID": "SearchRegionID",
            "CityID": "SearchCityID",
        }
    )
    ad_location = location.rename(
        columns={
            "LocationID": "AdLocationID",
            "Level": "AdLocationLevel",
            "RegionID": "AdRegionID",
            "CityID": "AdCityID",
        }
    )
    spine = spine.merge(search_location, on="SearchLocationID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    spine = spine.merge(ad_location, on="AdLocationID", how="left", sort=False, validate="many_to_one")
    assert_order(spine, sum(observed))
    labels_train = context.train.df[context.target_col].to_numpy(dtype=np.float32)
    labels_val = context.val.df[context.target_col].to_numpy(dtype=np.float32)
    phase("loaded and joined exact-key task spine", started)
    return FeatureBundle(
        context=context,
        spine=spine,
        labels_train=labels_train,
        labels_val=labels_val,
        n_train=observed[0],
        n_val=observed[1],
        n_test=observed[2],
        tables=tables,
    )


def numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan)


def build_static_features(spine: pd.DataFrame) -> pd.DataFrame:
    values: dict[str, np.ndarray] = {}
    hist = numeric(spine, "HistCTR")
    clipped_hist = np.clip(np.nan_to_num(hist, nan=0.0058), 1e-7, 1 - 1e-7)
    values["hist_ctr"] = hist
    values["hist_ctr_logit"] = np.log(clipped_hist / (1 - clipped_hist)).astype(np.float32)
    values["hist_ctr_missing"] = np.isnan(hist).astype(np.float32)
    for column in [
        "Position",
        "ObjectType",
        "SlateSize",
        "HistCTRRank",
        "HistCTRPct",
        "SearchID",
        "AdID",
        "UserID",
        "IPID",
        "IsUserLoggedOn",
        "SearchLocationID",
        "SearchCategoryID",
        "AdLocationID",
        "AdCategoryID",
        "IsContext",
        "UserAgentID",
        "UserAgentOSID",
        "UserDeviceID",
        "UserAgentFamilyID",
        "SearchCategoryLevel",
        "SearchParentCategoryID",
        "SearchSubcategoryID",
        "AdCategoryLevel",
        "AdParentCategoryID",
        "AdSubcategoryID",
        "SearchLocationLevel",
        "SearchRegionID",
        "SearchCityID",
        "AdLocationLevel",
        "AdRegionID",
        "AdCityID",
    ]:
        values[column.lower()] = numeric(spine, column)
    values["hist_gap_max"] = (numeric(spine, "HistCTRMax") - hist).astype(np.float32)
    values["hist_dev_mean"] = (hist - numeric(spine, "HistCTRMean")).astype(np.float32)
    dates = spine["SearchDate"]
    values["hour"] = dates.dt.hour.to_numpy(dtype=np.float32)
    values["weekday"] = dates.dt.dayofweek.to_numpy(dtype=np.float32)
    values["day"] = dates.dt.day.to_numpy(dtype=np.float32)
    values["age_days"] = ((dates - ORIGIN_TIMESTAMP).dt.total_seconds() / 86400).to_numpy(dtype=np.float32)
    values["days_to_v"] = ((V_TIMESTAMP - dates).dt.total_seconds() / 86400).to_numpy(dtype=np.float32)
    values["days_to_t"] = ((T_TIMESTAMP - dates).dt.total_seconds() / 86400).to_numpy(dtype=np.float32)
    query = spine["SearchQuery"].fillna("").astype(str)
    title = spine["Title"].fillna("").astype(str)
    values["query_present"] = spine["SearchQuery"].notna().to_numpy(dtype=np.float32)
    values["query_length"] = query.str.len().to_numpy(dtype=np.float32)
    values["query_words"] = query.str.count(r"\S+").to_numpy(dtype=np.float32)
    values["title_length"] = title.str.len().to_numpy(dtype=np.float32)
    values["title_words"] = title.str.count(r"\S+").to_numpy(dtype=np.float32)
    price = numeric(spine, "Price")
    values["price"] = np.clip(price, 0, 1e9).astype(np.float32)
    values["log_price"] = np.log1p(np.clip(price, 0, 1e9)).astype(np.float32)
    values["price_missing"] = np.isnan(price).astype(np.float32)
    comparisons = [
        ("category_match", "SearchCategoryID", "AdCategoryID"),
        ("parent_category_match", "SearchParentCategoryID", "AdParentCategoryID"),
        ("city_match", "SearchCityID", "AdCityID"),
        ("region_match", "SearchRegionID", "AdRegionID"),
        ("location_match", "SearchLocationID", "AdLocationID"),
    ]
    for name, left, right in comparisons:
        a = numeric(spine, left)
        b = numeric(spine, right)
        values[name] = (np.isfinite(a) & np.isfinite(b) & (a == b)).astype(np.float32)
    return pd.DataFrame(values, copy=False)


def splitmix64(values: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        result = values + np.uint64(0x9E3779B97F4A7C15)
        result = (result ^ (result >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        result = (result ^ (result >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return result ^ (result >> np.uint64(31))


def key_hash(frame: pd.DataFrame, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    size = len(frame)
    valid = np.ones(size, dtype=bool)
    hashed = np.full(size, np.uint64(0x6A09E667F3BCC909), dtype=np.uint64)
    for index, key in enumerate(keys):
        array = pd.to_numeric(frame[key], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        present = np.isfinite(array)
        valid &= present
        integer = np.zeros(size, dtype=np.int64)
        integer[present] = array[present].astype(np.int64)
        mixed = splitmix64(integer.view(np.uint64) + np.uint64(index * 0x9E3779B1))
        with np.errstate(over="ignore"):
            hashed = splitmix64(hashed ^ mixed)
    return hashed, valid


def seconds(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series).astype("int64").to_numpy(dtype=np.int64) // 1_000_000_000


def strict_history(
    events: pd.DataFrame,
    queries: pd.DataFrame,
    keys: list[str],
    event_time: str,
    query_time: str,
    windows: tuple[int, ...] = (),
    value_column: str | None = None,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "count": np.zeros(len(queries), dtype=np.float32),
        "recency_h": np.full(len(queries), np.nan, dtype=np.float32),
    }
    for window in windows:
        result[f"window_{window}"] = np.zeros(len(queries), dtype=np.float32)
    if value_column is not None:
        result["sum"] = np.zeros(len(queries), dtype=np.float32)
        for window in windows:
            result[f"window_sum_{window}"] = np.zeros(len(queries), dtype=np.float32)
    event_hash, event_valid = key_hash(events, keys)
    query_hash, query_valid = key_hash(queries, keys)
    event_seconds = seconds(events[event_time])
    query_seconds = seconds(queries[query_time])
    event_valid &= event_seconds > 0
    query_valid &= query_seconds > 0
    event_indices = np.flatnonzero(event_valid)
    query_indices = np.flatnonzero(query_valid)
    if len(event_indices) == 0 or len(query_indices) == 0:
        return result
    joined_hashes = np.concatenate([event_hash[event_indices], query_hash[query_indices]])
    codes, _ = pd.factorize(joined_hashes, sort=False, use_na_sentinel=False)
    event_codes = codes[: len(event_indices)].astype(np.int64, copy=False)
    query_codes = codes[len(event_indices) :].astype(np.int64, copy=False)
    del joined_hashes, codes, event_hash, query_hash
    event_time_values = event_seconds[event_indices]
    query_time_values = query_seconds[query_indices]
    origin = int(min(event_time_values.min(), query_time_values.min()))
    span = int(max(event_time_values.max(), query_time_values.max()) - origin + max(windows, default=0) + 2)
    base = max(span, 10_000_000)
    event_composite = event_codes * base + (event_time_values - origin)
    query_composite = query_codes * base + (query_time_values - origin)
    order = np.argsort(event_composite, kind="stable")
    sorted_composite = event_composite[order]
    starts = np.searchsorted(sorted_composite, query_codes * base, side="left")
    prior = np.searchsorted(sorted_composite, query_composite, side="left")
    counts = prior - starts
    result["count"][query_indices] = counts.astype(np.float32)
    has_prior = counts > 0
    if has_prior.any():
        prior_values = sorted_composite[prior[has_prior] - 1] - query_codes[has_prior] * base
        ages = (query_time_values[has_prior] - origin - prior_values) / 3600
        result["recency_h"][query_indices[has_prior]] = ages.astype(np.float32)
    for window in windows:
        lower = np.searchsorted(sorted_composite, query_composite - window, side="left")
        result[f"window_{window}"][query_indices] = (prior - np.maximum(lower, starts)).astype(np.float32)
    if value_column is not None:
        raw_values = pd.to_numeric(events[value_column], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
        ordered_values = np.nan_to_num(raw_values[event_indices][order], nan=0.0)
        cumulative = np.cumsum(ordered_values, dtype=np.float64)
        sums = np.zeros(len(query_indices), dtype=np.float64)
        if has_prior.any():
            selected = np.flatnonzero(has_prior)
            sums[selected] = cumulative[prior[has_prior] - 1]
            subtract = starts[has_prior] > 0
            if subtract.any():
                selected_subtract = selected[subtract]
                sums[selected_subtract] -= cumulative[starts[has_prior][subtract] - 1]
        result["sum"][query_indices] = sums.astype(np.float32)
        for window in windows:
            lower = np.searchsorted(sorted_composite, query_composite - window, side="left")
            lower = np.maximum(lower, starts)
            window_sums = np.zeros(len(query_indices), dtype=np.float64)
            has_window = prior > lower
            if has_window.any():
                selected = np.flatnonzero(has_window)
                window_sums[selected] = cumulative[prior[has_window] - 1]
                subtract = lower[has_window] > 0
                if subtract.any():
                    selected_subtract = selected[subtract]
                    window_sums[selected_subtract] -= cumulative[lower[has_window][subtract] - 1]
            result[f"window_sum_{window}"][query_indices] = window_sums.astype(np.float32)
    return result


def add_history(
    features: dict[str, np.ndarray],
    prefix: str,
    events: pd.DataFrame,
    queries: pd.DataFrame,
    keys: list[str],
    event_time: str,
    windows: tuple[int, ...],
    value_column: str | None = None,
) -> dict[str, np.ndarray]:
    started = time.time()
    history = strict_history(events, queries, keys, event_time, "SearchDate", windows, value_column)
    features[f"{prefix}_count"] = history["count"]
    features[f"{prefix}_recency_h"] = history["recency_h"]
    window_names = {3600: "1h", 86400: "1d", 259200: "3d", 518400: "6d"}
    for window in windows:
        features[f"{prefix}_{window_names[window]}"] = history[f"window_{window}"]
    if value_column is not None:
        features[f"{prefix}_sum"] = history["sum"]
    phase(f"built {prefix} strict history", started)
    return history


def first_occurrences(frame: pd.DataFrame, keys: list[str], distinct: str, time_column: str) -> pd.DataFrame:
    columns = keys + [distinct, time_column]
    clean = frame[columns].dropna(subset=keys + [distinct, time_column])
    return clean.groupby(keys + [distinct], sort=False, observed=True, as_index=False)[time_column].min()


def build_history_features(bundle: FeatureBundle) -> pd.DataFrame:
    started = time.time()
    features: dict[str, np.ndarray] = {}
    spine = bundle.spine
    search_stream = bundle.tables["SearchStream"]
    search_info = bundle.tables["SearchInfo"]
    ads_info = bundle.tables["AdsInfo"][["AdID", "CategoryID", "Price"]].rename(columns={"CategoryID": "AdCategoryID"})
    stream_events = search_stream[["SearchID", "AdID", "Position", "SearchDate"]].merge(
        search_info[["SearchID", "UserID", "IPID", "CategoryID"]].rename(columns={"CategoryID": "SearchCategoryID"}),
        on="SearchID",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    stream_events = stream_events.merge(ads_info[["AdID", "AdCategoryID"]], on="AdID", how="left", sort=False, validate="many_to_one")
    one_day = (86400,)
    broad = (3600, 86400, 259200, 518400)
    stream_specs = [
        ("imp_user", ["UserID"], broad),
        ("imp_ip", ["IPID"], broad),
        ("imp_ad", ["AdID"], broad),
        ("imp_user_ad", ["UserID", "AdID"], one_day),
        ("imp_ip_ad", ["IPID", "AdID"], one_day),
        ("imp_category", ["AdCategoryID"], broad),
        ("imp_ad_position", ["AdID", "Position"], one_day),
    ]
    for prefix, keys, windows in stream_specs:
        add_history(features, prefix, stream_events, spine, keys, "SearchDate", windows)
    search_events = search_info[["UserID", "IPID", "CategoryID", "SearchDate", "SearchQuery"]].copy()
    search_events["QueryPresent"] = search_events["SearchQuery"].notna().astype(np.float32)
    for prefix, key in [("search_user", "UserID"), ("search_ip", "IPID")]:
        history = add_history(features, prefix, search_events, spine, [key], "SearchDate", broad, "QueryPresent")
        features[f"{prefix}_query_rate"] = ((history["sum"] + 1) / (history["count"] + 2)).astype(np.float32)
        distinct = first_occurrences(search_events, [key], "CategoryID", "SearchDate")
        diversity = strict_history(distinct, spine, [key], "SearchDate", "SearchDate")
        features[f"{prefix}_category_diversity"] = diversity["count"]
        del distinct, diversity
    del search_events
    event_tables = [
        ("visit", bundle.tables["VisitStream"], "ViewDate"),
        ("phone", bundle.tables["PhoneRequestsStream"], "PhoneRequestDate"),
    ]
    for event_name, raw_events, time_column in event_tables:
        event_frame = raw_events[["UserID", "IPID", "AdID", time_column]].merge(
            ads_info[["AdID", "AdCategoryID"]], on="AdID", how="left", sort=False, validate="many_to_one"
        )
        for entity in ["UserID", "IPID"]:
            short = "user" if entity == "UserID" else "ip"
            prefix = f"{event_name}_{short}"
            add_history(features, prefix, event_frame, spine, [entity], time_column, broad)
            distinct = first_occurrences(event_frame, [entity], "AdID", time_column)
            distinct_history = strict_history(distinct, spine, [entity], time_column, "SearchDate")
            features[f"{prefix}_distinct_ads"] = distinct_history["count"]
            del distinct, distinct_history
            add_history(features, f"{prefix}_ad", event_frame, spine, [entity, "AdID"], time_column, one_day)
            add_history(
                features,
                f"{prefix}_category",
                event_frame,
                spine,
                [entity, "AdCategoryID"],
                time_column,
                one_day,
            )
        del event_frame
        gc.collect()
    first_ads = stream_events[["AdID", "SearchDate"]].dropna().groupby("AdID", sort=False, as_index=False)["SearchDate"].min()
    first_ads = first_ads.merge(ads_info, on="AdID", how="left", sort=False, validate="one_to_one").dropna(subset=["AdCategoryID", "Price"])
    cohort = add_history(
        features,
        "price_category_ads",
        first_ads,
        spine,
        ["AdCategoryID"],
        "SearchDate",
        (),
        "Price",
    )
    cohort_mean = cohort["sum"] / np.maximum(cohort["count"], 1)
    current_price = np.clip(numeric(spine, "Price"), 0, 1e9)
    features["price_category_mean"] = np.clip(cohort_mean, 0, 1e9).astype(np.float32)
    features["price_category_log_deviation"] = (
        np.log1p(current_price) - np.log1p(np.clip(cohort_mean, 0, 1e9))
    ).astype(np.float32)
    ratio_specs = [
        ("user", "visit_user_count", "imp_user_count", "visit_impression_user"),
        ("ip", "visit_ip_count", "imp_ip_count", "visit_impression_ip"),
        ("user", "phone_user_count", "imp_user_count", "phone_impression_user"),
        ("ip", "phone_ip_count", "imp_ip_count", "phone_impression_ip"),
        ("user", "phone_user_count", "visit_user_count", "phone_visit_user"),
        ("ip", "phone_ip_count", "visit_ip_count", "phone_visit_ip"),
        ("user_ad", "visit_user_ad_count", "imp_user_ad_count", "visit_impression_user_ad"),
        ("ip_ad", "visit_ip_ad_count", "imp_ip_ad_count", "visit_impression_ip_ad"),
        ("user_ad", "phone_user_ad_count", "imp_user_ad_count", "phone_impression_user_ad"),
        ("ip_ad", "phone_ip_ad_count", "imp_ip_ad_count", "phone_impression_ip_ad"),
    ]
    for _, numerator, denominator, output in ratio_specs:
        features[output] = ((features[numerator] + 0.5) / (features[denominator] + 10)).astype(np.float32)
    phase("completed all-table non-label histories", started)
    return pd.DataFrame(features, copy=False)


def build_context_history_features(bundle: FeatureBundle) -> pd.DataFrame:
    started = time.time()
    features: dict[str, np.ndarray] = {}
    spine = bundle.spine
    search_stream = bundle.tables["SearchStream"]
    search_info = bundle.tables["SearchInfo"]
    ads_info = bundle.tables["AdsInfo"][["AdID", "CategoryID"]].rename(columns={"CategoryID": "AdCategoryID"})
    events = search_stream[
        ["SearchID", "AdID", "Position", "ObjectType", "HistCTR", "SearchDate"]
    ].merge(
        search_info[["SearchID", "UserID", "IPID"]],
        on="SearchID",
        how="left",
        sort=False,
        validate="many_to_one",
    )
    events = events.merge(ads_info, on="AdID", how="left", sort=False, validate="many_to_one")
    events = events.loc[events["ObjectType"] == 3].reset_index(drop=True)
    broad = (3600, 86400, 259200, 518400)
    for prefix, keys in [
        ("context_imp_user", ["UserID"]),
        ("context_imp_ip", ["IPID"]),
        ("context_imp_ad", ["AdID"]),
        ("context_imp_category", ["AdCategoryID"]),
    ]:
        history = add_history(
            features,
            prefix,
            events,
            spine,
            keys,
            "SearchDate",
            broad,
            "HistCTR",
        )
        mean = history["sum"] / np.maximum(history["count"], 1)
        features[f"{prefix}_hist_mean"] = mean.astype(np.float32)
        features[f"{prefix}_hist_deviation"] = (numeric(spine, "HistCTR") - mean).astype(np.float32)
    add_history(
        features,
        "context_imp_user_ad",
        events,
        spine,
        ["UserID", "AdID"],
        "SearchDate",
        (86400,),
    )
    phase("completed contextual-impression extension", started)
    return pd.DataFrame(features, copy=False)


def build_query_history_features(bundle: FeatureBundle) -> pd.DataFrame:
    started = time.time()
    search_info = bundle.tables["SearchInfo"][["SearchDate", "SearchQuery"]].copy()
    present = search_info["SearchQuery"].notna().to_numpy()
    hashed = pd.util.hash_pandas_object(search_info["SearchQuery"].fillna(""), index=False).to_numpy(dtype=np.uint64)
    codes = np.full(len(search_info), np.nan, dtype=np.float64)
    codes[present] = (hashed[present] % np.uint64(2_147_483_647)).astype(np.float64)
    search_info["QueryCode"] = codes
    features: dict[str, np.ndarray] = {}
    add_history(
        features,
        "query_activity",
        search_info,
        bundle.spine,
        ["QueryCode"],
        "SearchDate",
        (3600, 86400, 259200, 518400),
    )
    phase("completed query-activity extension", started)
    return pd.DataFrame(features, copy=False)


def aggregate_stats(
    fit: pd.DataFrame,
    labels: np.ndarray,
    query: pd.DataFrame,
    keys: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    fit_hash, fit_valid = key_hash(fit, keys)
    query_hash, query_valid = key_hash(query, keys)
    fit_indices = np.flatnonzero(fit_valid)
    query_indices = np.flatnonzero(query_valid)
    counts_out = np.zeros(len(query), dtype=np.float32)
    clicks_out = np.zeros(len(query), dtype=np.float32)
    if len(fit_indices) == 0 or len(query_indices) == 0:
        return counts_out, clicks_out
    combined = np.concatenate([fit_hash[fit_indices], query_hash[query_indices]])
    codes, _ = pd.factorize(combined, sort=False, use_na_sentinel=False)
    fit_codes = codes[: len(fit_indices)]
    query_codes = codes[len(fit_indices) :]
    size = int(codes.max()) + 1
    counts = np.bincount(fit_codes, minlength=size)
    clicks = np.bincount(fit_codes, weights=labels[fit_indices], minlength=size)
    counts_out[query_indices] = counts[query_codes].astype(np.float32)
    clicks_out[query_indices] = clicks[query_codes].astype(np.float32)
    return counts_out, clicks_out


def collect_label_payload(
    fit: pd.DataFrame,
    labels: np.ndarray,
    query: pd.DataFrame,
    include_causal_fit: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray]:
    started = time.time()
    payload: dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray] = {}
    if include_causal_fit:
        fit_constant = fit[["SearchDate"]].copy()
        fit_constant["GlobalKey"] = 1
        global_history = strict_history(
            fit_constant.assign(Label=labels),
            fit_constant,
            ["GlobalKey"],
            "SearchDate",
            "SearchDate",
            value_column="Label",
        )
        global_count = np.concatenate(
            [global_history["count"], np.full(len(query), len(fit), dtype=np.float32)]
        )
        global_click = np.concatenate(
            [global_history["sum"], np.full(len(query), float(labels.sum()), dtype=np.float32)]
        )
    else:
        global_count = np.full(len(query), len(fit), dtype=np.float32)
        global_click = np.full(len(query), float(labels.sum()), dtype=np.float32)
    payload["global_count"] = global_count
    payload["global_click"] = global_click
    labeled_fit = fit.assign(Label=labels)
    for name, keys, _ in LABEL_GRAINS:
        query_count, query_click = aggregate_stats(fit, labels, query, keys)
        if include_causal_fit:
            history = strict_history(
                labeled_fit,
                fit,
                keys,
                "SearchDate",
                "SearchDate",
                value_column="Label",
            )
            count = np.concatenate([history["count"], query_count])
            click = np.concatenate([history["sum"], query_click])
        else:
            count = query_count
            click = query_click
        payload[name] = (count.astype(np.float32), click.astype(np.float32))
    phase("built causal label-stat payload", started)
    return payload


def strength_for(mode: str, name: str, parents: tuple[str, str] | None) -> float:
    if mode == "light":
        return 50.0
    if mode == "heavy":
        return 1000.0
    if parents is not None:
        return 1000.0
    if name in {"position", "object", "ad_category", "search_category", "ad_city", "search_city", "ad_region", "search_region"}:
        return 50.0
    return 200.0


def render_label_features(
    payload: dict[str, tuple[np.ndarray, np.ndarray] | np.ndarray],
    mode: str,
) -> pd.DataFrame:
    global_count = np.asarray(payload["global_count"], dtype=np.float32)
    global_click = np.asarray(payload["global_click"], dtype=np.float32)
    global_prior = np.divide(global_click, global_count, out=np.full_like(global_click, 0.006), where=global_count > 0)
    features: dict[str, np.ndarray] = {
        "label_global_count": global_count,
        "label_global_clicks": global_click,
        "label_global_ctr": global_prior.astype(np.float32),
    }
    posterior: dict[str, np.ndarray] = {}
    for name, _, parents in LABEL_GRAINS:
        count, click = payload[name]
        count = np.asarray(count, dtype=np.float32)
        click = np.asarray(click, dtype=np.float32)
        if parents is None:
            backoff = global_prior
        else:
            backoff = (posterior[parents[0]] + posterior[parents[1]] + global_prior) / 3
        strength = strength_for(mode, name, parents)
        ctr = (click + strength * backoff) / (count + strength)
        clipped = np.clip(ctr, 1e-7, 1 - 1e-7)
        uncertainty = np.sqrt(clipped * (1 - clipped) / (count + strength + 1))
        posterior[name] = ctr.astype(np.float32)
        features[f"label_{name}_count"] = count
        features[f"label_{name}_clicks"] = click
        features[f"label_{name}_ctr"] = ctr.astype(np.float32)
        features[f"label_{name}_logit"] = np.log(clipped / (1 - clipped)).astype(np.float32)
        features[f"label_{name}_uncertainty"] = uncertainty.astype(np.float32)
    return pd.DataFrame(features, copy=False)


def build_base_features(
    bundle: FeatureBundle,
    debug: bool,
    context_extension: bool = False,
    query_extension: bool = False,
) -> pd.DataFrame:
    started = time.time()
    static = build_static_features(bundle.spine)
    phase("built static feature matrix", started)
    if debug:
        return static
    cache_root = shared_cache_dir() / "generic_exp_0_lane0_v3"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / "base_features.pkl"
    if cache_path.exists():
        cached = pd.read_pickle(cache_path)
        if len(cached) == len(bundle.spine):
            phase("loaded cached all-table base matrix", started)
            if context_extension:
                extension = build_context_history_features(bundle)
                cached = pd.concat([cached, extension], axis=1, copy=False)
            if query_extension:
                extension = build_query_history_features(bundle)
                cached = pd.concat([cached, extension], axis=1, copy=False)
            return cached
        raise RuntimeError("cached base feature matrix row mismatch")
    histories = build_history_features(bundle)
    result = pd.concat([static, histories], axis=1, copy=False)
    if len(result) != len(bundle.spine):
        raise RuntimeError("base feature matrix row mismatch")
    temporary = cache_root / f"base_features.{os.getpid()}.tmp"
    result.to_pickle(temporary)
    os.replace(temporary, cache_path)
    phase("cached all-table base matrix", started)
    return result
