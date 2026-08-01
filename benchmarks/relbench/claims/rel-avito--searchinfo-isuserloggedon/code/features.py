from __future__ import annotations

import gc
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


BEHAVIOR_VERSION = "generic_exp_1_coverage_router_behavior_v1"


def _elapsed(start: float, phase: str) -> None:
    print(f"[features] {phase}: {time.time() - start:.1f}s", flush=True)


def _cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    return root


def _database_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db"


def _cross_code(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].fillna(-1)
    return (pd.util.hash_pandas_object(values, index=False).to_numpy(dtype=np.uint64) % np.uint64(2147483629)).astype(np.int32)


def _task_frame(ctx) -> pd.DataFrame:
    pieces = []
    target = ctx.target_col
    for split_code, table in enumerate((ctx.train, ctx.val, ctx.test)):
        part = table.df.copy()
        part["_split"] = np.int8(split_code)
        part["_split_row_id"] = np.arange(len(part), dtype=np.int32)
        if target not in part:
            part[target] = np.nan
        pieces.append(part[["SearchID", "SearchDate", target, "_split", "_split_row_id"]])
    task_rows = pd.concat(pieces, ignore_index=True)
    task_rows.rename(columns={target: "_target"}, inplace=True)
    task_rows["_row_id"] = np.arange(len(task_rows), dtype=np.int32)
    return task_rows


def _load_safe_search(task_rows: pd.DataFrame) -> pd.DataFrame:
    path = _database_root() / "SearchInfo.parquet"
    columns = ["UserID", "SearchID", "IPID", "SearchQuery", "LocationID", "CategoryID"]
    search = pd.read_parquet(path, columns=columns)
    frame = task_rows.merge(search, on="SearchID", how="left", sort=False, validate="one_to_one")
    if frame["UserID"].isna().all() or not np.array_equal(frame["_row_id"].to_numpy(), np.arange(len(frame))):
        raise RuntimeError("SearchInfo join failed to preserve task row order")
    return frame


def _add_static_features(frame: pd.DataFrame) -> pd.DataFrame:
    root = _database_root()
    users = pd.read_parquet(root / "UserInfo.parquet").rename(
        columns={
            "UserAgentID": "device_agent",
            "UserAgentOSID": "device_os",
            "UserDeviceID": "device_id",
            "UserAgentFamilyID": "device_family",
        }
    )
    categories = pd.read_parquet(root / "Category.parquet").rename(
        columns={
            "Level": "search_category_level",
            "ParentCategoryID": "search_parent_category",
            "SubcategoryID": "search_subcategory",
        }
    )
    locations = pd.read_parquet(root / "Location.parquet").rename(
        columns={
            "Level": "search_location_level",
            "RegionID": "search_region",
            "CityID": "search_city",
        }
    )
    frame = frame.merge(users, on="UserID", how="left", sort=False, validate="many_to_one")
    frame = frame.merge(categories, on="CategoryID", how="left", sort=False, validate="many_to_one")
    frame = frame.merge(locations, on="LocationID", how="left", sort=False, validate="many_to_one")
    query = frame.pop("SearchQuery")
    present = query.notna()
    frame["query_present"] = present.astype(np.int8)
    frame["query_length"] = query.str.len().fillna(0).clip(0, 500).astype(np.int16)
    frame["query_words"] = (query.str.count(r"\s+").fillna(-1) + 1).clip(0, 50).astype(np.int8)
    frame["query_digits"] = query.str.count(r"\d").fillna(0).clip(0, 50).astype(np.int8)
    frame["hour"] = frame["SearchDate"].dt.hour.astype(np.int8)
    frame["hour_band"] = (frame["hour"] // 4).astype(np.int8)
    frame["day_of_week"] = frame["SearchDate"].dt.dayofweek.astype(np.int8)
    frame["day_index"] = ((frame["SearchDate"].dt.floor("D") - pd.Timestamp("2015-04-25")) / pd.Timedelta(days=1)).astype(np.int8)
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(np.int8)
    frame["device_cross"] = _cross_code(frame, ["device_agent", "device_os", "device_id", "device_family"])
    frame["category_location_cross"] = _cross_code(frame, ["CategoryID", "LocationID"])
    frame["device_category_cross"] = _cross_code(frame, ["device_cross", "CategoryID"])
    frame["hour_category_cross"] = _cross_code(frame, ["hour_band", "CategoryID"])
    return frame


def _add_prior_search_features(frame: pd.DataFrame, key: str, prefix: str) -> None:
    n = len(frame)
    count = np.zeros(n, dtype=np.int32)
    count_1d = np.zeros(n, dtype=np.int32)
    count_7d = np.zeros(n, dtype=np.int32)
    age = np.full(n, np.nan, dtype=np.float32)
    since_first = np.zeros(n, dtype=np.float32)
    valid = frame[key].notna().to_numpy()
    positions = np.flatnonzero(valid)
    work = pd.DataFrame(
        {
            "position": positions,
            "key": frame.loc[valid, key].to_numpy(dtype=np.int64),
            "time": frame.loc[valid, "SearchDate"].astype("int64").to_numpy(),
        }
    ).sort_values(["key", "time", "position"], kind="mergesort")
    day_ns = np.int64(86_400_000_000_000)
    for locs in work.groupby("key", sort=False).indices.values():
        pos = work.iloc[locs]["position"].to_numpy(dtype=np.int64)
        times = work.iloc[locs]["time"].to_numpy(dtype=np.int64)
        before = np.searchsorted(times, times, side="left")
        lower_1d = np.searchsorted(times, times - day_ns, side="left")
        lower_7d = np.searchsorted(times, times - 7 * day_ns, side="left")
        count[pos] = before
        count_1d[pos] = before - lower_1d
        count_7d[pos] = before - lower_7d
        has_previous = before > 0
        if has_previous.any():
            age[pos[has_previous]] = ((times[has_previous] - times[before[has_previous] - 1]) / 1e9).astype(np.float32)
        since_first[pos] = ((times - times[0]) / 1e9).astype(np.float32)
    frame[f"{prefix}_past_search_count"] = count
    frame[f"{prefix}_past_search_1d"] = count_1d
    frame[f"{prefix}_past_search_7d"] = count_7d
    frame[f"{prefix}_last_search_age"] = age
    frame[f"{prefix}_search_since_first"] = since_first


def _same_search_aggregates() -> pd.DataFrame:
    root = _database_root()
    search_stream = str(root / "SearchStream.parquet")
    search_info = str(root / "SearchInfo.parquet")
    ads = str(root / "AdsInfo.parquet")
    categories = str(root / "Category.parquet")
    locations = str(root / "Location.parquet")
    con = duckdb.connect()
    query = f"""
        SELECT ss.SearchID,
               count(*)::INTEGER AS result_count,
               count(DISTINCT ss.AdID)::INTEGER AS result_unique_ads,
               avg(ss.Position)::FLOAT AS result_position_mean,
               min(ss.Position)::FLOAT AS result_position_min,
               max(ss.Position)::FLOAT AS result_position_max,
               stddev_pop(ss.Position)::FLOAT AS result_position_std,
               avg(ss.ObjectType)::FLOAT AS result_object_mean,
               count(DISTINCT ss.ObjectType)::INTEGER AS result_object_breadth,
               avg((ss.ObjectType = 1)::INTEGER)::FLOAT AS result_object1_rate,
               avg((ss.ObjectType = 3)::INTEGER)::FLOAT AS result_object3_rate,
               count(ss.HistCTR)::INTEGER AS result_histctr_count,
               avg(ss.HistCTR)::FLOAT AS result_histctr_mean,
               min(ss.HistCTR)::FLOAT AS result_histctr_min,
               max(ss.HistCTR)::FLOAT AS result_histctr_max,
               stddev_pop(ss.HistCTR)::FLOAT AS result_histctr_std,
               avg(ln(1 + greatest(a.Price, 0)))::FLOAT AS result_logprice_mean,
               min(ln(1 + greatest(a.Price, 0)))::FLOAT AS result_logprice_min,
               max(ln(1 + greatest(a.Price, 0)))::FLOAT AS result_logprice_max,
               stddev_pop(ln(1 + greatest(a.Price, 0)))::FLOAT AS result_logprice_std,
               avg(a.IsContext)::FLOAT AS result_context_rate,
               avg(length(a.Title))::FLOAT AS result_title_length,
               avg((a.CategoryID = si.CategoryID)::INTEGER)::FLOAT AS result_category_match,
               avg((a.LocationID = si.LocationID)::INTEGER)::FLOAT AS result_location_match,
               avg((ac.ParentCategoryID = sc.ParentCategoryID)::INTEGER)::FLOAT AS result_parent_match,
               avg((al.RegionID = sl.RegionID)::INTEGER)::FLOAT AS result_region_match,
               count(DISTINCT a.CategoryID)::INTEGER AS result_category_breadth,
               count(DISTINCT a.LocationID)::INTEGER AS result_location_breadth
        FROM read_parquet('{search_stream}') ss
        JOIN read_parquet('{search_info}') si ON ss.SearchID = si.SearchID
        LEFT JOIN read_parquet('{ads}') a ON ss.AdID = a.AdID
        LEFT JOIN read_parquet('{categories}') ac ON a.CategoryID = ac.CategoryID
        LEFT JOIN read_parquet('{categories}') sc ON si.CategoryID = sc.CategoryID
        LEFT JOIN read_parquet('{locations}') al ON a.LocationID = al.LocationID
        LEFT JOIN read_parquet('{locations}') sl ON si.LocationID = sl.LocationID
        WHERE ss.SearchDate <= si.SearchDate
        GROUP BY ss.SearchID
    """
    result = con.execute(query).fetchdf()
    con.close()
    return result


def _event_projection(table: str, time_col: str) -> pd.DataFrame:
    root = _database_root()
    event_path = str(root / f"{table}.parquet")
    ads_path = str(root / "AdsInfo.parquet")
    con = duckdb.connect()
    query = f"""
        SELECT e.UserID, e.IPID, e.AdID, e.{time_col} AS event_time,
               ln(1 + greatest(a.Price, 0))::FLOAT AS event_logprice,
               a.IsContext::FLOAT AS event_context,
               a.CategoryID AS event_category,
               a.LocationID AS event_location
        FROM read_parquet('{event_path}') e
        LEFT JOIN read_parquet('{ads_path}') a ON e.AdID = a.AdID
    """
    result = con.execute(query).fetchdf()
    con.close()
    return result


def _event_state(frame: pd.DataFrame, events: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    events = events.loc[events[key].notna()].copy()
    events[key] = events[key].astype(np.int64)
    events.sort_values([key, "event_time", "AdID"], inplace=True, kind="mergesort")
    groups = events.groupby(key, sort=False)
    events["state_count"] = groups.cumcount().to_numpy(dtype=np.int32) + 1
    events["state_new_ad"] = (~events.duplicated([key, "AdID"])).astype(np.int8)
    events["state_new_category"] = (~events.duplicated([key, "event_category"])).astype(np.int8)
    events["state_ad_breadth"] = events.groupby(key, sort=False)["state_new_ad"].cumsum().astype(np.int32)
    events["state_category_breadth"] = events.groupby(key, sort=False)["state_new_category"].cumsum().astype(np.int16)
    for source, name in (("event_logprice", "price"), ("event_context", "context")):
        valid = events[source].notna().astype(np.int8)
        events[f"state_{name}_sum"] = events[source].fillna(0).groupby(events[key], sort=False).cumsum().astype(np.float32)
        events[f"state_{name}_count"] = valid.groupby(events[key], sort=False).cumsum().astype(np.int32)
    state_columns = [
        key,
        "event_time",
        "state_count",
        "state_ad_breadth",
        "state_category_breadth",
        "state_price_sum",
        "state_price_count",
        "state_context_sum",
        "state_context_count",
    ]
    right = events[state_columns].sort_values(["event_time", key], kind="mergesort")
    valid_rows = frame[key].notna().to_numpy()
    left = pd.DataFrame(
        {
            "position": np.flatnonzero(valid_rows),
            key: frame.loc[valid_rows, key].to_numpy(dtype=np.int64),
            "SearchDate": frame.loc[valid_rows, "SearchDate"].to_numpy(),
        }
    ).sort_values(["SearchDate", key], kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="SearchDate",
        right_on="event_time",
        by=key,
        direction="backward",
        allow_exact_matches=False,
    )
    output = pd.DataFrame(index=np.arange(len(frame)))
    output.loc[merged["position"], f"{prefix}_count"] = merged["state_count"].to_numpy()
    output.loc[merged["position"], f"{prefix}_ad_breadth"] = merged["state_ad_breadth"].to_numpy()
    output.loc[merged["position"], f"{prefix}_category_breadth"] = merged["state_category_breadth"].to_numpy()
    output.loc[merged["position"], f"{prefix}_recency"] = (merged["SearchDate"] - merged["event_time"]).dt.total_seconds().to_numpy()
    price_n = merged["state_price_count"].to_numpy(dtype=np.float64)
    context_n = merged["state_context_count"].to_numpy(dtype=np.float64)
    output.loc[merged["position"], f"{prefix}_logprice_mean"] = merged["state_price_sum"].to_numpy() / np.maximum(price_n, 1)
    output.loc[merged["position"], f"{prefix}_context_rate"] = merged["state_context_sum"].to_numpy() / np.maximum(context_n, 1)
    return output.astype(np.float32)


def _add_event_features(frame: pd.DataFrame, table: str, time_col: str, name: str) -> None:
    events = _event_projection(table, time_col)
    for key, key_name in (("UserID", "user"), ("IPID", "ip")):
        states = _event_state(frame, events, key, f"{key_name}_{name}")
        for column in states:
            frame[column] = states[column].to_numpy(dtype=np.float32)
        del states
        gc.collect()
    del events
    gc.collect()


def build_behavior_store(ctx, use_cache: bool = True) -> tuple[pd.DataFrame, list[str], list[str]]:
    start = time.time()
    cache_path = _cache_root() / f"{BEHAVIOR_VERSION}.pkl"
    if use_cache and cache_path.exists():
        frame = pd.read_pickle(cache_path)
        if len(frame) == len(ctx.train) + len(ctx.val) + len(ctx.test):
            _elapsed(start, "loaded cached all-table store")
            model_columns = _model_columns(frame)
            return frame, model_columns, _categorical_columns(model_columns)
    frame = _task_frame(ctx)
    frame = _load_safe_search(frame)
    frame = _add_static_features(frame)
    _elapsed(start, "task, search, device, and hierarchy features")
    _add_prior_search_features(frame, "UserID", "user")
    _add_prior_search_features(frame, "IPID", "ip")
    _elapsed(start, "causal prior-search states")
    results = _same_search_aggregates()
    frame = frame.merge(results, on="SearchID", how="left", sort=False, validate="one_to_one")
    del results
    gc.collect()
    _elapsed(start, "same-search result and ad composition")
    _add_event_features(frame, "VisitStream", "ViewDate", "visit")
    _elapsed(start, "causal visit states")
    _add_event_features(frame, "PhoneRequestsStream", "PhoneRequestDate", "phone")
    _elapsed(start, "causal phone-request states")
    if not np.array_equal(frame["_row_id"].to_numpy(), np.arange(len(frame))):
        frame.sort_values("_row_id", inplace=True)
        frame.reset_index(drop=True, inplace=True)
    for column in _model_columns(frame):
        if frame[column].dtype == np.float64:
            frame[column] = frame[column].astype(np.float32)
        elif frame[column].dtype == np.int64:
            minimum = frame[column].min()
            maximum = frame[column].max()
            if minimum >= np.iinfo(np.int32).min and maximum <= np.iinfo(np.int32).max:
                frame[column] = frame[column].astype(np.int32)
    frame.to_pickle(cache_path)
    _elapsed(start, "cached all-table store")
    model_columns = _model_columns(frame)
    return frame, model_columns, _categorical_columns(model_columns)


def _model_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "_split",
        "_split_row_id",
        "_row_id",
        "_target",
        "SearchID",
        "SearchDate",
        "UserID",
        "IPID",
    }
    return [column for column in frame.columns if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])]


def _categorical_columns(model_columns: list[str]) -> list[str]:
    candidates = {
        "CategoryID",
        "search_category_level",
        "search_parent_category",
        "search_subcategory",
        "hour",
        "hour_band",
        "day_of_week",
        "is_weekend",
        "query_present",
    }
    return [column for column in model_columns if column in candidates]


class FrozenLabelBuilder:
    def __init__(self, labeled: pd.DataFrame):
        self.labeled = labeled.copy()
        valid = self.labeled["UserID"].notna()
        ordered = self.labeled.loc[valid].sort_values(["UserID", "SearchDate", "SearchID"], kind="mergesort").copy()
        ordered["_previous_user_label"] = ordered.groupby("UserID", sort=False)["_target"].shift(1)
        previous_time = ordered.groupby("UserID", sort=False)["SearchDate"].shift(1)
        ordered["_transition"] = ((ordered["_target"] != ordered["_previous_user_label"]) & ordered["_previous_user_label"].notna()).astype(np.int8)
        ordered["_transition_cumulative"] = ordered.groupby("UserID", sort=False)["_transition"].cumsum().astype(np.int32)
        new_run = ((ordered["_target"] != ordered["_previous_user_label"]) | ordered["_previous_user_label"].isna()).astype(np.int8)
        run_id = new_run.groupby(ordered["UserID"], sort=False).cumsum()
        ordered["_streak"] = ordered.groupby([ordered["UserID"], run_id], sort=False).cumcount().astype(np.int32) + 1
        gap = (ordered["SearchDate"] - previous_time).dt.total_seconds().to_numpy()
        ordered["_history_gap_bucket"] = self._gap_bucket(gap)
        auxiliary = ordered[["_row_id", "_previous_user_label", "_transition_cumulative", "_streak", "_history_gap_bucket"]]
        self.labeled = self.labeled.merge(auxiliary, on="_row_id", how="left", sort=False, validate="one_to_one")

    @staticmethod
    def _gap_bucket(seconds: np.ndarray) -> np.ndarray:
        values = np.asarray(seconds, dtype=np.float64)
        return np.select(
            [values < 1800, values < 21600, values < 86400, values < 259200, values < 604800],
            [0, 1, 2, 3, 4],
            default=5,
        ).astype(np.int8)

    @staticmethod
    def _map_single(batch: pd.DataFrame, key: str, series: pd.Series) -> np.ndarray:
        return batch[key].map(series).to_numpy(dtype=np.float64)

    @staticmethod
    def _map_multi(batch: pd.DataFrame, keys: list[str], series: pd.Series) -> np.ndarray:
        index = pd.MultiIndex.from_frame(batch[keys])
        return series.reindex(index).to_numpy(dtype=np.float64)

    def transform(self, batch: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        cutoff = pd.Timestamp(cutoff)
        history = self.labeled.loc[self.labeled["SearchDate"] < cutoff]
        prior = float(history["_target"].mean()) if len(history) else 0.35
        output = pd.DataFrame(index=batch.index)
        output["global_label_prior"] = np.float32(prior)
        user_history = history.loc[history["UserID"].notna()].sort_values(["UserID", "SearchDate", "SearchID"], kind="mergesort")
        if len(user_history):
            aggregates = user_history.groupby("UserID", sort=False)["_target"].agg(["sum", "count"])
            count = self._map_single(batch, "UserID", aggregates["count"])
            total = self._map_single(batch, "UserID", aggregates["sum"])
            last = user_history.groupby("UserID", sort=False).tail(1).set_index("UserID")
            output["user_history_count"] = np.nan_to_num(count).astype(np.float32)
            output["user_label_mean"] = np.where(np.isfinite(count), (np.nan_to_num(total) + 20 * prior) / (np.nan_to_num(count) + 20), prior).astype(np.float32)
            output["user_last_label"] = np.nan_to_num(self._map_single(batch, "UserID", last["_target"]), nan=prior).astype(np.float32)
            output["user_second_label"] = np.nan_to_num(self._map_single(batch, "UserID", last["_previous_user_label"]), nan=prior).astype(np.float32)
            output["user_label_streak"] = np.nan_to_num(self._map_single(batch, "UserID", last["_streak"])).astype(np.float32)
            transitions = self._map_single(batch, "UserID", last["_transition_cumulative"])
            output["user_transition_rate"] = (np.nan_to_num(transitions) / np.maximum(np.nan_to_num(count) - 1, 1)).astype(np.float32)
            last_time = batch["UserID"].map(last["SearchDate"])
            age = (batch["SearchDate"] - last_time).dt.total_seconds().to_numpy(dtype=np.float64)
            output["user_last_label_age"] = age.astype(np.float32)
            output["user_gap_bucket"] = self._gap_bucket(age)
            for days in (1, 3, 7):
                recent = user_history.loc[user_history["SearchDate"] >= cutoff - pd.Timedelta(days=days)]
                means = recent.groupby("UserID", sort=False)["_target"].mean()
                output[f"user_label_mean_{days}d"] = np.nan_to_num(self._map_single(batch, "UserID", means), nan=prior).astype(np.float32)
        else:
            output["user_history_count"] = np.zeros(len(batch), dtype=np.float32)
            for name in ("user_label_mean", "user_last_label", "user_second_label", "user_label_mean_1d", "user_label_mean_3d", "user_label_mean_7d"):
                output[name] = np.full(len(batch), prior, dtype=np.float32)
            for name in ("user_label_streak", "user_transition_rate", "user_last_label_age"):
                output[name] = np.zeros(len(batch), dtype=np.float32)
            output["user_gap_bucket"] = np.full(len(batch), 5, dtype=np.int8)
        ip_history = history.loc[history["IPID"].notna()].sort_values(["IPID", "SearchDate", "SearchID"], kind="mergesort")
        if len(ip_history):
            aggregates = ip_history.groupby("IPID", sort=False)["_target"].agg(["sum", "count"])
            count = self._map_single(batch, "IPID", aggregates["count"])
            total = self._map_single(batch, "IPID", aggregates["sum"])
            last = ip_history.groupby("IPID", sort=False).tail(1).set_index("IPID")
            output["ip_label_count"] = np.nan_to_num(count).astype(np.float32)
            output["ip_label_mean"] = np.where(np.isfinite(count), (np.nan_to_num(total) + 20 * prior) / (np.nan_to_num(count) + 20), prior).astype(np.float32)
            output["ip_last_label"] = np.nan_to_num(self._map_single(batch, "IPID", last["_target"]), nan=prior).astype(np.float32)
            last_time = batch["IPID"].map(last["SearchDate"])
            output["ip_last_label_age"] = (batch["SearchDate"] - last_time).dt.total_seconds().to_numpy(dtype=np.float32)
        else:
            output["ip_label_count"] = np.zeros(len(batch), dtype=np.float32)
            output["ip_label_mean"] = np.full(len(batch), prior, dtype=np.float32)
            output["ip_last_label"] = np.full(len(batch), prior, dtype=np.float32)
            output["ip_last_label_age"] = np.full(len(batch), np.nan, dtype=np.float32)
        cohort_columns = [
            "hour_band",
            "CategoryID",
            "LocationID",
            "device_cross",
            "search_parent_category",
            "search_region",
            "category_location_cross",
            "device_category_cross",
            "hour_category_cross",
            "query_present",
        ]
        for column in cohort_columns:
            valid_history = history.loc[history[column].notna()]
            stats = valid_history.groupby(column, sort=False)["_target"].agg(["sum", "count"])
            count = self._map_single(batch, column, stats["count"])
            total = self._map_single(batch, column, stats["sum"])
            output[f"cohort_{column}_count"] = np.nan_to_num(count).astype(np.float32)
            output[f"cohort_{column}_mean"] = np.where(np.isfinite(count), (np.nan_to_num(total) + 80 * prior) / (np.nan_to_num(count) + 80), prior).astype(np.float32)
        conditions = ["hour_band", "device_cross", "CategoryID", "LocationID"]
        for condition in conditions:
            valid_history = user_history.loc[user_history[condition].notna()]
            stats = valid_history.groupby(["UserID", condition], sort=False)["_target"].agg(["sum", "count"])
            count = self._map_multi(batch, ["UserID", condition], stats["count"])
            total = self._map_multi(batch, ["UserID", condition], stats["sum"])
            output[f"user_condition_{condition}_mean"] = np.where(np.isfinite(count), (np.nan_to_num(total) + 20 * prior) / (np.nan_to_num(count) + 20), prior).astype(np.float32)
        if len(user_history):
            gap_history = user_history.loc[user_history["_history_gap_bucket"].notna()]
            stats = gap_history.groupby(["UserID", "_history_gap_bucket"], sort=False)["_target"].agg(["sum", "count"])
            lookup = pd.DataFrame({"UserID": batch["UserID"], "_history_gap_bucket": output["user_gap_bucket"]}, index=batch.index)
            count = self._map_multi(lookup, ["UserID", "_history_gap_bucket"], stats["count"])
            total = self._map_multi(lookup, ["UserID", "_history_gap_bucket"], stats["sum"])
            output["user_condition_gap_mean"] = np.where(np.isfinite(count), (np.nan_to_num(total) + 20 * prior) / (np.nan_to_num(count) + 20), prior).astype(np.float32)
        else:
            output["user_condition_gap_mean"] = np.full(len(batch), prior, dtype=np.float32)
        output["has_user_history"] = (output["user_history_count"].to_numpy() > 0).astype(np.int8)
        output.replace([np.inf, -np.inf], np.nan, inplace=True)
        return output

    def daily_transform(self, rows: pd.DataFrame) -> pd.DataFrame:
        outputs = []
        origins = rows["SearchDate"].dt.floor("D")
        for cutoff in sorted(origins.unique()):
            mask = origins == cutoff
            part = self.transform(rows.loc[mask], pd.Timestamp(cutoff))
            part["_row_id"] = rows.loc[mask, "_row_id"].to_numpy()
            outputs.append(part)
        output = pd.concat(outputs, axis=0).sort_values("_row_id")
        output.drop(columns="_row_id", inplace=True)
        output.index = rows.sort_values("_row_id").index
        return output.loc[rows.index]
