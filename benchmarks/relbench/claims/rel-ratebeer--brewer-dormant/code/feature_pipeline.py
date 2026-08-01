from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


DAY_NS = 86_400_000_000_000
YEAR_DAYS = 365.0
WINDOWS = (30, 90, 183, 365, 730, 1095)
VERSION = "lane0_dense_landmark_v3"


def phase(name: str, start: float) -> float:
    elapsed = time.time() - start
    print(f"[phase] {name}: {elapsed:.1f}s elapsed", flush=True)
    return elapsed


def to_ns(values) -> np.ndarray:
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype(np.int64)


def clean_int(values, fill: int = -1) -> np.ndarray:
    return pd.to_numeric(values, errors="coerce").fillna(fill).to_numpy(dtype=np.int64)


@dataclass
class EventStore:
    brewer_ids: np.ndarray
    times: list[np.ndarray]
    styles: list[np.ndarray]
    categories: list[np.ndarray]
    seasonal: list[np.ndarray]
    one_off: list[np.ndarray]
    alias: list[np.ndarray]
    alcohol: list[np.ndarray]
    ibu: list[np.ndarray]
    by_time_ns: np.ndarray
    by_time_brewer: np.ndarray
    id_to_pos: dict[int, int]

    @classmethod
    def from_context(cls, ctx) -> "EventStore":
        beers = ctx.db.table_dict["beers"].df
        styles = ctx.db.table_dict["beer_styles"].df[["style_id", "category"]]
        frame = beers[["brewer_id", "created_at", "style_id", "is_seasonal", "is_one_off", "is_alias", "alcohol_pct", "ibu"]].copy()
        frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
        frame = frame.dropna(subset=["brewer_id", "created_at"])
        frame = frame.merge(styles, on="style_id", how="left", sort=False)
        frame = frame.sort_values(["brewer_id", "created_at", "style_id"], kind="mergesort")
        brewer_ids = frame["brewer_id"].drop_duplicates().to_numpy(dtype=np.int64)
        id_to_pos = {int(v): i for i, v in enumerate(brewer_ids)}
        times = []
        style_values = []
        categories = []
        seasonal = []
        one_off = []
        alias = []
        alcohol = []
        ibu = []
        for _, group in frame.groupby("brewer_id", sort=False, observed=True):
            times.append(to_ns(group["created_at"]))
            style_values.append(clean_int(group["style_id"]))
            categories.append(clean_int(group["category"]))
            seasonal.append(group["is_seasonal"].fillna(False).to_numpy(dtype=np.float32))
            one_off.append(group["is_one_off"].fillna(False).to_numpy(dtype=np.float32))
            alias.append(group["is_alias"].fillna(False).to_numpy(dtype=np.float32))
            alcohol.append(pd.to_numeric(group["alcohol_pct"], errors="coerce").to_numpy(dtype=np.float32))
            ibu.append(pd.to_numeric(group["ibu"], errors="coerce").to_numpy(dtype=np.float32))
        chron = frame.sort_values("created_at", kind="mergesort")
        return cls(
            brewer_ids=brewer_ids,
            times=times,
            styles=style_values,
            categories=categories,
            seasonal=seasonal,
            one_off=one_off,
            alias=alias,
            alcohol=alcohol,
            ibu=ibu,
            by_time_ns=to_ns(chron["created_at"]),
            by_time_brewer=chron["brewer_id"].to_numpy(dtype=np.int64),
            id_to_pos=id_to_pos,
        )

    def origin(self, timestamp: pd.Timestamp) -> pd.DataFrame:
        t = int(timestamp.value)
        lo = np.searchsorted(self.by_time_ns, t - 365 * DAY_NS, side="right")
        hi = np.searchsorted(self.by_time_ns, t, side="right")
        eligible = np.unique(self.by_time_brewer[lo:hi])
        dormant = np.empty(len(eligible), dtype=np.int8)
        duration = np.empty(len(eligible), dtype=np.float32)
        for j, brewer_id in enumerate(eligible):
            values = self.times[self.id_to_pos[int(brewer_id)]]
            k = np.searchsorted(values, t, side="right")
            if k < len(values) and values[k] <= t + 365 * DAY_NS:
                dormant[j] = 0
                duration[j] = max((values[k] - t) / DAY_NS, 1e-4)
            else:
                dormant[j] = 1
                duration[j] = 365.0
        return pd.DataFrame(
            {
                "timestamp": np.full(len(eligible), timestamp, dtype="datetime64[ns]"),
                "brewer_id": eligible,
                "dormant": dormant,
                "duration": duration,
            }
        )

    def assert_official(self, official: pd.DataFrame) -> None:
        parts = [self.origin(pd.Timestamp(t)) for t in pd.to_datetime(official["timestamp"]).drop_duplicates().sort_values()]
        regenerated = pd.concat(parts, ignore_index=True)[["timestamp", "brewer_id", "dormant"]]
        expected = official[["timestamp", "brewer_id", "dormant"]].copy()
        regenerated = regenerated.sort_values(["timestamp", "brewer_id"]).reset_index(drop=True)
        expected = expected.sort_values(["timestamp", "brewer_id"]).reset_index(drop=True)
        if not regenerated.equals(expected.astype(regenerated.dtypes.to_dict())):
            merged = regenerated.merge(expected, on=["timestamp", "brewer_id"], how="outer", suffixes=("_new", "_old"), indicator=True)
            mismatch = int(((merged["_merge"] != "both") | (merged["dormant_new"] != merged["dormant_old"])).sum())
            raise AssertionError(f"official label regeneration mismatch: {mismatch}")
        print(f"[labels] exact official SQL reproduction: {len(expected)} rows across {expected.timestamp.nunique()} origins", flush=True)


def attach_duration(frame: pd.DataFrame, store: EventStore) -> pd.DataFrame:
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    durations = np.empty(len(result), dtype=np.float32)
    labels = result["dormant"].to_numpy(dtype=np.int8)
    for i, (brewer_id, timestamp) in enumerate(zip(result["brewer_id"].to_numpy(), to_ns(result["timestamp"]))):
        values = store.times[store.id_to_pos[int(brewer_id)]]
        j = np.searchsorted(values, timestamp, side="right")
        if j < len(values) and values[j] <= timestamp + 365 * DAY_NS:
            durations[i] = max((values[j] - timestamp) / DAY_NS, 1e-4)
            if labels[i] != 0:
                raise AssertionError("duration disagrees with official active label")
        else:
            durations[i] = 365.0
            if labels[i] != 1:
                raise AssertionError("duration disagrees with official dormant label")
    result["duration"] = durations
    return result


def deduplicate_episodes(parts: list[pd.DataFrame]) -> pd.DataFrame:
    data = pd.concat(parts, ignore_index=True)
    priority = {"quarterly": 1, "monthly": 2, "official_train": 3, "official_val": 4}
    data["priority"] = data["source"].map(priority).astype(np.int8)
    data = data.sort_values(["timestamp", "brewer_id", "priority"], kind="mergesort")
    data = data.drop_duplicates(["timestamp", "brewer_id"], keep="last")
    return data.drop(columns="priority").sort_values(["timestamp", "brewer_id"], kind="mergesort").reset_index(drop=True)


def build_episode_pools(ctx, store: EventStore, debug: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    official_train = attach_duration(ctx.train.df, store)
    official_train["source"] = "official_train"
    official_train["base_weight"] = 1.0
    if debug:
        origins = pd.to_datetime(["2015-01-01", "2016-01-01", "2017-01-01"])
    else:
        origins = pd.date_range("2003-01-01", "2017-07-01", freq="QS-JAN")
    generated = []
    for timestamp in origins:
        episode = store.origin(timestamp)
        episode["source"] = "quarterly"
        episode["base_weight"] = 0.25
        generated.append(episode)
    pool_a = deduplicate_episodes([official_train] + generated)
    if debug and len(pool_a) > 50_000:
        recent = pool_a.sort_values("timestamp", kind="mergesort").tail(50_000)
        pool_a = recent.sort_values(["timestamp", "brewer_id"], kind="mergesort").reset_index(drop=True)
    official_val = attach_duration(ctx.val.df, store)
    official_val["source"] = "official_val"
    official_val["base_weight"] = 1.0
    if debug:
        recent_origins = pd.to_datetime(["2017-10-01", "2018-06-01", "2018-12-25"])
    else:
        recent_origins = list(pd.date_range("2017-10-01", "2018-12-01", freq="MS")) + [pd.Timestamp("2018-12-25")]
    recent = []
    for timestamp in recent_origins:
        if timestamp + pd.Timedelta(days=365) > pd.Timestamp("2020-01-01"):
            continue
        episode = store.origin(timestamp)
        episode["source"] = "monthly"
        episode["base_weight"] = 0.20
        recent.append(episode)
    pool_b = deduplicate_episodes([pool_a, official_val] + recent)
    print(f"[episodes] model_a={len(pool_a)} model_b={len(pool_b)} a_origins={pool_a.timestamp.nunique()} b_origins={pool_b.timestamp.nunique()}", flush=True)
    return pool_a, pool_b


def immutable_seed(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["timestamp", "brewer_id"]].copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["row_position"] = np.arange(len(result), dtype=np.int64)
    return result


def safe_ratio(numerator, denominator):
    return np.asarray(numerator, dtype=np.float64) / np.maximum(np.asarray(denominator, dtype=np.float64), 1.0)


def numeric_summary(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.nan, np.nan
    return float(finite.mean()), float(finite.std())


def core_features(seeds: pd.DataFrame, store: EventStore, brewer_static: pd.DataFrame) -> pd.DataFrame:
    n = len(seeds)
    brewer = seeds["brewer_id"].to_numpy(dtype=np.int64)
    timestamp = to_ns(seeds["timestamp"])
    values: dict[str, np.ndarray] = {}
    for window in WINDOWS:
        values[f"release_count_{window}"] = np.zeros(n, dtype=np.float32)
        values[f"release_days_{window}"] = np.zeros(n, dtype=np.float32)
        values[f"release_prior_{window}"] = np.zeros(n, dtype=np.float32)
    scalar_names = [
        "release_count_life", "release_days_life", "days_since_release_1", "days_since_release_2", "days_since_release_3",
        "gap16_mean", "gap16_median", "gap16_std", "gap16_max", "gap16_q25", "gap16_q75", "gap16_q90",
        "gaplife_mean", "gaplife_median", "gaplife_std", "gaplife_max", "gaplife_q25", "gaplife_q75", "gaplife_q90",
        "current_over_median", "current_over_q75", "current_over_max", "long_gap_count", "returned_long_gap",
        "release_age_days", "active_year_fraction", "same_month_fraction", "styles_life", "categories_life",
        "styles_365", "categories_365", "new_styles_365", "seasonal_life", "one_off_life", "alias_life",
        "seasonal_365", "one_off_365", "alias_365", "alcohol_life_mean", "alcohol_365_mean", "ibu_life_mean", "ibu_365_mean",
    ]
    for name in scalar_names:
        values[name] = np.full(n, np.nan, dtype=np.float32)
    order = np.argsort(brewer, kind="mergesort")
    sorted_brewer = brewer[order]
    starts = np.r_[0, np.flatnonzero(sorted_brewer[1:] != sorted_brewer[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        brewer_id = int(sorted_brewer[a])
        pos = store.id_to_pos.get(brewer_id)
        if pos is None:
            continue
        event_time = store.times[pos]
        event_style = store.styles[pos]
        event_category = store.categories[pos]
        event_seasonal = store.seasonal[pos]
        event_one_off = store.one_off[pos]
        event_alias = store.alias[pos]
        event_alcohol = store.alcohol[pos]
        event_ibu = store.ibu[pos]
        query = timestamp[rows]
        event_index = np.searchsorted(event_time, query, side="right")
        unique_day, first_index = np.unique(event_time.astype("datetime64[ns]").astype("datetime64[D]").astype(np.int64), return_index=True)
        unique_time = unique_day * DAY_NS
        unique_index = np.searchsorted(unique_time, query, side="right")
        gaps = np.diff(unique_time).astype(np.float64) / DAY_NS
        long_prefix = np.r_[0, np.cumsum(gaps > 365)]
        year = pd.to_datetime(event_time).year.to_numpy()
        year_new = np.r_[True, year[1:] != year[:-1]]
        year_prefix = np.cumsum(year_new)
        month = pd.to_datetime(event_time).month.to_numpy()
        first_style_times = []
        for style in np.unique(event_style):
            first_style_times.append(event_time[np.flatnonzero(event_style == style)[0]])
        first_style_times = np.sort(np.asarray(first_style_times, dtype=np.int64))
        for local, row in enumerate(rows):
            t = query[local]
            k = int(event_index[local])
            u = int(unique_index[local])
            values["release_count_life"][row] = k
            values["release_days_life"][row] = u
            for window in WINDOWS:
                left = np.searchsorted(event_time, t - window * DAY_NS, side="right")
                prior = np.searchsorted(event_time, t - 2 * window * DAY_NS, side="right")
                dleft = np.searchsorted(unique_time, t - window * DAY_NS, side="right")
                values[f"release_count_{window}"][row] = k - left
                values[f"release_days_{window}"][row] = u - dleft
                values[f"release_prior_{window}"][row] = left - prior
            for lag in range(1, 4):
                if k >= lag:
                    values[f"days_since_release_{lag}"][row] = (t - event_time[k - lag]) / DAY_NS
            if u > 0:
                current_gap = (t - unique_time[u - 1]) / DAY_NS
                values["release_age_days"][row] = (t - event_time[0]) / DAY_NS
                values["active_year_fraction"][row] = year_prefix[k - 1] / max(values["release_age_days"][row] / 365.25 + 1.0, 1.0)
                qmonth = pd.Timestamp(t).month
                values["same_month_fraction"][row] = np.mean(month[:k] == qmonth)
            else:
                current_gap = np.nan
            if u > 1:
                history = gaps[: u - 1]
                last = history[-16:]
                for prefix, part in (("gap16", last), ("gaplife", history)):
                    values[f"{prefix}_mean"][row] = np.mean(part)
                    values[f"{prefix}_median"][row] = np.median(part)
                    values[f"{prefix}_std"][row] = np.std(part)
                    values[f"{prefix}_max"][row] = np.max(part)
                    values[f"{prefix}_q25"][row] = np.quantile(part, 0.25)
                    values[f"{prefix}_q75"][row] = np.quantile(part, 0.75)
                    values[f"{prefix}_q90"][row] = np.quantile(part, 0.90)
                values["current_over_median"][row] = current_gap / max(values["gaplife_median"][row], 1.0)
                values["current_over_q75"][row] = current_gap / max(values["gaplife_q75"][row], 1.0)
                values["current_over_max"][row] = current_gap / max(values["gaplife_max"][row], 1.0)
                values["long_gap_count"][row] = long_prefix[u - 1]
                values["returned_long_gap"][row] = float(long_prefix[u - 1] > 0)
            if k > 0:
                recent_left = np.searchsorted(event_time, t - 365 * DAY_NS, side="right")
                recent_slice = slice(recent_left, k)
                values["styles_life"][row] = len(np.unique(event_style[:k]))
                values["categories_life"][row] = len(np.unique(event_category[:k]))
                values["styles_365"][row] = len(np.unique(event_style[recent_slice]))
                values["categories_365"][row] = len(np.unique(event_category[recent_slice]))
                values["new_styles_365"][row] = np.searchsorted(first_style_times, t, side="right") - np.searchsorted(first_style_times, t - 365 * DAY_NS, side="right")
                values["seasonal_life"][row] = np.mean(event_seasonal[:k])
                values["one_off_life"][row] = np.mean(event_one_off[:k])
                values["alias_life"][row] = np.mean(event_alias[:k])
                values["seasonal_365"][row] = np.mean(event_seasonal[recent_slice])
                values["one_off_365"][row] = np.mean(event_one_off[recent_slice])
                values["alias_365"][row] = np.mean(event_alias[recent_slice])
                values["alcohol_life_mean"][row], _ = numeric_summary(event_alcohol[:k])
                values["alcohol_365_mean"][row], _ = numeric_summary(event_alcohol[recent_slice])
                values["ibu_life_mean"][row], _ = numeric_summary(event_ibu[:k])
                values["ibu_365_mean"][row], _ = numeric_summary(event_ibu[recent_slice])
    result = pd.DataFrame(values)
    for window in WINDOWS:
        result[f"release_ratio_{window}"] = safe_ratio(result[f"release_count_{window}"] + 0.5, result[f"release_prior_{window}"] + 0.5).astype(np.float32)
    timestamp_dt = pd.to_datetime(seeds["timestamp"])
    result["origin_year"] = timestamp_dt.dt.year.to_numpy(dtype=np.float32)
    result["origin_month"] = timestamp_dt.dt.month.to_numpy(dtype=np.float32)
    result["origin_month_sin"] = np.sin(2 * np.pi * (result["origin_month"] - 1) / 12).astype(np.float32)
    result["origin_month_cos"] = np.cos(2 * np.pi * (result["origin_month"] - 1) / 12).astype(np.float32)
    static = brewer_static.reindex(brewer)
    for col in ["country_id", "state_id", "type_id"]:
        result[col] = pd.to_numeric(static[col], errors="coerce").fillna(-1).to_numpy(dtype=np.int32)
    for col in ["opened_at_parsed", "created_at_parsed"]:
        date = pd.to_datetime(static[col], errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)
        valid = (date > 0) & (date <= timestamp)
        result[f"brewer_{col}_known"] = valid.astype(np.float32)
        age = np.where(valid, (timestamp - date) / DAY_NS, np.nan)
        result[f"brewer_{col}_age"] = age.astype(np.float32)
    return result


def cache_root() -> Path:
    root = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache")) / VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def register_artifact(path: Path, name: str, description: str) -> None:
    root = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    registry = root / "artifacts.json"
    lock_path = root / "artifacts.lock"
    import fcntl
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            records = []
        relative = str(path.relative_to(root))
        if not any(x.get("path") == relative for x in records):
            records.append({"name": name, "path": relative, "description": description, "content_key": VERSION, "rebuild_hint": "run main.py full to rebuild from the sanitized RelBench cache"})
            temporary = registry.with_suffix(".tmp.lane0")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def demand_tables(ctx, debug: bool) -> dict[str, pd.DataFrame]:
    root = cache_root()
    suffix = "debug" if debug else "full"
    paths = {name: root / f"{name}_{suffix}.parquet" for name in ["rating", "place_country", "place_state", "place_type"]}
    if all(path.exists() for path in paths.values()):
        result = {name: pd.read_parquet(path) for name, path in paths.items()}
        print(f"[cache] loaded demand tables {suffix}", flush=True)
        return result
    ratings = ctx.db.table_dict["beer_ratings"].df[["user_id", "beer_id", "total_score", "created_at"]]
    if debug and len(ratings) > 250_000:
        take = np.linspace(0, len(ratings) - 1, 250_000, dtype=np.int64)
        ratings = ratings.iloc[take]
    beers = ctx.db.table_dict["beers"].df[["beer_id", "brewer_id", "created_at"]]
    users = ctx.db.table_dict["users"].df[["user_id", "created_at"]]
    con = duckdb.connect()
    con.execute("SET threads TO 11")
    con.register("rating_input", ratings)
    con.register("beer_input", beers)
    con.register("user_input", users)
    rating = con.execute(
        """
        SELECT CAST(b.brewer_id AS BIGINT) AS group_key,
               date_trunc('week', r.created_at) + INTERVAL 7 DAY AS bucket_end,
               count(*)::DOUBLE AS event_count,
               approx_count_distinct(r.user_id)::DOUBLE AS distinct_count,
               sum(r.total_score)::DOUBLE AS value_sum,
               sum(r.total_score * r.total_score)::DOUBLE AS value_sq_sum,
               sum(CASE WHEN r.created_at >= b.created_at AND r.created_at <= b.created_at + INTERVAL 365 DAY THEN 1 ELSE 0 END)::DOUBLE AS recent_entity_count,
               sum(CASE WHEN u.created_at <= r.created_at THEN date_diff('day', u.created_at, r.created_at) ELSE NULL END)::DOUBLE AS account_age_sum,
               count(CASE WHEN u.created_at <= r.created_at THEN 1 ELSE NULL END)::DOUBLE AS account_age_count
        FROM rating_input r
        INNER JOIN beer_input b ON r.beer_id = b.beer_id
        LEFT JOIN user_input u ON r.user_id = u.user_id
        WHERE r.created_at IS NOT NULL AND b.brewer_id IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    places = ctx.db.table_dict["places"].df[["place_id", "country_id", "state_id", "type_id"]]
    place_ratings = ctx.db.table_dict["place_ratings"].df[["place_id", "total_score", "created_at"]]
    if debug and len(place_ratings) > 100_000:
        take = np.linspace(0, len(place_ratings) - 1, 100_000, dtype=np.int64)
        place_ratings = place_ratings.iloc[take]
    con.register("place_rating_input", place_ratings)
    con.register("place_input", places)
    result = {"rating": rating}
    for name, column in [("place_country", "country_id"), ("place_state", "state_id"), ("place_type", "type_id")]:
        result[name] = con.execute(
            f"""
            SELECT CAST(p.{column} AS BIGINT) AS group_key,
                   date_trunc('week', r.created_at) + INTERVAL 7 DAY AS bucket_end,
                   count(*)::DOUBLE AS event_count,
                   count(*)::DOUBLE AS distinct_count,
                   sum(r.total_score)::DOUBLE AS value_sum,
                   sum(r.total_score * r.total_score)::DOUBLE AS value_sq_sum,
                   0.0::DOUBLE AS recent_entity_count,
                   0.0::DOUBLE AS account_age_sum,
                   0.0::DOUBLE AS account_age_count
            FROM place_rating_input r
            INNER JOIN place_input p ON r.place_id = p.place_id
            WHERE r.created_at IS NOT NULL AND p.{column} IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 2
            """
        ).df()
    con.close()
    for name, frame in result.items():
        frame.to_parquet(paths[name], index=False)
        if not debug:
            register_artifact(paths[name], f"lane0 {name} weekly buckets", "Leak-free completed-week relational demand aggregates")
    print(f"[cache] built demand tables {suffix}: " + ", ".join(f"{k}={len(v)}" for k, v in result.items()), flush=True)
    return result


def temporal_group_features(seeds: pd.DataFrame, keys: np.ndarray, table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    n = len(seeds)
    timestamp = to_ns(seeds["timestamp"])
    names = []
    for window in WINDOWS:
        names.extend([f"{prefix}_count_{window}", f"{prefix}_distinct_{window}", f"{prefix}_value_mean_{window}", f"{prefix}_value_std_{window}"])
    names.extend([f"{prefix}_recency", f"{prefix}_recent_entity_share_365", f"{prefix}_account_age_365", f"{prefix}_count_ratio_90", f"{prefix}_count_ratio_365", f"{prefix}_reviewer_ratio_90", f"{prefix}_account_age_trend"])
    output = {name: np.full(n, np.nan, dtype=np.float32) for name in names}
    groups = {int(key): group for key, group in table.groupby("group_key", sort=False, observed=True)}
    order = np.argsort(keys, kind="mergesort")
    sorted_key = keys[order]
    starts = np.r_[0, np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1, n]
    columns = ["event_count", "distinct_count", "value_sum", "value_sq_sum", "recent_entity_count", "account_age_sum", "account_age_count"]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        key = int(sorted_key[a])
        group = groups.get(key)
        if group is None:
            continue
        event_time = to_ns(group["bucket_end"])
        raw = {col: group[col].to_numpy(dtype=np.float64) for col in columns}
        cumulative = {col: np.r_[0.0, np.cumsum(raw[col])] for col in columns}
        query = timestamp[rows]
        right = np.searchsorted(event_time, query, side="right")
        for local, row in enumerate(rows):
            t = query[local]
            r = int(right[local])
            if r > 0:
                output[f"{prefix}_recency"][row] = (t - event_time[r - 1]) / DAY_NS
            stats = {}
            for window in WINDOWS:
                left = np.searchsorted(event_time, t - window * DAY_NS, side="right")
                count = cumulative["event_count"][r] - cumulative["event_count"][left]
                distinct = cumulative["distinct_count"][r] - cumulative["distinct_count"][left]
                total = cumulative["value_sum"][r] - cumulative["value_sum"][left]
                square = cumulative["value_sq_sum"][r] - cumulative["value_sq_sum"][left]
                output[f"{prefix}_count_{window}"][row] = count
                output[f"{prefix}_distinct_{window}"][row] = distinct
                if count > 0:
                    mean = total / count
                    output[f"{prefix}_value_mean_{window}"][row] = mean
                    output[f"{prefix}_value_std_{window}"][row] = math.sqrt(max(square / count - mean * mean, 0.0))
                stats[window] = (left, count, distinct)
            left365, count365, _ = stats[365]
            recent_entity = cumulative["recent_entity_count"][r] - cumulative["recent_entity_count"][left365]
            account_age = cumulative["account_age_sum"][r] - cumulative["account_age_sum"][left365]
            account_n = cumulative["account_age_count"][r] - cumulative["account_age_count"][left365]
            output[f"{prefix}_recent_entity_share_365"][row] = recent_entity / max(count365, 1.0)
            output[f"{prefix}_account_age_365"][row] = account_age / max(account_n, 1.0)
            for window in [90, 365]:
                left, count, distinct = stats[window]
                previous = np.searchsorted(event_time, t - 2 * window * DAY_NS, side="right")
                prior_count = cumulative["event_count"][left] - cumulative["event_count"][previous]
                output[f"{prefix}_count_ratio_{window}"][row] = (count + 1.0) / (prior_count + 1.0)
                if window == 90:
                    prior_distinct = cumulative["distinct_count"][left] - cumulative["distinct_count"][previous]
                    output[f"{prefix}_reviewer_ratio_90"][row] = (distinct + 1.0) / (prior_distinct + 1.0)
                    recent_age = cumulative["account_age_sum"][r] - cumulative["account_age_sum"][left]
                    recent_n = cumulative["account_age_count"][r] - cumulative["account_age_count"][left]
                    prior_age = cumulative["account_age_sum"][left] - cumulative["account_age_sum"][previous]
                    prior_n = cumulative["account_age_count"][left] - cumulative["account_age_count"][previous]
                    output[f"{prefix}_account_age_trend"][row] = (recent_age / max(recent_n, 1.0) + 1.0) / (prior_age / max(prior_n, 1.0) + 1.0)
    return pd.DataFrame(output)


def personal_renewal_features(seeds: pd.DataFrame, store: EventStore) -> pd.DataFrame:
    n = len(seeds)
    brewer = seeds["brewer_id"].to_numpy(dtype=np.int64)
    timestamp = to_ns(seeds["timestamp"])
    names = [
        "personal_survival_life", "personal_survival_16", "personal_at_risk_life", "personal_at_risk_16",
        "current_gap_percentile", "gap_last4_mean", "gap_previous4_mean", "gap_last4_ratio", "gap_last8_slope",
        "gap_burst7_fraction", "gap_burst30_fraction", "gap_quiet183_fraction", "release_cluster_count",
        "current_cluster_size", "current_cluster_age", "days_since_non_alias", "days_since_non_one_off",
        "days_since_non_seasonal", "active_months_365", "active_months_730", "release_month_entropy",
    ]
    output = {name: np.full(n, np.nan, dtype=np.float32) for name in names}
    order = np.argsort(brewer, kind="mergesort")
    sorted_brewer = brewer[order]
    starts = np.r_[0, np.flatnonzero(sorted_brewer[1:] != sorted_brewer[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        pos = store.id_to_pos.get(int(sorted_brewer[a]))
        if pos is None:
            continue
        event_time = store.times[pos]
        event_day = event_time.astype("datetime64[ns]").astype("datetime64[D]").astype(np.int64)
        unique_day = np.unique(event_day)
        gaps = np.diff(unique_day).astype(np.float64)
        alias = store.alias[pos]
        one_off = store.one_off[pos]
        seasonal = store.seasonal[pos]
        event_month = pd.to_datetime(event_time).month.to_numpy(dtype=np.int16)
        event_period = event_time.astype("datetime64[ns]").astype("datetime64[M]").astype(np.int64)
        for local, row in enumerate(rows):
            t = timestamp[row]
            seed_day = t // DAY_NS
            k = int(np.searchsorted(event_time, t, side="right"))
            u = int(np.searchsorted(unique_day, seed_day, side="right"))
            if k == 0 or u == 0:
                continue
            current = float(seed_day - unique_day[u - 1])
            history = gaps[: max(u - 1, 0)]
            if len(history):
                at_risk = float(np.sum(history > current))
                future = float(np.sum(history > current + 365.0))
                recent = history[-16:]
                recent_risk = float(np.sum(recent > current))
                recent_future = float(np.sum(recent > current + 365.0))
                output["personal_survival_life"][row] = (future + 1.0) / (at_risk + 3.0)
                output["personal_survival_16"][row] = (recent_future + 1.0) / (recent_risk + 3.0)
                output["personal_at_risk_life"][row] = np.log1p(at_risk)
                output["personal_at_risk_16"][row] = np.log1p(recent_risk)
                output["current_gap_percentile"][row] = np.mean(history <= current)
                last4 = history[-4:]
                previous4 = history[-8:-4]
                output["gap_last4_mean"][row] = np.mean(last4)
                if len(previous4):
                    output["gap_previous4_mean"][row] = np.mean(previous4)
                    output["gap_last4_ratio"][row] = (np.mean(last4) + 1.0) / (np.mean(previous4) + 1.0)
                last8 = history[-8:]
                if len(last8) >= 2:
                    centered = np.arange(len(last8), dtype=np.float64) - (len(last8) - 1) / 2.0
                    output["gap_last8_slope"][row] = np.sum(centered * (last8 - last8.mean())) / max(np.sum(centered * centered), 1.0)
                output["gap_burst7_fraction"][row] = np.mean(history <= 7)
                output["gap_burst30_fraction"][row] = np.mean(history <= 30)
                output["gap_quiet183_fraction"][row] = np.mean(history > 183)
                cluster_breaks = np.flatnonzero(history > 7)
                output["release_cluster_count"][row] = len(cluster_breaks) + 1
                last_break = int(cluster_breaks[-1] + 1) if len(cluster_breaks) else 0
                output["current_cluster_size"][row] = u - last_break
                output["current_cluster_age"][row] = seed_day - unique_day[last_break]
            for name, mask in [("days_since_non_alias", alias[:k] < 0.5), ("days_since_non_one_off", one_off[:k] < 0.5), ("days_since_non_seasonal", seasonal[:k] < 0.5)]:
                candidates = np.flatnonzero(mask)
                if len(candidates):
                    output[name][row] = (t - event_time[candidates[-1]]) / DAY_NS
            left365 = np.searchsorted(event_time, t - 365 * DAY_NS, side="right")
            left730 = np.searchsorted(event_time, t - 730 * DAY_NS, side="right")
            output["active_months_365"][row] = len(np.unique(event_period[left365:k]))
            output["active_months_730"][row] = len(np.unique(event_period[left730:k]))
            counts = np.bincount(event_month[:k], minlength=13)[1:].astype(np.float64)
            probability = counts[counts > 0] / max(counts.sum(), 1.0)
            output["release_month_entropy"][row] = -np.sum(probability * np.log(probability)) / np.log(12.0)
    return pd.DataFrame(output)


def load_exact_daily_counts(ctx) -> tuple[pd.DataFrame, pd.DataFrame]:
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    rating_path = shared / "target_transport_lane1_v4_rating_daily_v2.parquet"
    reviewer_path = shared / "target_transport_lane1_v4_reviewer_first_v2.parquet"
    if rating_path.exists() and reviewer_path.exists():
        print("[cache] reused registered exact daily rating and first-reviewer aggregates", flush=True)
        return pd.read_parquet(rating_path), pd.read_parquet(reviewer_path)
    ratings = ctx.db.table_dict["beer_ratings"].df[["user_id", "beer_id", "created_at"]]
    beers = ctx.db.table_dict["beers"].df[["beer_id", "brewer_id"]]
    con = duckdb.connect()
    con.execute("SET threads TO 11")
    con.register("rating_input", ratings)
    con.register("beer_input", beers)
    joined = con.execute(
        """
        SELECT b.brewer_id::BIGINT AS brewer_id,
               floor(epoch(r.created_at) / 86400)::BIGINT AS day,
               r.user_id::BIGINT AS user_id
        FROM rating_input r
        INNER JOIN beer_input b ON r.beer_id = b.beer_id
        WHERE r.created_at IS NOT NULL AND b.brewer_id IS NOT NULL
        """
    ).df()
    rating = joined.groupby(["brewer_id", "day"], observed=True).size().rename("n").reset_index()
    first = joined.groupby(["brewer_id", "user_id"], observed=True)["day"].min().reset_index()
    reviewer = first.groupby(["brewer_id", "day"], observed=True).size().rename("n").reset_index()
    con.close()
    return rating, reviewer


def daily_count_features(seeds: pd.DataFrame, table: pd.DataFrame, prefix: str) -> pd.DataFrame:
    n = len(seeds)
    brewer = seeds["brewer_id"].to_numpy(dtype=np.int64)
    seed_day = to_ns(seeds["timestamp"]) // DAY_NS
    names = [f"{prefix}_count_{window}" for window in WINDOWS]
    names += [f"{prefix}_count_life", f"{prefix}_recency", f"{prefix}_ratio_90", f"{prefix}_ratio_365"]
    output = {name: np.zeros(n, dtype=np.float32) for name in names}
    output[f"{prefix}_recency"].fill(np.nan)
    groups = {int(key): group for key, group in table.groupby("brewer_id", sort=False, observed=True)}
    order = np.argsort(brewer, kind="mergesort")
    sorted_brewer = brewer[order]
    starts = np.r_[0, np.flatnonzero(sorted_brewer[1:] != sorted_brewer[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        group = groups.get(int(sorted_brewer[a]))
        if group is None:
            continue
        days = group["day"].to_numpy(dtype=np.int64)
        counts = group["n"].to_numpy(dtype=np.float64)
        cumulative = np.r_[0.0, np.cumsum(counts)]
        for row in rows:
            t = seed_day[row]
            right = np.searchsorted(days, t, side="left")
            output[f"{prefix}_count_life"][row] = cumulative[right]
            if right:
                output[f"{prefix}_recency"][row] = t - days[right - 1]
            stats = {}
            for window in WINDOWS:
                left = np.searchsorted(days, t - window, side="right")
                current = cumulative[right] - cumulative[left]
                output[f"{prefix}_count_{window}"][row] = current
                stats[window] = (left, current)
            for window in [90, 365]:
                left, current = stats[window]
                previous = np.searchsorted(days, t - 2 * window, side="right")
                prior = cumulative[left] - cumulative[previous]
                output[f"{prefix}_ratio_{window}"][row] = (current + 1.0) / (prior + 1.0)
    return pd.DataFrame(output)


def widening_features(seeds: pd.DataFrame, store: EventStore, base: pd.DataFrame, ctx) -> pd.DataFrame:
    rating_daily, reviewer_daily = load_exact_daily_counts(ctx)
    personal = personal_renewal_features(seeds, store)
    rating = daily_count_features(seeds, rating_daily, "rating_exact")
    reviewer = daily_count_features(seeds, reviewer_daily, "new_reviewer")
    derived = {}
    for window in WINDOWS:
        release = base[f"release_count_{window}"].to_numpy(dtype=np.float64)
        release_day = base[f"release_days_{window}"].to_numpy(dtype=np.float64)
        rating_count = rating[f"rating_exact_count_{window}"].to_numpy(dtype=np.float64)
        reviewer_count = reviewer[f"new_reviewer_count_{window}"].to_numpy(dtype=np.float64)
        derived[f"rating_per_release_{window}"] = np.log1p(rating_count) / np.maximum(np.log1p(release), 0.25)
        derived[f"rating_per_release_day_{window}"] = np.log1p(rating_count) / np.maximum(np.log1p(release_day), 0.25)
        derived[f"new_reviewer_share_{window}"] = (reviewer_count + 1.0) / (rating_count + 2.0)
    derived["rating_release_recency_delta"] = rating["rating_exact_recency"].to_numpy(dtype=np.float64) - base["days_since_release_1"].to_numpy(dtype=np.float64)
    derived["release_activity_bin"] = np.log2(base["release_count_life"].to_numpy(dtype=np.float64) + 1.0)
    derived["current_gap_year_fraction"] = base["days_since_release_1"].to_numpy(dtype=np.float64) / 365.0
    derived_frame = pd.DataFrame(derived)
    result = pd.concat([personal, rating, reviewer, derived_frame], axis=1)
    return result.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def portfolio_identity_features(seeds: pd.DataFrame, store: EventStore) -> pd.DataFrame:
    n = len(seeds)
    brewer = seeds["brewer_id"].to_numpy(dtype=np.int64)
    timestamp = to_ns(seeds["timestamp"])
    names = ["cx_last_style_id", "cx_last_category_id", "cx_mode_style_id", "cx_mode_category_id", "cx_recent_mode_style_id", "cx_recent_mode_category_id"]
    output = {name: np.full(n, -1, dtype=np.int32) for name in names}
    output.update({name: np.full(n, np.nan, dtype=np.float32) for name in ["cx_style_switch_fraction", "cx_category_switch_fraction", "cx_current_style_age"]})
    order = np.argsort(brewer, kind="mergesort")
    sorted_brewer = brewer[order]
    starts = np.r_[0, np.flatnonzero(sorted_brewer[1:] != sorted_brewer[:-1]) + 1, n]
    for a, b in zip(starts[:-1], starts[1:]):
        rows = order[a:b]
        pos = store.id_to_pos.get(int(sorted_brewer[a]))
        if pos is None:
            continue
        event_time = store.times[pos]
        styles = store.styles[pos]
        categories = store.categories[pos]
        for row in rows:
            t = timestamp[row]
            k = int(np.searchsorted(event_time, t, side="right"))
            if k == 0:
                continue
            left = int(np.searchsorted(event_time, t - 365 * DAY_NS, side="right"))
            last_style = int(styles[k - 1])
            last_category = int(categories[k - 1])
            output["cx_last_style_id"][row] = last_style
            output["cx_last_category_id"][row] = last_category
            for name, array in [("cx_mode_style_id", styles[:k]), ("cx_mode_category_id", categories[:k]), ("cx_recent_mode_style_id", styles[left:k]), ("cx_recent_mode_category_id", categories[left:k])]:
                unique, counts = np.unique(array, return_counts=True)
                output[name][row] = int(unique[np.argmax(counts)]) if len(unique) else -1
            if k > 1:
                output["cx_style_switch_fraction"][row] = np.mean(styles[1:k] != styles[: k - 1])
                output["cx_category_switch_fraction"][row] = np.mean(categories[1:k] != categories[: k - 1])
            first_current = np.flatnonzero(styles[:k] == last_style)
            output["cx_current_style_age"][row] = (t - event_time[first_current[0]]) / DAY_NS
    return pd.DataFrame(output)


def context_activity_tables(ctx, brewer_static: pd.DataFrame) -> dict[str, pd.DataFrame]:
    root = cache_root()
    names = ["country", "state", "type", "style", "category", "global"]
    paths = {name: root / f"context_{name}_v1.parquet" for name in names}
    if all(path.exists() for path in paths.values()):
        return {name: pd.read_parquet(path) for name, path in paths.items()}
    beers = ctx.db.table_dict["beers"].df[["brewer_id", "style_id", "created_at", "alcohol_pct", "is_seasonal", "is_one_off"]].copy()
    styles = ctx.db.table_dict["beer_styles"].df[["style_id", "category"]]
    static = brewer_static[["country_id", "state_id", "type_id"]].reset_index()
    beers = beers.merge(styles, on="style_id", how="left", sort=False).merge(static, on="brewer_id", how="left", sort=False)
    beers["created_at"] = pd.to_datetime(beers["created_at"], errors="coerce")
    beers = beers.dropna(subset=["created_at", "brewer_id"])
    beers["bucket_end"] = beers["created_at"].dt.to_period("W-SUN").dt.start_time + pd.Timedelta(days=7)
    beers["alcohol"] = pd.to_numeric(beers["alcohol_pct"], errors="coerce")
    beers["alcohol_sq"] = beers["alcohol"] * beers["alcohol"]
    beers["seasonal"] = beers["is_seasonal"].fillna(False).astype(np.float64)
    beers["one_off"] = beers["is_one_off"].fillna(False).astype(np.float64)
    result = {}
    for name, column in [("country", "country_id"), ("state", "state_id"), ("type", "type_id"), ("style", "style_id"), ("category", "category")]:
        frame = beers.dropna(subset=[column]).groupby([column, "bucket_end"], observed=True).agg(
            event_count=("brewer_id", "size"),
            distinct_count=("brewer_id", "nunique"),
            value_sum=("alcohol", "sum"),
            value_sq_sum=("alcohol_sq", "sum"),
            recent_entity_count=("seasonal", "sum"),
            account_age_sum=("one_off", "sum"),
            account_age_count=("brewer_id", "size"),
        ).reset_index().rename(columns={column: "group_key"})
        result[name] = frame.sort_values(["group_key", "bucket_end"], kind="mergesort")
    global_frame = beers.assign(group_key=0).groupby(["group_key", "bucket_end"], observed=True).agg(
        event_count=("brewer_id", "size"),
        distinct_count=("brewer_id", "nunique"),
        value_sum=("alcohol", "sum"),
        value_sq_sum=("alcohol_sq", "sum"),
        recent_entity_count=("seasonal", "sum"),
        account_age_sum=("one_off", "sum"),
        account_age_count=("brewer_id", "size"),
    ).reset_index()
    result["global"] = global_frame.sort_values(["group_key", "bucket_end"], kind="mergesort")
    for name, frame in result.items():
        frame.to_parquet(paths[name], index=False)
        register_artifact(paths[name], f"lane0 {name} release context", "Completed-week release activity for contextual renewal features")
    return result


def contextual_features(seeds: pd.DataFrame, store: EventStore, brewer_static: pd.DataFrame, base: pd.DataFrame, ctx) -> pd.DataFrame:
    identity = portfolio_identity_features(seeds, store)
    tables = context_activity_tables(ctx, brewer_static)
    brewer_ids = seeds["brewer_id"].to_numpy(dtype=np.int64)
    static = brewer_static.reindex(brewer_ids)
    keys = {
        "country": pd.to_numeric(static["country_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
        "state": pd.to_numeric(static["state_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
        "type": pd.to_numeric(static["type_id"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64),
        "style": identity["cx_last_style_id"].to_numpy(dtype=np.int64),
        "category": identity["cx_last_category_id"].to_numpy(dtype=np.int64),
        "global": np.zeros(len(seeds), dtype=np.int64),
    }
    blocks = [identity]
    for name in ["country", "state", "type", "style", "category", "global"]:
        full = temporal_group_features(seeds, keys[name], tables[name], f"cx_{name}_release")
        selected = []
        for window in [90, 183, 365, 730]:
            selected.extend([f"cx_{name}_release_count_{window}", f"cx_{name}_release_distinct_{window}"])
        selected.extend([
            f"cx_{name}_release_value_mean_365", f"cx_{name}_release_value_std_365",
            f"cx_{name}_release_recent_entity_share_365", f"cx_{name}_release_account_age_365",
            f"cx_{name}_release_count_ratio_90", f"cx_{name}_release_count_ratio_365",
        ])
        blocks.append(full[selected])
    result = pd.concat(blocks, axis=1)
    for name in ["country", "state", "type", "style", "category", "global"]:
        for window in [90, 365, 730]:
            result[f"cx_relative_to_{name}_{window}"] = np.log1p(base[f"release_count_{window}"].to_numpy(dtype=np.float64)) / np.maximum(np.log1p(result[f"cx_{name}_release_count_{window}"].to_numpy(dtype=np.float64)), 0.25)
    categorical = {"cx_last_style_id", "cx_last_category_id", "cx_mode_style_id", "cx_mode_category_id", "cx_recent_mode_style_id", "cx_recent_mode_category_id"}
    result = result.replace([np.inf, -np.inf], np.nan)
    for column in result.columns:
        result[column] = result[column].fillna(-1).astype(np.int32) if column in categorical else result[column].astype(np.float32)
    return result


def brewer_static_table(ctx) -> pd.DataFrame:
    frame = ctx.db.table_dict["brewers"].df[["brewer_id", "country_id", "state_id", "type_id", "opened_at", "created_at"]].copy()
    frame["opened_at_parsed"] = pd.to_datetime(frame["opened_at"], errors="coerce", format="mixed")
    frame["created_at_parsed"] = pd.to_datetime(frame["created_at"], errors="coerce", format="mixed")
    return frame.set_index("brewer_id", drop=True)


def gap_records(store: EventStore, brewer_static: pd.DataFrame) -> pd.DataFrame:
    records = []
    bins = np.asarray([2, 4, 8, 16, 32, 64, 128])
    for brewer_id, event_time in zip(store.brewer_ids, store.times):
        unique_time = np.unique(event_time.astype("datetime64[ns]").astype("datetime64[D]").astype(np.int64)) * DAY_NS
        if len(unique_time) < 2:
            continue
        gaps = np.diff(unique_time) / DAY_NS
        activity = np.searchsorted(bins, np.arange(2, len(unique_time) + 1), side="right")
        static = brewer_static.loc[int(brewer_id)]
        country = int(static["country_id"]) if pd.notna(static["country_id"]) else -1
        brewer_type = int(static["type_id"]) if pd.notna(static["type_id"]) else -1
        records.append(pd.DataFrame({"gap_end": unique_time[1:], "duration_bin": np.minimum((gaps // 30).astype(np.int16), 319), "country": country, "brewer_type": brewer_type, "activity_bin": activity.astype(np.int8)}))
    return pd.concat(records, ignore_index=True).sort_values("gap_end", kind="mergesort")


def cohort_features(seeds: pd.DataFrame, store: EventStore, brewer_static: pd.DataFrame) -> pd.DataFrame:
    records = gap_records(store, brewer_static)
    countries = clean_int(brewer_static["country_id"])
    country_values = np.unique(np.r_[countries, -1])
    country_map = {int(x): i for i, x in enumerate(country_values)}
    types = clean_int(brewer_static["type_id"])
    type_values = np.unique(np.r_[types, -1])
    type_map = {int(x): i for i, x in enumerate(type_values)}
    nc = len(country_values)
    nt = len(type_values)
    na = 8
    nb = 320
    hist_cta = np.zeros((nc * nt * na, nb), dtype=np.int32)
    hist_ca = np.zeros((nc * na, nb), dtype=np.int32)
    hist_ta = np.zeros((nt * na, nb), dtype=np.int32)
    hist_c = np.zeros((nc, nb), dtype=np.int32)
    hist_g = np.zeros((1, nb), dtype=np.int32)
    record_country = np.asarray([country_map.get(int(x), country_map[-1]) for x in records["country"]], dtype=np.int32)
    record_type = np.asarray([type_map.get(int(x), type_map[-1]) for x in records["brewer_type"]], dtype=np.int32)
    record_activity = records["activity_bin"].to_numpy(dtype=np.int32)
    record_duration = records["duration_bin"].to_numpy(dtype=np.int32)
    record_end = records["gap_end"].to_numpy(dtype=np.int64)
    brewer_ids = seeds["brewer_id"].to_numpy(dtype=np.int64)
    timestamps = to_ns(seeds["timestamp"])
    static = brewer_static.reindex(brewer_ids)
    seed_country = np.asarray([country_map.get(int(x) if pd.notna(x) else -1, country_map[-1]) for x in static["country_id"]], dtype=np.int32)
    seed_type = np.asarray([type_map.get(int(x) if pd.notna(x) else -1, type_map[-1]) for x in static["type_id"]], dtype=np.int32)
    life_count = np.empty(len(seeds), dtype=np.int32)
    current_age = np.empty(len(seeds), dtype=np.float32)
    for i, (brewer_id, timestamp) in enumerate(zip(brewer_ids, timestamps)):
        event_time = store.times[store.id_to_pos[int(brewer_id)]]
        k = np.searchsorted(event_time, timestamp, side="right")
        life_count[i] = k
        current_age[i] = (timestamp - event_time[k - 1]) / DAY_NS if k else 0.0
    activity_bins = np.searchsorted(np.asarray([2, 4, 8, 16, 32, 64, 128]), life_count, side="right").astype(np.int32)
    result = {name: np.full(len(seeds), np.nan, dtype=np.float32) for name in ["cohort_cta", "cohort_ca", "cohort_ta", "cohort_country", "cohort_global", "cohort_selected", "cohort_at_risk"]}
    cursor = 0
    for timestamp in np.unique(timestamps):
        end = np.searchsorted(record_end, timestamp, side="right")
        if end > cursor:
            sl = slice(cursor, end)
            c = record_country[sl]
            ty = record_type[sl]
            ac = record_activity[sl]
            dbin = record_duration[sl]
            np.add.at(hist_cta, (c * nt * na + ty * na + ac, dbin), 1)
            np.add.at(hist_ca, (c * na + ac, dbin), 1)
            np.add.at(hist_ta, (ty * na + ac, dbin), 1)
            np.add.at(hist_c, (c, dbin), 1)
            np.add.at(hist_g, (np.zeros(end - cursor, dtype=np.int32), dbin), 1)
            cursor = end
        rows = np.flatnonzero(timestamps == timestamp)
        age = np.minimum((current_age[rows] // 30).astype(np.int32), nb - 14)
        future = np.minimum(age + 12, nb - 1)
        suffix_cta = np.cumsum(hist_cta[:, ::-1], axis=1)[:, ::-1]
        suffix_ca = np.cumsum(hist_ca[:, ::-1], axis=1)[:, ::-1]
        suffix_ta = np.cumsum(hist_ta[:, ::-1], axis=1)[:, ::-1]
        suffix_c = np.cumsum(hist_c[:, ::-1], axis=1)[:, ::-1]
        suffix_g = np.cumsum(hist_g[:, ::-1], axis=1)[:, ::-1]
        indices = [seed_country[rows] * nt * na + seed_type[rows] * na + activity_bins[rows], seed_country[rows] * na + activity_bins[rows], seed_type[rows] * na + activity_bins[rows], seed_country[rows], np.zeros(len(rows), dtype=np.int32)]
        suffixes = [suffix_cta, suffix_ca, suffix_ta, suffix_c, suffix_g]
        names = ["cohort_cta", "cohort_ca", "cohort_ta", "cohort_country", "cohort_global"]
        denominators = []
        estimates = []
        for name, suffix, index in zip(names, suffixes, indices):
            denominator = suffix[index, age]
            numerator = suffix[index, future]
            estimate = (numerator + 2.0) / (denominator + 4.0)
            result[name][rows] = estimate.astype(np.float32)
            denominators.append(denominator)
            estimates.append(estimate)
        selected = estimates[-1].copy()
        selected_n = denominators[-1].copy()
        for estimate, denominator in reversed(list(zip(estimates[:-1], denominators[:-1]))):
            use = denominator >= 30
            selected[use] = estimate[use]
            selected_n[use] = denominator[use]
        result["cohort_selected"][rows] = selected.astype(np.float32)
        result["cohort_at_risk"][rows] = np.log1p(selected_n).astype(np.float32)
    return pd.DataFrame(result)


def build_features(seeds: pd.DataFrame, store: EventStore, brewer_static: pd.DataFrame, demand: dict[str, pd.DataFrame], include_cohort: bool = True) -> pd.DataFrame:
    start = time.time()
    blocks = [core_features(seeds, store, brewer_static)]
    phase(f"core features {len(seeds)}", start)
    brewer_ids = seeds["brewer_id"].to_numpy(dtype=np.int64)
    blocks.append(temporal_group_features(seeds, brewer_ids, demand["rating"], "rating"))
    static = brewer_static.reindex(brewer_ids)
    for name, column in [("place_country", "country_id"), ("place_state", "state_id"), ("place_type", "type_id")]:
        key = pd.to_numeric(static[column], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        blocks.append(temporal_group_features(seeds, key, demand[name], name))
    phase(f"demand features {len(seeds)}", start)
    if include_cohort:
        blocks.append(cohort_features(seeds, store, brewer_static))
        phase(f"cohort features {len(seeds)}", start)
    frame = pd.concat(blocks, axis=1)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    categorical = {"country_id", "state_id", "type_id", "origin_month"}
    for column in frame.columns:
        if column in categorical:
            frame[column] = frame[column].fillna(-1).astype(np.int32)
        else:
            frame[column] = frame[column].astype(np.float32)
    return frame


def cache_feature_matrix(name: str, seeds: pd.DataFrame, builder) -> pd.DataFrame:
    root = cache_root()
    feature_path = root / f"{name}.parquet"
    key_path = root / f"{name}_keys.parquet"
    keys = seeds[["timestamp", "brewer_id"]].copy()
    if feature_path.exists() and key_path.exists():
        stored = pd.read_parquet(key_path)
        stored["timestamp"] = pd.to_datetime(stored["timestamp"])
        if stored.equals(keys.reset_index(drop=True)):
            print(f"[cache] loaded feature matrix {name}", flush=True)
            return pd.read_parquet(feature_path)
    frame = builder()
    frame.to_parquet(feature_path, index=False)
    keys.to_parquet(key_path, index=False)
    register_artifact(feature_path, f"lane0 {name} feature matrix", "Temporally censored dense-landmark feature matrix")
    return frame
