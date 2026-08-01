from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit


DAY_MS = 86_400_000
BASE_MS = 3_000_000_000_000
RATING_WINDOWS = (1, 3, 7, 14, 30, 60, 90, 180, 365, 730)
FEATURE_VERSION = "dense_origin_wide_v7"


@njit(cache=True)
def decayed_counts(event_users, event_times, query_users, query_times, half_life_days):
    result = np.zeros(len(query_users), dtype=np.float32)
    rate = math.log(2.0) / (half_life_days * DAY_MS)
    i = 0
    while i < len(query_users):
        user = query_users[i]
        j = i + 1
        while j < len(query_users) and query_users[j] == user:
            j += 1
        left = np.searchsorted(event_users, user, side="left")
        right = np.searchsorted(event_users, user, side="right")
        pointer = left
        state = 0.0
        last_time = query_times[i]
        if pointer < right:
            last_time = event_times[pointer]
        for q in range(i, j):
            seed = query_times[q]
            while pointer < right and event_times[pointer] <= seed:
                event_time = event_times[pointer]
                state = state * math.exp(-rate * (event_time - last_time)) + 1.0
                last_time = event_time
                pointer += 1
            if state > 0.0:
                result[q] = state * math.exp(-rate * (seed - last_time))
        i = j
    return result


@njit(cache=True)
def rolling_category_stats(event_users, event_times, categories, query_users, query_times, window_ms, max_category, max_user_events):
    n = len(query_users)
    distinct_out = np.zeros(n, dtype=np.float32)
    entropy_out = np.zeros(n, dtype=np.float32)
    top_share_out = np.zeros(n, dtype=np.float32)
    hhi_out = np.zeros(n, dtype=np.float32)
    top_count_out = np.zeros(n, dtype=np.float32)
    counts = np.zeros(max_category + 1, dtype=np.int32)
    frequencies = np.zeros(max_user_events + 2, dtype=np.int32)
    i = 0
    while i < n:
        user = query_users[i]
        j = i + 1
        while j < n and query_users[j] == user:
            j += 1
        event_left = np.searchsorted(event_users, user, side="left")
        event_right = np.searchsorted(event_users, user, side="right")
        left = event_left
        right = event_left
        distinct = 0
        total = 0
        maximum = 0
        sum_count_log_count = 0.0
        sum_squares = 0.0
        for q in range(i, j):
            seed = query_times[q]
            while right < event_right and event_times[right] <= seed:
                category = categories[right]
                old = counts[category]
                if old > 0:
                    frequencies[old] -= 1
                    sum_count_log_count -= old * math.log(old)
                    sum_squares -= old * old
                else:
                    distinct += 1
                new = old + 1
                counts[category] = new
                frequencies[new] += 1
                sum_count_log_count += new * math.log(new)
                sum_squares += new * new
                if new > maximum:
                    maximum = new
                total += 1
                right += 1
            lower = seed - window_ms
            while left < right and event_times[left] <= lower:
                category = categories[left]
                old = counts[category]
                frequencies[old] -= 1
                sum_count_log_count -= old * math.log(old)
                sum_squares -= old * old
                new = old - 1
                counts[category] = new
                if new > 0:
                    frequencies[new] += 1
                    sum_count_log_count += new * math.log(new)
                    sum_squares += new * new
                else:
                    distinct -= 1
                total -= 1
                if old == maximum and frequencies[old] == 0:
                    while maximum > 0 and frequencies[maximum] == 0:
                        maximum -= 1
                left += 1
            distinct_out[q] = distinct
            top_count_out[q] = maximum
            if total > 0:
                entropy_out[q] = math.log(total) - sum_count_log_count / total
                top_share_out[q] = maximum / total
                hhi_out[q] = sum_squares / (total * total)
        while left < right:
            category = categories[left]
            old = counts[category]
            frequencies[old] -= 1
            new = old - 1
            counts[category] = new
            if new > 0:
                frequencies[new] += 1
            left += 1
        frequencies[0] = 0
        i = j
    return distinct_out, entropy_out, top_share_out, hhi_out, top_count_out


@njit(cache=True)
def first_category_flags(event_users, categories, max_category):
    result = np.zeros(len(event_users), dtype=np.uint8)
    counts = np.zeros(max_category + 1, dtype=np.uint8)
    i = 0
    while i < len(event_users):
        user = event_users[i]
        j = i + 1
        while j < len(event_users) and event_users[j] == user:
            j += 1
        for k in range(i, j):
            category = categories[k]
            if counts[category] == 0:
                result[k] = 1
                counts[category] = 1
        for k in range(i, j):
            counts[categories[k]] = 0
        i = j
    return result


def source_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-ratebeer"


def content_key() -> str:
    raw = "|".join(
        [
            "rel-ratebeer",
            "user-count",
            "T=2020-01-01",
            FEATURE_VERSION,
            "windows=" + ",".join(map(str, RATING_WINDOWS)),
            "beer_ratings_max=2019-12-31T23:59:31.660",
            "place_ratings_max=2019-12-31T23:59:22.817",
            "favorites_max=2019-12-31T23:49:36.687",
            "availability_max=2025-02-03T05:59:42.507",
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def artifact_root(shared_cache: Path) -> Path:
    return shared_cache / f"lane0_{FEATURE_VERSION}_{content_key()}"


def register_artifact(shared_cache: Path, name: str, path: Path, description: str) -> None:
    import fcntl

    registry = shared_cache / "artifacts.json"
    lock_path = shared_cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        relative = str(path.relative_to(shared_cache))
        record = {
            "name": name,
            "path": relative,
            "description": description,
            "content_key": content_key(),
            "rebuild_hint": "Run main.py; cache construction is check-before-compute and atomic.",
        }
        if not any(item.get("name") == name and item.get("content_key") == content_key() for item in entries):
            entries.append(record)
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _save_arrays(directory: Path, arrays: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, values in arrays.items():
        np.save(directory / f"{name}.npy", np.asarray(values))


def _duckdb_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET memory_limit='70GB'")
    return con


def ensure_event_cache(shared_cache: Path) -> Path:
    root = artifact_root(shared_cache)
    ready = root / "events" / "ready.json"
    if ready.exists():
        print(f"[cache] event warehouse hit {root.name}")
        return root / "events"
    started = time.time()
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="events_build_", dir=root))
    db = source_root() / "db"
    con = _duckdb_connection()
    rating_sql = f"""
        SELECT
            br.user_id::INTEGER AS user,
            epoch_ms(br.created_at)::BIGINT AS time,
            coalesce(br.beer_id, 0)::INTEGER AS beer,
            coalesce(b.style_id, 0)::INTEGER AS style,
            coalesce(b.brewer_id, 0)::INTEGER AS brewer,
            coalesce(bw.country_id, 0)::INTEGER AS brewer_country,
            coalesce(b.alcohol_pct, 0)::FLOAT AS alcohol,
            coalesce(br.total_score, 0)::FLOAT AS score,
            CASE WHEN br.comments IS NULL OR length(trim(br.comments)) = 0 THEN 0 ELSE 1 END::UTINYINT AS comment_present,
            least(coalesce(length(br.comments), 0), 10000)::FLOAT AS comment_length,
            CASE lower(coalesce(br.language, 'null'))
                WHEN 'en' THEN 1 WHEN 'pl' THEN 2 WHEN 'fr' THEN 3 WHEN 'de' THEN 4
                WHEN 'nl' THEN 5 WHEN 'sv' THEN 6 WHEN 'it' THEN 7 WHEN 'es' THEN 8
                WHEN 'no' THEN 9 WHEN 'da' THEN 10 WHEN 'pt' THEN 11 WHEN 'hu' THEN 12
                WHEN 'sk' THEN 13 WHEN 'cs' THEN 14 WHEN 'null' THEN 15 ELSE 16
            END::UTINYINT AS language
        FROM read_parquet('{db / 'beer_ratings.parquet'}') br
        LEFT JOIN read_parquet('{db / 'beers.parquet'}') b ON br.beer_id = b.beer_id
        LEFT JOIN read_parquet('{db / 'brewers.parquet'}') bw ON b.brewer_id = bw.brewer_id
        ORDER BY br.user_id, br.created_at, br.rating_id
    """
    print("[cache] building sorted beer-rating event warehouse")
    rating_arrays = con.execute(rating_sql).fetchnumpy()
    _save_arrays(temporary / "rating", rating_arrays)
    del rating_arrays
    gc.collect()
    place_sql = f"""
        SELECT
            pr.user_id::INTEGER AS user,
            epoch_ms(pr.created_at)::BIGINT AS time,
            coalesce(pr.place_id, 0)::INTEGER AS place,
            coalesce(p.type_id, 0)::INTEGER AS place_type,
            coalesce(p.country_id, 0)::INTEGER AS country,
            coalesce(p.state_id, 0)::INTEGER AS state,
            coalesce(pr.total_score, 0)::FLOAT AS score
        FROM read_parquet('{db / 'place_ratings.parquet'}') pr
        LEFT JOIN read_parquet('{db / 'places.parquet'}') p ON pr.place_id = p.place_id
        ORDER BY pr.user_id, pr.created_at, pr.rating_id
    """
    print("[cache] building sorted place-rating event warehouse")
    place_arrays = con.execute(place_sql).fetchnumpy()
    _save_arrays(temporary / "place", place_arrays)
    del place_arrays
    favorite_sql = f"""
        SELECT
            f.user_id::INTEGER AS user,
            epoch_ms(f.created_at)::BIGINT AS time,
            coalesce(f.beer_id, 0)::INTEGER AS beer,
            coalesce(b.style_id, 0)::INTEGER AS style,
            coalesce(b.brewer_id, 0)::INTEGER AS brewer
        FROM read_parquet('{db / 'favorites.parquet'}') f
        LEFT JOIN read_parquet('{db / 'beers.parquet'}') b ON f.beer_id = b.beer_id
        ORDER BY f.user_id, f.created_at, f.favorite_id
    """
    print("[cache] building sorted favorite event warehouse")
    favorite_arrays = con.execute(favorite_sql).fetchnumpy()
    _save_arrays(temporary / "favorite", favorite_arrays)
    del favorite_arrays
    con.close()
    final = root / "events"
    if final.exists():
        shutil.rmtree(temporary)
    else:
        os.replace(temporary, final)
    ready.write_text(json.dumps({"content_key": content_key(), "seconds": time.time() - started}, indent=2))
    register_artifact(shared_cache, f"lane0-event-warehouse-{FEATURE_VERSION}", final, "Sorted, dimension-enriched, temporally indexed beer/place/favorite events.")
    print(f"[cache] event warehouse complete in {time.time() - started:.1f}s")
    return final


def ensure_episode_cache(shared_cache: Path) -> Path:
    root = artifact_root(shared_cache)
    final = root / "dense_episodes.parquet"
    if final.exists():
        print(f"[cache] dense episodes hit {final.name}")
        return final
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"dense_episodes_{os.getpid()}.parquet"
    ratings = source_root() / "db" / "beer_ratings.parquet"
    con = _duckdb_connection()
    query = f"""
        COPY (
            WITH origins AS (
                SELECT unnest(generate_series(TIMESTAMP '2012-01-01', TIMESTAMP '2019-10-01', INTERVAL '1 month')) AS timestamp
                UNION ALL SELECT TIMESTAMP '2019-10-03'
            ), eligible AS (
                SELECT o.timestamp, br.user_id
                FROM origins o
                JOIN read_parquet('{ratings}') br
                  ON br.created_at > o.timestamp - INTERVAL '90 days'
                 AND br.created_at <= o.timestamp
                GROUP BY o.timestamp, br.user_id
            ), labeled AS (
                SELECT e.timestamp, e.user_id, count(br.rating_id)::INTEGER AS num_ratings
                FROM eligible e
                LEFT JOIN read_parquet('{ratings}') br
                  ON br.user_id = e.user_id
                 AND br.created_at > e.timestamp
                 AND br.created_at <= e.timestamp + INTERVAL '90 days'
                GROUP BY e.timestamp, e.user_id
            )
            SELECT * FROM labeled ORDER BY timestamp, user_id
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    started = time.time()
    con.execute(query)
    con.close()
    if final.exists():
        temporary.unlink(missing_ok=True)
    else:
        os.replace(temporary, final)
    register_artifact(shared_cache, f"lane0-dense-episodes-{FEATURE_VERSION}", final, "Exact monthly eligible 90-day labels through 2019-10-03.")
    print(f"[cache] dense episodes complete in {time.time() - started:.1f}s")
    return final


def load_event_arrays(event_root: Path, group: str) -> dict[str, np.ndarray]:
    result = {}
    for path in sorted((event_root / group).glob("*.npy")):
        result[path.stem] = np.load(path, mmap_mode="r")
    return result


def _prefix_window(values: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    prefix = np.empty(len(values) + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(values, dtype=np.float64, out=prefix[1:])
    result = (prefix[right] - prefix[left]).astype(np.float32)
    del prefix
    return result


def _site_features(seed_times: np.ndarray) -> pd.DataFrame:
    unique_times = pd.DataFrame({"timestamp": pd.to_datetime(np.unique(seed_times.astype("datetime64[ms]")))})
    ratings = source_root() / "db" / "beer_ratings.parquet"
    con = _duckdb_connection()
    con.register("origins", unique_times)
    query = f"""
        SELECT
            o.timestamp,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '30 days')::DOUBLE AS site_count_30,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '60 days')::DOUBLE AS site_count_60,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '90 days')::DOUBLE AS site_count_90,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '180 days')::DOUBLE AS site_count_180,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '365 days')::DOUBLE AS site_count_365,
            count(*) FILTER (WHERE br.created_at <= o.timestamp - INTERVAL '365 days')::DOUBLE AS site_count_prev365,
            count(*) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '455 days' AND br.created_at <= o.timestamp - INTERVAL '365 days')::DOUBLE AS site_count_year_match90,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '30 days')::DOUBLE AS site_users_30,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '60 days')::DOUBLE AS site_users_60,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '90 days')::DOUBLE AS site_users_90,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '180 days')::DOUBLE AS site_users_180,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '365 days')::DOUBLE AS site_users_365,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at <= o.timestamp - INTERVAL '365 days')::DOUBLE AS site_users_prev365,
            count(DISTINCT br.user_id) FILTER (WHERE br.created_at > o.timestamp - INTERVAL '455 days' AND br.created_at <= o.timestamp - INTERVAL '365 days')::DOUBLE AS site_users_year_match90
        FROM origins o
        LEFT JOIN read_parquet('{ratings}') br
          ON br.created_at > o.timestamp - INTERVAL '730 days'
         AND br.created_at <= o.timestamp
        GROUP BY o.timestamp
        ORDER BY o.timestamp
    """
    frame = con.execute(query).df()
    con.close()
    return frame


class FeatureCollector:
    def __init__(self):
        self.values: list[np.ndarray] = []
        self.names: list[str] = []
        self.groups: list[str] = []

    def add(self, name: str, values: np.ndarray, group: str) -> None:
        array = np.asarray(values, dtype=np.float32)
        array[~np.isfinite(array)] = 0.0
        self.values.append(array)
        self.names.append(name)
        self.groups.append(group)

    def matrix(self) -> np.ndarray:
        return np.ascontiguousarray(np.column_stack(self.values), dtype=np.float32)


def _category_features(collector: FeatureCollector, prefix: str, group: str, event_users: np.ndarray, event_times: np.ndarray, categories: np.ndarray, query_users: np.ndarray, query_times: np.ndarray, windows: tuple[int, ...], include_distribution: bool = True) -> None:
    max_category = int(np.max(categories)) if len(categories) else 0
    _, per_user = np.unique(event_users, return_counts=True)
    max_user_events = int(per_user.max()) if len(per_user) else 1
    for days in windows:
        distinct, entropy, top_share, hhi, top_count = rolling_category_stats(
            event_users,
            event_times,
            categories,
            query_users,
            query_times,
            days * DAY_MS,
            max_category,
            max_user_events,
        )
        collector.add(f"{prefix}_unique_{days}", distinct, group)
        if include_distribution:
            collector.add(f"{prefix}_entropy_{days}", entropy, group)
            collector.add(f"{prefix}_top_share_{days}", top_share, group)
            collector.add(f"{prefix}_hhi_{days}", hhi, group)
        collector.add(f"{prefix}_max_count_{days}", top_count, group)


def _generic_activity_features(collector: FeatureCollector, prefix: str, group: str, events: dict[str, np.ndarray], query_users: np.ndarray, query_times: np.ndarray, score_col: str | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    event_users = np.asarray(events["user"], dtype=np.int32)
    event_times = np.asarray(events["time"], dtype=np.int64)
    event_keys = event_users.astype(np.int64) * BASE_MS + event_times
    query_keys = query_users.astype(np.int64) * BASE_MS + query_times
    right = np.searchsorted(event_keys, query_keys, side="right")
    starts = np.searchsorted(event_users, query_users, side="left")
    window_counts = {}
    for days in (30, 90, 180, 365):
        left = np.searchsorted(event_keys, query_keys - days * DAY_MS, side="right")
        values = (right - left).astype(np.float32)
        window_counts[days] = values
        collector.add(f"{prefix}_count_{days}", values, group)
        collector.add(f"{prefix}_log_count_{days}", np.log1p(values), group)
        if score_col is not None and days in (90, 365):
            score_sum = _prefix_window(np.asarray(events[score_col]), left, right)
            collector.add(f"{prefix}_score_mean_{days}", score_sum / np.maximum(values, 1.0), group)
    collector.add(f"{prefix}_count_prev90", window_counts[180] - window_counts[90], group)
    collector.add(f"{prefix}_trend_90", window_counts[90] - (window_counts[180] - window_counts[90]), group)
    collector.add(f"{prefix}_ratio_90", (window_counts[90] + 1.0) / (window_counts[180] - window_counts[90] + 1.0), group)
    lifetime = (right - starts).astype(np.float32)
    collector.add(f"{prefix}_lifetime", lifetime, group)
    recency = np.full(len(query_users), 7300.0, dtype=np.float32)
    valid = right > starts
    recency[valid] = (query_times[valid] - event_times[right[valid] - 1]) / DAY_MS
    collector.add(f"{prefix}_recency", recency, group)
    decay = decayed_counts(event_users, event_times, query_users, query_times, 90.0)
    collector.add(f"{prefix}_decay_90", decay, group)
    return event_keys, right, starts


def build_features(seed_frame: pd.DataFrame, event_root: Path, wide: bool = True) -> tuple[np.ndarray, list[str], list[str]]:
    started = time.time()
    original_users = seed_frame["user_id"].to_numpy(dtype=np.int32)
    original_times = seed_frame["timestamp"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    order = np.lexsort((original_times, original_users))
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order))
    query_users = original_users[order]
    query_times = original_times[order]
    collector = FeatureCollector()
    rating = load_event_arrays(event_root, "rating")
    event_users = np.asarray(rating["user"], dtype=np.int32)
    event_times = np.asarray(rating["time"], dtype=np.int64)
    event_keys = event_users.astype(np.int64) * BASE_MS + event_times
    query_keys = query_users.astype(np.int64) * BASE_MS + query_times
    right = np.searchsorted(event_keys, query_keys, side="right")
    starts = np.searchsorted(event_users, query_users, side="left")
    counts = {}
    left_indices = {}
    for days in RATING_WINDOWS:
        left = np.searchsorted(event_keys, query_keys - days * DAY_MS, side="right")
        values = (right - left).astype(np.float32)
        counts[days] = values
        left_indices[days] = left
        collector.add(f"rating_count_{days}", values, "core_rating")
        collector.add(f"rating_log_count_{days}", np.log1p(values), "core_rating")
    for days in (2, 6, 9, 21, 28, 42, 120, 270, 360, 450, 455, 540, 630, 720, 820):
        left = np.searchsorted(event_keys, query_keys - days * DAY_MS, side="right")
        counts[days] = (right - left).astype(np.float32)
    block_0 = counts[90]
    block_1 = counts[180] - counts[90]
    block_2 = counts[270] - counts[180]
    block_3 = counts[360] - counts[270]
    year_match_1 = counts[455] - counts[365]
    year_match_2 = counts[820] - counts[730]
    for index, values in enumerate((block_0, block_1, block_2, block_3)):
        collector.add(f"rating_block90_{index}", values, "core_rating")
    collector.add("rating_year_match_1", year_match_1, "core_rating")
    collector.add("rating_year_match_2", year_match_2, "core_rating")
    collector.add("rating_diff_90", block_0 - block_1, "core_rating")
    collector.add("rating_acceleration_90", block_0 - 2.0 * block_1 + block_2, "core_rating")
    collector.add("rating_ratio_90", (block_0 + 1.0) / (block_1 + 1.0), "core_rating")
    collector.add("rating_ratio_year", (block_0 + 1.0) / (year_match_1 + 1.0), "core_rating")
    collector.add("rating_diff_30", counts[30] - (counts[60] - counts[30]), "core_rating")
    for days in (1, 3, 7, 14, 30, 60, 90):
        current = counts[days]
        previous = counts[2 * days] - counts[days]
        previous_two = counts[3 * days] - counts[2 * days]
        collector.add(f"rating_multiscale_diff_{days}", current - previous, "user_trajectory_addition")
        collector.add(f"rating_multiscale_ratio_{days}", (current + 1.0) / (previous + 1.0), "user_trajectory_addition")
        collector.add(f"rating_multiscale_acceleration_{days}", current - 2.0 * previous + previous_two, "user_trajectory_addition")
        collector.add(f"rating_multiscale_rate_{days}", current / days, "user_trajectory_addition")
    cumulative_blocks = [counts[90], counts[180], counts[270], counts[360], counts[450], counts[540], counts[630], counts[720]]
    blocks = [cumulative_blocks[0]] + [cumulative_blocks[index] - cumulative_blocks[index - 1] for index in range(1, len(cumulative_blocks))]
    for index in range(4, 8):
        collector.add(f"rating_block90_{index}", blocks[index], "user_trajectory_addition")
    block_matrix = np.column_stack(blocks).astype(np.float32)
    block_mean = block_matrix.mean(axis=1)
    collector.add("rating_block90_mean8", block_mean, "user_trajectory_addition")
    collector.add("rating_block90_std8", block_matrix.std(axis=1), "user_trajectory_addition")
    collector.add("rating_block90_max8", block_matrix.max(axis=1), "user_trajectory_addition")
    collector.add("rating_block90_active_share8", (block_matrix > 0).mean(axis=1), "user_trajectory_addition")
    collector.add("rating_block90_current_vs_mean8", (blocks[0] + 1.0) / (block_mean + 1.0), "user_trajectory_addition")
    centered_positions = np.arange(8, dtype=np.float32) - 3.5
    collector.add("rating_block90_slope8", block_matrix @ centered_positions / float(np.sum(centered_positions ** 2)), "user_trajectory_addition")
    del block_matrix
    lifetime = (right - starts).astype(np.float32)
    collector.add("rating_lifetime", lifetime, "core_rating")
    recency = np.zeros(len(query_users), dtype=np.float32)
    valid = right > starts
    recency[valid] = (query_times[valid] - event_times[right[valid] - 1]) / DAY_MS
    collector.add("rating_recency", recency, "core_rating")
    tenure = np.zeros(len(query_users), dtype=np.float32)
    tenure[valid] = (query_times[valid] - event_times[starts[valid]]) / DAY_MS
    collector.add("rating_tenure", tenure, "core_rating")
    gap_sum = np.zeros(len(query_users), dtype=np.float64)
    gap_sq = np.zeros(len(query_users), dtype=np.float64)
    gap_min = np.full(len(query_users), np.inf, dtype=np.float64)
    gap_max = np.zeros(len(query_users), dtype=np.float64)
    gap_count = np.zeros(len(query_users), dtype=np.float32)
    last_gap = np.zeros(len(query_users), dtype=np.float32)
    for offset in range(1, 6):
        index = right - offset
        ok = index - 1 >= starts
        gap = np.zeros(len(query_users), dtype=np.float64)
        gap[ok] = (event_times[index[ok]] - event_times[index[ok] - 1]) / DAY_MS
        if offset == 1:
            last_gap[ok] = gap[ok]
        gap_sum[ok] += gap[ok]
        gap_sq[ok] += gap[ok] * gap[ok]
        gap_min[ok] = np.minimum(gap_min[ok], gap[ok])
        gap_max[ok] = np.maximum(gap_max[ok], gap[ok])
        gap_count[ok] += 1.0
    gap_mean = gap_sum / np.maximum(gap_count, 1.0)
    gap_std = np.sqrt(np.maximum(gap_sq / np.maximum(gap_count, 1.0) - gap_mean * gap_mean, 0.0))
    gap_min[~np.isfinite(gap_min)] = 0.0
    collector.add("rating_last_gap", last_gap, "core_cadence")
    collector.add("rating_gap_mean5", gap_mean, "core_cadence")
    collector.add("rating_gap_std5", gap_std, "core_cadence")
    collector.add("rating_gap_min5", gap_min, "core_cadence")
    collector.add("rating_gap_max5", gap_max, "core_cadence")
    weekly_total = np.zeros(len(query_users), dtype=np.float32)
    weekly_square_total = np.zeros(len(query_users), dtype=np.float32)
    weekly_weighted_total = np.zeros(len(query_users), dtype=np.float32)
    weekly_active = np.zeros(len(query_users), dtype=np.float32)
    weekly_maximum = np.zeros(len(query_users), dtype=np.float32)
    previous_cumulative = np.zeros(len(query_users), dtype=np.float32)
    first_four = np.zeros(len(query_users), dtype=np.float32)
    for week_index in range(52):
        weekly_left = np.searchsorted(event_keys, query_keys - (week_index + 1) * 7 * DAY_MS, side="right")
        cumulative = (right - weekly_left).astype(np.float32)
        weekly_count = cumulative - previous_cumulative
        previous_cumulative = cumulative
        weekly_total += weekly_count
        weekly_square_total += weekly_count * weekly_count
        weekly_weighted_total += week_index * weekly_count
        weekly_active += weekly_count > 0
        weekly_maximum = np.maximum(weekly_maximum, weekly_count)
        if week_index + 1 in (1, 2, 4, 8, 13, 26, 52):
            collector.add(f"rating_week_lag_{week_index + 1}", weekly_count, "weekly_trajectory_addition")
        if week_index == 3:
            first_four = weekly_total.copy()
        if week_index == 7:
            prior_four = weekly_total - first_four
            collector.add("rating_week_recent4_diff", first_four - prior_four, "weekly_trajectory_addition")
            collector.add("rating_week_recent4_ratio", (first_four + 1.0) / (prior_four + 1.0), "weekly_trajectory_addition")
        horizon = week_index + 1
        if horizon in (13, 26, 52):
            weekly_mean = weekly_total / horizon
            position_sum = horizon * (horizon - 1) / 2.0
            position_square_sum = horizon * (horizon - 1) * (2 * horizon - 1) / 6.0
            denominator = horizon * position_square_sum - position_sum * position_sum
            slope = (horizon * weekly_weighted_total - position_sum * weekly_total) / max(denominator, 1.0)
            collector.add(f"rating_week_mean_{horizon}", weekly_mean, "weekly_trajectory_addition")
            collector.add(f"rating_week_std_{horizon}", np.sqrt(np.maximum(weekly_square_total / horizon - weekly_mean * weekly_mean, 0.0)), "weekly_trajectory_addition")
            collector.add(f"rating_week_active_share_{horizon}", weekly_active / horizon, "weekly_trajectory_addition")
            collector.add(f"rating_week_max_{horizon}", weekly_maximum, "weekly_trajectory_addition")
            collector.add(f"rating_week_slope_{horizon}", slope, "weekly_trajectory_addition")
            collector.add(f"rating_week_current_vs_mean_{horizon}", (counts[7] + 1.0) / (weekly_mean + 1.0), "weekly_trajectory_addition")
    del weekly_total, weekly_square_total, weekly_weighted_total, weekly_active, weekly_maximum, previous_cumulative, first_four
    for half_life in (7.0, 30.0, 90.0, 180.0):
        collector.add(f"rating_decay_{int(half_life)}", decayed_counts(event_users, event_times, query_users, query_times, half_life), "core_cadence")
    for days in (30, 90, 365):
        left = left_indices[days]
        count = counts[days]
        score_sum = _prefix_window(np.asarray(rating["score"]), left, right)
        score_sq = _prefix_window(np.asarray(rating["score"], dtype=np.float32) ** 2, left, right)
        mean = score_sum / np.maximum(count, 1.0)
        std = np.sqrt(np.maximum(score_sq / np.maximum(count, 1.0) - mean * mean, 0.0))
        comment_count = _prefix_window(np.asarray(rating["comment_present"]), left, right)
        comment_length = _prefix_window(np.asarray(rating["comment_length"]), left, right)
        collector.add(f"rating_score_mean_{days}", mean, "core_content")
        collector.add(f"rating_score_std_{days}", std, "core_content")
        collector.add(f"rating_comment_rate_{days}", comment_count / np.maximum(count, 1.0), "core_content")
        collector.add(f"rating_comment_length_mean_{days}", comment_length / np.maximum(count, 1.0), "core_content")
    day_categories = (event_times // DAY_MS).astype(np.int32)
    week_categories = (event_times // (7 * DAY_MS)).astype(np.int32)
    for days in (30, 90, 365):
        distinct, entropy, top_share, hhi, maximum = rolling_category_stats(event_users, event_times, day_categories, query_users, query_times, days * DAY_MS, int(day_categories.max()), int(np.diff(np.r_[np.flatnonzero(np.r_[True, event_users[1:] != event_users[:-1]]), len(event_users)]).max()))
        collector.add(f"rating_active_days_{days}", distinct, "core_cadence")
        collector.add(f"rating_daily_max_{days}", maximum, "core_cadence")
        collector.add(f"rating_daily_entropy_{days}", entropy, "cadence_distribution_addition")
        collector.add(f"rating_daily_top_share_{days}", top_share, "cadence_distribution_addition")
        collector.add(f"rating_daily_hhi_{days}", hhi, "cadence_distribution_addition")
        daily_mean = counts[days] / np.maximum(distinct, 1.0)
        collector.add(f"rating_daily_count_std_{days}", np.sqrt(np.maximum(hhi * counts[days] * counts[days] / np.maximum(distinct, 1.0) - daily_mean * daily_mean, 0.0)), "cadence_distribution_addition")
    for days in (90, 365):
        distinct, entropy, top_share, hhi, maximum = rolling_category_stats(event_users, event_times, week_categories, query_users, query_times, days * DAY_MS, int(week_categories.max()), int(np.diff(np.r_[np.flatnonzero(np.r_[True, event_users[1:] != event_users[:-1]]), len(event_users)]).max()))
        collector.add(f"rating_active_weeks_{days}", distinct, "core_cadence")
        collector.add(f"rating_weekly_max_{days}", maximum, "core_cadence")
        collector.add(f"rating_weekly_entropy_{days}", entropy, "cadence_distribution_addition")
        collector.add(f"rating_weekly_top_share_{days}", top_share, "cadence_distribution_addition")
        collector.add(f"rating_weekly_hhi_{days}", hhi, "cadence_distribution_addition")
        weekly_mean = counts[days] / np.maximum(distinct, 1.0)
        collector.add(f"rating_weekly_count_std_{days}", np.sqrt(np.maximum(hhi * counts[days] * counts[days] / np.maximum(distinct, 1.0) - weekly_mean * weekly_mean, 0.0)), "cadence_distribution_addition")
    _category_features(collector, "rating_language", "core_content", event_users, event_times, np.asarray(rating["language"], dtype=np.int32), query_users, query_times, (90, 365))
    place = load_event_arrays(event_root, "place")
    place_keys, place_right, _ = _generic_activity_features(collector, "place_rating", "core_aux", place, query_users, query_times, "score")
    favorite = load_event_arrays(event_root, "favorite")
    _generic_activity_features(collector, "favorite", "core_aux", favorite, query_users, query_times, None)
    collector.add("availability_count_90", np.zeros(len(query_users)), "core_aux")
    collector.add("availability_count_365", np.zeros(len(query_users)), "core_aux")
    site = _site_features(query_times.astype("datetime64[ms]"))
    site_index = pd.Index(site["timestamp"].to_numpy(dtype="datetime64[ms]").astype(np.int64))
    locations = site_index.get_indexer(query_times)
    site_columns = [column for column in site.columns if column != "timestamp"]
    site_values = {}
    drift_site_columns = {"site_count_60", "site_count_180", "site_count_year_match90", "site_users_60", "site_users_180", "site_users_year_match90"}
    for column in site_columns:
        values = site[column].to_numpy(dtype=np.float32)[locations]
        site_values[column] = values
        feature_group = "site_drift_addition" if column in drift_site_columns else "core_platform"
        collector.add(column, values, feature_group)
        collector.add(f"log_{column}", np.log1p(values), feature_group)
    collector.add("site_count_yoy_ratio", (site_values["site_count_365"] + 1.0) / (site_values["site_count_prev365"] + 1.0), "core_platform")
    collector.add("site_users_yoy_ratio", (site_values["site_users_365"] + 1.0) / (site_values["site_users_prev365"] + 1.0), "core_platform")
    collector.add("user_site_share_90", counts[90] / np.maximum(site_values["site_count_90"], 1.0), "core_platform")
    collector.add("eligible_cohort_size", site_values["site_users_90"], "core_platform")
    rank_frame = pd.DataFrame({"timestamp": query_times, "value": counts[90]})
    percentile = rank_frame.groupby("timestamp", sort=False)["value"].rank(method="average", pct=True).to_numpy(dtype=np.float32)
    collector.add("cohort_activity_percentile", percentile, "core_platform")
    site_prev30 = site_values["site_count_60"] - site_values["site_count_30"]
    site_prev90 = site_values["site_count_180"] - site_values["site_count_90"]
    users_prev30 = site_values["site_users_60"] - site_values["site_users_30"]
    users_prev90 = site_values["site_users_180"] - site_values["site_users_90"]
    collector.add("site_count_diff_30", site_values["site_count_30"] - site_prev30, "site_drift_addition")
    collector.add("site_count_ratio_30", (site_values["site_count_30"] + 1.0) / (site_prev30 + 1.0), "site_drift_addition")
    collector.add("site_count_diff_90", site_values["site_count_90"] - site_prev90, "site_drift_addition")
    collector.add("site_count_ratio_90", (site_values["site_count_90"] + 1.0) / (site_prev90 + 1.0), "site_drift_addition")
    collector.add("site_count_acceleration_90", site_values["site_count_90"] - 2.0 * site_prev90 + site_values["site_count_year_match90"], "site_drift_addition")
    collector.add("site_count_year_match_ratio90", (site_values["site_count_90"] + 1.0) / (site_values["site_count_year_match90"] + 1.0), "site_drift_addition")
    collector.add("site_users_diff_30", site_values["site_users_30"] - users_prev30, "site_drift_addition")
    collector.add("site_users_ratio_30", (site_values["site_users_30"] + 1.0) / (users_prev30 + 1.0), "site_drift_addition")
    collector.add("site_users_diff_90", site_values["site_users_90"] - users_prev90, "site_drift_addition")
    collector.add("site_users_ratio_90", (site_values["site_users_90"] + 1.0) / (users_prev90 + 1.0), "site_drift_addition")
    collector.add("site_users_year_match_ratio90", (site_values["site_users_90"] + 1.0) / (site_values["site_users_year_match90"] + 1.0), "site_drift_addition")
    for days in (30, 90, 365):
        site_mean = site_values[f"site_count_{days}"] / np.maximum(site_values[f"site_users_{days}"], 1.0)
        collector.add(f"site_intensity_per_user_{days}", site_mean, "site_drift_addition")
        collector.add(f"user_intensity_vs_site_{days}", counts[days] / np.maximum(site_mean, 0.01), "site_drift_addition")
        rank_values = pd.DataFrame({"timestamp": query_times, "value": counts[days]}).groupby("timestamp", sort=False)["value"].rank(method="average", pct=True).to_numpy(dtype=np.float32)
        collector.add(f"cohort_percentile_{days}", rank_values, "site_drift_addition")
    dates = pd.to_datetime(query_times, unit="ms")
    month = dates.month.to_numpy(dtype=np.float32)
    day_of_year = dates.dayofyear.to_numpy(dtype=np.float32)
    collector.add("month", month, "core_calendar")
    collector.add("annual_sin", np.sin(2.0 * np.pi * day_of_year / 365.25), "core_calendar")
    collector.add("annual_cos", np.cos(2.0 * np.pi * day_of_year / 365.25), "core_calendar")
    platform_start = np.datetime64("2000-04-02", "ms").astype(np.int64)
    collector.add("platform_age_days", (query_times - platform_start) / DAY_MS, "core_calendar")
    core_count = len(collector.names)
    if wide:
        for category_name in ("beer", "style", "brewer", "brewer_country"):
            _category_features(collector, f"rating_{category_name}", "wide_beer_relational", event_users, event_times, np.asarray(rating[category_name], dtype=np.int32), query_users, query_times, (90, 365))
        novel = first_category_flags(event_users, np.asarray(rating["beer"], dtype=np.int32), int(np.max(rating["beer"])))
        for days in (90, 365):
            novel_count = _prefix_window(novel, left_indices[days], right)
            collector.add(f"novel_beer_share_{days}", novel_count / np.maximum(counts[days], 1.0), "wide_beer_relational")
            alcohol_sum = _prefix_window(np.asarray(rating["alcohol"]), left_indices[days], right)
            alcohol_sq = _prefix_window(np.asarray(rating["alcohol"], dtype=np.float32) ** 2, left_indices[days], right)
            alcohol_mean = alcohol_sum / np.maximum(counts[days], 1.0)
            collector.add(f"alcohol_mean_{days}", alcohol_mean, "wide_beer_relational")
            collector.add(f"alcohol_std_{days}", np.sqrt(np.maximum(alcohol_sq / np.maximum(counts[days], 1.0) - alcohol_mean * alcohol_mean, 0.0)), "wide_beer_relational")
        place_users = np.asarray(place["user"], dtype=np.int32)
        place_times = np.asarray(place["time"], dtype=np.int64)
        for category_name in ("place", "place_type", "country", "state"):
            _category_features(collector, f"place_rating_{category_name}", "wide_place_relational", place_users, place_times, np.asarray(place[category_name], dtype=np.int32), query_users, query_times, (90, 365), category_name in ("place_type", "country"))
        favorite_users = np.asarray(favorite["user"], dtype=np.int32)
        favorite_times = np.asarray(favorite["time"], dtype=np.int64)
        for category_name in ("beer", "style", "brewer"):
            _category_features(collector, f"favorite_{category_name}", "wide_favorite_relational", favorite_users, favorite_times, np.asarray(favorite[category_name], dtype=np.int32), query_users, query_times, (90, 365), False)
    matrix_sorted = collector.matrix()
    matrix = matrix_sorted[inverse]
    names = collector.names
    groups = collector.groups
    print(f"[features] built {len(seed_frame)}x{len(names)} ({core_count} core) in {time.time() - started:.1f}s")
    return matrix, names, groups


def brute_force_checks(seed_frame: pd.DataFrame, matrix: np.ndarray, names: list[str]) -> None:
    ratings = source_root() / "db" / "beer_ratings.parquet"
    con = _duckdb_connection()
    indices = np.linspace(0, len(seed_frame) - 1, min(8, len(seed_frame)), dtype=int)
    checked = 0
    for index in indices:
        row = seed_frame.iloc[index]
        result = con.execute(
            f"""
                SELECT
                    count(*) FILTER (WHERE created_at > ? - INTERVAL '30 days') AS c30,
                    count(*) FILTER (WHERE created_at > ? - INTERVAL '90 days') AS c90,
                    count(*) FILTER (WHERE created_at > ? - INTERVAL '365 days') AS c365,
                    count(DISTINCT beer_id) FILTER (WHERE created_at > ? - INTERVAL '90 days') AS b90
                FROM read_parquet('{ratings}')
                WHERE user_id = ? AND created_at <= ?
            """,
            [row.timestamp, row.timestamp, row.timestamp, row.timestamp, int(row.user_id), row.timestamp],
        ).fetchone()
        for feature, value in zip(("rating_count_30", "rating_count_90", "rating_count_365", "rating_beer_unique_90"), result):
            if feature in names and not np.isclose(matrix[index, names.index(feature)], float(value)):
                raise RuntimeError(f"feature audit mismatch row={index} feature={feature} expected={value} actual={matrix[index, names.index(feature)]}")
        checked += 1
    con.close()
    print(f"[audit] brute-force temporal checks passed for {checked} representative rows")
