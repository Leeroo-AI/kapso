from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


N_BEERS = 751_524
HALF_LIVES = np.asarray([14.0, 45.0, 120.0], dtype=np.float32)
FEATURE_NAMES = [
    "beer_rating_decay_14", "beer_rating_decay_45", "beer_rating_decay_120",
    "beer_favorite_decay_14", "beer_favorite_decay_45", "beer_favorite_decay_120",
    "beer_rating_count", "beer_favorite_count", "beer_rating_mean", "beer_rating_std",
    "beer_liked_ratio", "beer_favorite_conversion", "beer_rating_velocity_short",
    "beer_rating_velocity_long", "beer_favorite_velocity_short", "beer_favorite_velocity_long",
    "beer_rating_acceleration", "beer_favorite_acceleration", "beer_month_popularity",
    "beer_month_share", "beer_age", "beer_created_after_origin", "beer_alcohol",
    "beer_ibu", "beer_ibu_missing", "beer_seasonal", "beer_one_off", "beer_alias",
    "beer_verified", "beer_retired", "beer_has_picture", "brewer_country",
    "user_ratings_7", "user_ratings_30", "user_ratings_90", "user_ratings_365",
    "user_ratings_lifetime", "user_favorites_7", "user_favorites_30", "user_favorites_90",
    "user_favorites_365", "user_favorites_lifetime", "user_score_mean", "user_score_std",
    "user_score_max", "user_score_min", "user_liked_ratio", "user_rating_gap",
    "user_tenure", "user_active_span", "user_repeat_rate", "user_style_diversity",
    "user_brewer_diversity", "user_top_style_share", "user_top_brewer_share",
    "user_place_ratings_30", "user_place_ratings_90", "user_place_ratings_365",
    "user_place_ratings_lifetime", "pair_rating_count", "pair_best_score", "pair_mean_score",
    "pair_latest_score", "pair_last_rating_age", "pair_style_count", "pair_style_share",
    "pair_style_score", "pair_brewer_count", "pair_brewer_share", "pair_brewer_score",
    "pair_geography_match", "pair_own_rank", "pair_previously_favorited",
    "source_popularity_flag", "source_popularity_rank", "source_popularity_score",
    "source_own_flag", "source_own_rank", "source_own_score", "source_bm25_flag",
    "source_bm25_rank", "source_bm25_score", "source_cofavorite_flag",
    "source_cofavorite_rank", "source_cofavorite_score", "source_affinity_flag",
    "source_affinity_rank", "source_affinity_score", "origin_month_sin", "origin_month_cos",
]


def register_artifact(cache: Path, entry: dict) -> None:
    path = cache / "artifacts.json"
    path.touch(exist_ok=True)
    with path.open("r+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        raw = handle.read().strip()
        entries = json.loads(raw) if raw else []
        if not any(value.get("content_key") == entry["content_key"] for value in entries):
            entries.append(entry)
            handle.seek(0)
            handle.truncate()
            json.dump(entries, handle, indent=2)
            handle.write("\n")
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def database_fingerprint(db_dir: Path) -> str:
    digest = hashlib.sha256()
    for name in ("beer_ratings.parquet", "favorites.parquet", "place_ratings.parquet", "beers.parquet"):
        path = db_dir / name
        stat = path.stat()
        digest.update(f"{name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:16]


def timestamp_seconds(value: pd.Timestamp | dt.datetime | np.datetime64) -> int:
    return int(pd.Timestamp(value).value // 1_000_000_000)


def make_episodes(connection: duckdb.DuckDBPyConnection, db_dir: Path, cache: Path, chain: str, start: str, end: str, fingerprint: str) -> pd.DataFrame:
    key = f"ratebeer_replay_{chain}_{start}_{end}_90d_{fingerprint}_v2"
    directory = cache / "temporal_ranker_lane0"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.parquet"
    if not path.exists():
        favorites = db_dir / "favorites.parquet"
        ratings = db_dir / "beer_ratings.parquet"
        query = f"""
            WITH origins AS (
                SELECT generate_series origin
                FROM generate_series(DATE '{start}', DATE '{end}', INTERVAL 7 DAY)
            )
            SELECT o.origin, f.user_id, list(DISTINCT f.beer_id ORDER BY f.beer_id) labels
            FROM origins o
            JOIN read_parquet('{favorites}') f
              ON f.created_at > o.origin
             AND f.created_at <= o.origin + INTERVAL 90 DAY
            WHERE f.user_id IS NOT NULL AND f.beer_id IS NOT NULL
              AND EXISTS (
                  SELECT 1 FROM read_parquet('{ratings}') br
                  WHERE br.user_id = f.user_id
                    AND br.created_at > o.origin - INTERVAL 90 DAY
                    AND br.created_at <= o.origin
              )
            GROUP BY o.origin, f.user_id
            ORDER BY o.origin, f.user_id
        """
        connection.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        register_artifact(cache, {
            "name": f"RateBeer {chain} weekly replay episodes",
            "path": str(path.relative_to(cache)),
            "description": "Exact 90-day favorite labels with the prior-90-day active-rater condition.",
            "content_key": key,
            "rebuild_hint": "Run full temporal_ranker.py pipeline against the same sanitized database fingerprint.",
        })
    frame = pd.read_parquet(path)
    expected = 8_443 if chain == "A" else 68_626
    if len(frame) != expected:
        raise RuntimeError(f"episode count for chain {chain} is {len(frame)}, expected {expected}")
    return frame


class StaticStore:
    def __init__(self, connection: duckdb.DuckDBPyConnection, db_dir: Path):
        query = f"""
            SELECT b.beer_id, b.brewer_id, b.style_id, epoch(b.created_at)::BIGINT created_time,
                b.alcohol_pct, b.ibu, b.is_seasonal, b.is_one_off, b.is_alias, b.is_verified,
                b.is_retired, b.has_pic, br.country_id brewer_country
            FROM read_parquet('{db_dir / 'beers.parquet'}') b
            LEFT JOIN read_parquet('{db_dir / 'brewers.parquet'}') br USING (brewer_id)
        """
        frame = connection.execute(query).fetchdf()
        self.brewer = np.full(N_BEERS, -1, dtype=np.int32)
        self.style = np.full(N_BEERS, -1, dtype=np.int16)
        self.created = np.zeros(N_BEERS, dtype=np.int64)
        self.alcohol = np.zeros(N_BEERS, dtype=np.float32)
        self.ibu = np.zeros(N_BEERS, dtype=np.float32)
        self.ibu_missing = np.ones(N_BEERS, dtype=np.float32)
        self.seasonal = np.zeros(N_BEERS, dtype=np.float32)
        self.one_off = np.zeros(N_BEERS, dtype=np.float32)
        self.alias = np.zeros(N_BEERS, dtype=np.float32)
        self.verified = np.zeros(N_BEERS, dtype=np.float32)
        self.retired = np.zeros(N_BEERS, dtype=np.float32)
        self.picture = np.zeros(N_BEERS, dtype=np.float32)
        self.brewer_country = np.full(N_BEERS, -1, dtype=np.int16)
        ids = frame["beer_id"].to_numpy(np.int64)
        self.brewer[ids] = frame["brewer_id"].fillna(-1).to_numpy(np.int32)
        self.style[ids] = frame["style_id"].fillna(-1).to_numpy(np.int16)
        self.created[ids] = frame["created_time"].fillna(0).to_numpy(np.int64)
        self.alcohol[ids] = frame["alcohol_pct"].fillna(0).to_numpy(np.float32)
        ibu = frame["ibu"]
        self.ibu[ids] = ibu.fillna(0).to_numpy(np.float32)
        self.ibu_missing[ids] = ibu.isna().to_numpy(np.float32)
        for column, target in (("is_seasonal", self.seasonal), ("is_one_off", self.one_off), ("is_alias", self.alias), ("is_verified", self.verified), ("is_retired", self.retired), ("has_pic", self.picture)):
            target[ids] = frame[column].fillna(False).to_numpy(np.float32)
        self.brewer_country[ids] = frame["brewer_country"].fillna(-1).to_numpy(np.int16)


class GroupedEvents:
    def __init__(self, users: np.ndarray, items: np.ndarray, times: np.ndarray, scores: np.ndarray | None = None, countries: np.ndarray | None = None):
        self.users = users
        self.items = items
        self.times = times
        self.scores = scores
        self.countries = countries
        unique, starts, counts = np.unique(users, return_index=True, return_counts=True)
        self.bounds = {int(user): (int(start), int(start + count)) for user, start, count in zip(unique, starts, counts)}

    def stop(self, user: int, origin: int) -> tuple[int, int]:
        start, end = self.bounds.get(int(user), (0, 0))
        if end == 0:
            return 0, 0
        stop = start + int(np.searchsorted(self.times[start:end], origin, side="right"))
        return start, stop


def load_user_events(connection: duckdb.DuckDBPyConnection, db_dir: Path, users: np.ndarray, end_time: int) -> tuple[GroupedEvents, GroupedEvents, GroupedEvents]:
    connection.register("rank_users", pd.DataFrame({"user_id": np.asarray(users, dtype=np.int64)}))
    ratings = connection.execute(f"""
        SELECT user_id::INTEGER user_id, beer_id::INTEGER beer_id,
            epoch(created_at)::BIGINT event_time, total_score::FLOAT score
        FROM read_parquet('{db_dir / 'beer_ratings.parquet'}')
        SEMI JOIN rank_users USING (user_id)
        WHERE beer_id IS NOT NULL AND total_score IS NOT NULL
          AND created_at <= to_timestamp({end_time})::TIMESTAMP
        ORDER BY user_id, created_at, beer_id
    """).fetchnumpy()
    favorites = connection.execute(f"""
        SELECT user_id::INTEGER user_id, beer_id::INTEGER beer_id,
            epoch(created_at)::BIGINT event_time
        FROM read_parquet('{db_dir / 'favorites.parquet'}')
        SEMI JOIN rank_users USING (user_id)
        WHERE beer_id IS NOT NULL AND created_at <= to_timestamp({end_time})::TIMESTAMP
        ORDER BY user_id, created_at, beer_id
    """).fetchnumpy()
    places = connection.execute(f"""
        SELECT pr.user_id::INTEGER user_id, coalesce(pr.place_id, -1)::INTEGER place_id,
            epoch(pr.created_at)::BIGINT event_time, coalesce(p.country_id, -1)::INTEGER country_id
        FROM read_parquet('{db_dir / 'place_ratings.parquet'}') pr
        SEMI JOIN rank_users USING (user_id)
        LEFT JOIN read_parquet('{db_dir / 'places.parquet'}') p USING (place_id)
        WHERE pr.created_at <= to_timestamp({end_time})::TIMESTAMP
        ORDER BY pr.user_id, pr.created_at, pr.place_id
    """).fetchnumpy()
    connection.unregister("rank_users")
    rating_store = GroupedEvents(ratings["user_id"], ratings["beer_id"], ratings["event_time"], ratings["score"])
    favorite_store = GroupedEvents(favorites["user_id"], favorites["beer_id"], favorites["event_time"])
    place_store = GroupedEvents(places["user_id"], places["place_id"], places["event_time"], countries=places["country_id"])
    return rating_store, favorite_store, place_store


class GlobalEvents:
    def __init__(self, connection: duckdb.DuckDBPyConnection, db_dir: Path):
        started = time.time()
        ratings = connection.execute(f"""
            SELECT beer_id::INTEGER beer_id, epoch(created_at)::BIGINT event_time,
                total_score::FLOAT score, month(created_at)::UTINYINT event_month
            FROM read_parquet('{db_dir / 'beer_ratings.parquet'}')
            WHERE beer_id IS NOT NULL AND total_score IS NOT NULL
            ORDER BY created_at, beer_id
        """).fetchnumpy()
        favorites = connection.execute(f"""
            SELECT user_id::INTEGER user_id, beer_id::INTEGER beer_id,
                epoch(created_at)::BIGINT event_time
            FROM read_parquet('{db_dir / 'favorites.parquet'}')
            WHERE user_id IS NOT NULL AND beer_id IS NOT NULL
            ORDER BY created_at, user_id, beer_id
        """).fetchnumpy()
        self.rating_items = ratings["beer_id"]
        self.rating_times = ratings["event_time"]
        self.rating_scores = ratings["score"]
        self.rating_months = ratings["event_month"]
        self.favorite_users = favorites["user_id"]
        self.favorite_items = favorites["beer_id"]
        self.favorite_times = favorites["event_time"]
        print(f"[ranker] loaded global events ratings={len(self.rating_items)} favorites={len(self.favorite_items)} elapsed={time.time() - started:.1f}s")


@dataclass
class PopularityContext:
    popular_ids: np.ndarray
    popular_scores: np.ndarray
    popular_rank: np.ndarray
    style_lists: dict[int, np.ndarray]
    brewer_lists: dict[int, np.ndarray]


class GlobalTimeline:
    def __init__(self, events: GlobalEvents, static: StaticStore):
        self.events = events
        self.static = static
        self.current = int(events.rating_times[0])
        self.rating_pointer = 0
        self.favorite_pointer = 0
        self.rating_decay = np.zeros((3, N_BEERS), dtype=np.float32)
        self.favorite_decay = np.zeros((3, N_BEERS), dtype=np.float32)
        self.rating_count = np.zeros(N_BEERS, dtype=np.int32)
        self.favorite_count = np.zeros(N_BEERS, dtype=np.int32)
        self.rating_sum = np.zeros(N_BEERS, dtype=np.float32)
        self.rating_sumsq = np.zeros(N_BEERS, dtype=np.float32)
        self.liked_count = np.zeros(N_BEERS, dtype=np.int32)
        self.month_count = np.zeros((12, N_BEERS), dtype=np.int32)

    def advance(self, origin: int) -> None:
        if origin < self.current:
            raise RuntimeError("global timeline cannot move backwards")
        elapsed_days = (origin - self.current) / 86400
        if self.rating_pointer or self.favorite_pointer:
            self.rating_decay *= np.exp(-math.log(2) * elapsed_days / HALF_LIVES)[:, None]
            self.favorite_decay *= np.exp(-math.log(2) * elapsed_days / HALF_LIVES)[:, None]
        rating_stop = int(np.searchsorted(self.events.rating_times, origin, side="right"))
        if rating_stop > self.rating_pointer:
            section = slice(self.rating_pointer, rating_stop)
            items = self.events.rating_items[section]
            times = self.events.rating_times[section]
            scores = self.events.rating_scores[section]
            months = self.events.rating_months[section].astype(np.int64) - 1
            age = (origin - times).astype(np.float32) / 86400
            for index, half_life in enumerate(HALF_LIVES):
                np.add.at(self.rating_decay[index], items, np.exp(-math.log(2) * age / half_life))
            np.add.at(self.rating_count, items, 1)
            np.add.at(self.rating_sum, items, scores)
            np.add.at(self.rating_sumsq, items, scores * scores)
            np.add.at(self.liked_count, items[scores >= 3.8], 1)
            np.add.at(self.month_count, (months, items), 1)
            self.rating_pointer = rating_stop
        favorite_stop = int(np.searchsorted(self.events.favorite_times, origin, side="right"))
        if favorite_stop > self.favorite_pointer:
            section = slice(self.favorite_pointer, favorite_stop)
            items = self.events.favorite_items[section]
            times = self.events.favorite_times[section]
            age = (origin - times).astype(np.float32) / 86400
            for index, half_life in enumerate(HALF_LIVES):
                np.add.at(self.favorite_decay[index], items, np.exp(-math.log(2) * age / half_life))
            np.add.at(self.favorite_count, items, 1)
            self.favorite_pointer = favorite_stop
        self.current = origin

    def popularity_context(self) -> PopularityContext:
        score = (
            2.5 * np.log1p(self.favorite_decay[0]) +
            1.5 * np.log1p(self.favorite_decay[1]) +
            np.log1p(self.favorite_decay[2]) +
            0.25 * np.log1p(self.rating_decay[0]) +
            0.15 * np.log1p(self.rating_decay[1]) +
            0.1 * np.log1p(self.rating_decay[2])
        )
        count = min(50_000, int(np.count_nonzero(self.rating_count)))
        indices = np.argpartition(score, -count)[-count:]
        order = np.lexsort((indices, -score[indices]))
        ranked = indices[order].astype(np.int32)
        popular_ids = ranked[:2000]
        popular_scores = score[popular_ids].astype(np.float32)
        popular_rank = np.zeros(N_BEERS, dtype=np.int16)
        popular_rank[popular_ids] = np.arange(1, len(popular_ids) + 1, dtype=np.int16)
        styles: dict[int, list[int]] = defaultdict(list)
        brewers: dict[int, list[int]] = defaultdict(list)
        for item in ranked:
            style = int(self.static.style[item])
            brewer = int(self.static.brewer[item])
            if len(styles[style]) < 160:
                styles[style].append(int(item))
            if len(brewers[brewer]) < 80:
                brewers[brewer].append(int(item))
        style_lists = {key: np.asarray(value, dtype=np.int32) for key, value in styles.items()}
        brewer_lists = {key: np.asarray(value, dtype=np.int32) for key, value in brewers.items()}
        return PopularityContext(popular_ids, popular_scores, popular_rank, style_lists, brewer_lists)


class FavoriteTimeline:
    def __init__(self, events: GlobalEvents):
        self.users = events.favorite_users
        self.items = events.favorite_items
        self.times = events.favorite_times
        self.pointer = 0
        self.current = 0
        self.by_user: dict[int, set[int]] = defaultdict(set)
        self.by_item: dict[int, list[int]] = defaultdict(list)

    def advance(self, origin: int) -> None:
        if origin < self.current:
            raise RuntimeError("favorite timeline cannot move backwards")
        stop = int(np.searchsorted(self.times, origin, side="right"))
        for user, item in zip(self.users[self.pointer:stop], self.items[self.pointer:stop]):
            user_value = int(user)
            item_value = int(item)
            self.by_user[user_value].add(item_value)
            self.by_item[item_value].append(user_value)
        self.pointer = stop
        self.current = origin

    def candidates(self, user: int, excluded: set[int]) -> tuple[np.ndarray, np.ndarray]:
        scores: dict[int, float] = defaultdict(float)
        for shared_item in excluded:
            neighbors = self.by_item.get(shared_item, [])
            for neighbor in neighbors[-100:]:
                if neighbor == user:
                    continue
                other = self.by_user.get(neighbor, set())
                weight = 1.0 / math.log2(2 + len(other))
                for item in other:
                    if item not in excluded:
                        scores[item] += weight
        ranked = sorted(scores, key=lambda item: (-scores[item], item))[:300]
        return np.asarray(ranked, dtype=np.int32), np.asarray([scores[item] for item in ranked], dtype=np.float32)


class NeighborSnapshot:
    def __init__(self, frame: pd.DataFrame):
        self.seeds = frame["seed"].to_numpy(np.int32)
        self.candidates = frame["candidate"].to_numpy(np.int32)
        self.scores = frame["score"].to_numpy(np.float32)
        unique, starts, counts = np.unique(self.seeds, return_index=True, return_counts=True)
        self.bounds = {int(seed): (int(start), int(start + count)) for seed, start, count in zip(unique, starts, counts)}

    def get(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = self.bounds.get(int(seed), (0, 0))
        return self.candidates[start:stop], self.scores[start:stop]


class BM25Store:
    def __init__(self, connection: duckdb.DuckDBPyConnection, db_dir: Path, cache: Path, seed_items: np.ndarray, fingerprint: str):
        self.connection = connection
        self.db_dir = db_dir
        self.cache = cache
        self.seed_items = np.asarray(seed_items, dtype=np.int32)
        self.fingerprint = fingerprint
        self.loaded: dict[str, NeighborSnapshot] = {}
        self.seed_hash = hashlib.sha256(self.seed_items.tobytes()).hexdigest()[:12]

    def snapshot_date(self, origin: int) -> dt.date:
        value = dt.datetime.fromtimestamp(origin, tz=dt.timezone.utc)
        month = 1 + 3 * ((value.month - 1) // 3)
        return dt.date(value.year, month, 1)

    def get(self, origin: int) -> NeighborSnapshot:
        cutoff = self.snapshot_date(origin).isoformat()
        if cutoff in self.loaded:
            return self.loaded[cutoff]
        directory = self.cache / "temporal_ranker_lane0" / "bm25"
        directory.mkdir(parents=True, exist_ok=True)
        key = f"ratebeer_bm25_{cutoff}_{self.fingerprint}_{self.seed_hash}_v3_last30"
        path = directory / f"{key}.parquet"
        if not path.exists():
            self.connection.register("bm25_seed_items", pd.DataFrame({"beer_id": self.seed_items}))
            ratings = self.db_dir / "beer_ratings.parquet"
            query = f"""
                WITH liked AS (
                    SELECT user_id, beer_id
                    FROM read_parquet('{ratings}')
                    WHERE created_at <= DATE '{cutoff}' AND total_score >= 3.8 AND beer_id IS NOT NULL
                    QUALIFY row_number() OVER (PARTITION BY user_id ORDER BY created_at DESC, beer_id) <= 30
                ), docs AS (
                    SELECT user_id, count(*) dl FROM liked GROUP BY user_id
                ), corpus AS (
                    SELECT count(*) n, avg(dl) avgdl FROM docs
                ), dfs AS (
                    SELECT beer_id, count(*) df FROM liked GROUP BY beer_id
                ), pairs AS (
                    SELECT a.beer_id seed, b.beer_id candidate,
                        sum(ln(1 + (corpus.n - dfs.df + 0.5) / (dfs.df + 0.5)) /
                            (0.25 + 0.75 * docs.dl / corpus.avgdl)) score
                    FROM liked a
                    SEMI JOIN bm25_seed_items s ON a.beer_id = s.beer_id
                    JOIN liked b ON a.user_id = b.user_id AND a.beer_id <> b.beer_id
                    JOIN docs ON docs.user_id = a.user_id
                    JOIN dfs ON dfs.beer_id = a.beer_id
                    CROSS JOIN corpus
                    GROUP BY a.beer_id, b.beer_id
                )
                SELECT seed, candidate, score
                FROM pairs
                QUALIFY row_number() OVER (PARTITION BY seed ORDER BY score DESC, candidate) <= 30
                ORDER BY seed, score DESC, candidate
            """
            self.connection.execute(f"COPY ({query}) TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
            self.connection.unregister("bm25_seed_items")
            register_artifact(self.cache, {
                "name": f"RateBeer BM25 neighbors at {cutoff}",
                "path": str(path.relative_to(self.cache)),
                "description": "Quarterly temporally censored BM25 rating co-occurrence neighbors.",
                "content_key": key,
                "rebuild_hint": "Rebuild from beer_ratings with the recorded cutoff and seed hash.",
            })
        frame = pd.read_parquet(path)
        snapshot = NeighborSnapshot(frame)
        self.loaded[cutoff] = snapshot
        print(f"[ranker] BM25 snapshot={cutoff} pairs={len(frame)}")
        return snapshot


@dataclass
class UserContext:
    user_features: np.ndarray
    pair_info: dict[int, tuple[float, float, float, float, float, float]]
    own_ids: np.ndarray
    own_scores: np.ndarray
    style_stats: dict[int, tuple[float, float]]
    brewer_stats: dict[int, tuple[float, float]]
    favorite_set: set[int]
    liked_recent: np.ndarray
    country: int
    rating_count: int


def log_counts(times: np.ndarray, origin: int, windows: tuple[int, ...]) -> list[float]:
    values = []
    for window in windows:
        start = origin - window * 86400
        values.append(math.log1p(len(times) - int(np.searchsorted(times, start, side="right"))))
    values.append(math.log1p(len(times)))
    return values


def make_user_context(user: int, origin: int, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore) -> UserContext:
    rating_start, rating_stop = ratings.stop(user, origin)
    rating_times = ratings.times[rating_start:rating_stop]
    rating_items = ratings.items[rating_start:rating_stop]
    rating_scores = ratings.scores[rating_start:rating_stop]
    favorite_start, favorite_stop = favorites.stop(user, origin)
    favorite_times = favorites.times[favorite_start:favorite_stop]
    favorite_items = favorites.items[favorite_start:favorite_stop]
    place_start, place_stop = places.stop(user, origin)
    place_times = places.times[place_start:place_stop]
    countries = places.countries[place_start:place_stop]
    pair_raw: dict[int, list[float]] = {}
    for item, event_time, score in zip(rating_items, rating_times, rating_scores):
        key = int(item)
        if key not in pair_raw:
            pair_raw[key] = [1.0, float(score), float(score), float(score), float(event_time)]
        else:
            value = pair_raw[key]
            value[0] += 1
            value[1] = max(value[1], float(score))
            value[2] += float(score)
            if event_time >= value[4]:
                value[3] = float(score)
                value[4] = float(event_time)
    own_values = []
    pair_info: dict[int, tuple[float, float, float, float, float, float]] = {}
    for item, value in pair_raw.items():
        count, best, total, latest, event_time = value
        mean = total / count
        age = max(0.0, (origin - event_time) / 86400)
        own_score = best + 0.25 * mean + 0.35 * math.exp(-math.log(2) * age / 45) + 0.08 * math.log1p(count)
        own_values.append((item, own_score))
        pair_info[item] = (count, best, mean, latest, age, 501.0)
    own_values.sort(key=lambda value: (-value[1], value[0]))
    own_values = own_values[:500]
    own_ids = np.asarray([value[0] for value in own_values], dtype=np.int32)
    own_scores = np.asarray([value[1] for value in own_values], dtype=np.float32)
    for rank, item in enumerate(own_ids, 1):
        value = pair_info[int(item)]
        pair_info[int(item)] = value[:5] + (float(rank),)
    style_stats_raw: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    brewer_stats_raw: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for item, value in pair_info.items():
        count, _, mean, _, _, _ = value
        style = int(static.style[item])
        brewer = int(static.brewer[item])
        style_stats_raw[style][0] += count
        style_stats_raw[style][1] += count * mean
        brewer_stats_raw[brewer][0] += count
        brewer_stats_raw[brewer][1] += count * mean
    style_stats = {key: (value[0], value[1] / max(value[0], 1)) for key, value in style_stats_raw.items()}
    brewer_stats = {key: (value[0], value[1] / max(value[0], 1)) for key, value in brewer_stats_raw.items()}
    rating_windows = log_counts(rating_times, origin, (7, 30, 90, 365))
    favorite_windows = log_counts(favorite_times, origin, (7, 30, 90, 365))
    if len(rating_scores):
        score_values = [float(np.mean(rating_scores)), float(np.std(rating_scores)), float(np.max(rating_scores)), float(np.min(rating_scores)), float(np.mean(rating_scores >= 3.8))]
        gap = math.log1p(max(0, (origin - int(rating_times[-1])) / 86400))
        tenure = math.log1p(max(0, (origin - int(rating_times[0])) / 86400))
        span = math.log1p(max(0, (int(rating_times[-1]) - int(rating_times[0])) / 86400))
    else:
        score_values = [0.0] * 5
        gap = tenure = span = 0.0
    repeat_rate = 1.0 - len(pair_info) / max(len(rating_items), 1)
    total_ratings = max(len(rating_items), 1)
    top_style = max((value[0] for value in style_stats.values()), default=0.0) / total_ratings
    top_brewer = max((value[0] for value in brewer_stats.values()), default=0.0) / total_ratings
    place_windows = log_counts(place_times, origin, (30, 90, 365))
    user_features = np.asarray(
        rating_windows + favorite_windows + score_values + [gap, tenure, span, repeat_rate,
        math.log1p(len(style_stats)), math.log1p(len(brewer_stats)), top_style, top_brewer] + place_windows,
        dtype=np.float32,
    )
    country = -1
    if len(countries):
        valid = countries[countries >= 0]
        if len(valid):
            values, counts = np.unique(valid, return_counts=True)
            country = int(values[np.argmax(counts)])
    favorite_set = set(int(value) for value in favorite_items)
    liked_mask = rating_scores >= 3.8
    liked_recent = rating_items[liked_mask][-30:][::-1].astype(np.int32)
    return UserContext(user_features, pair_info, own_ids, own_scores, style_stats, brewer_stats, favorite_set, liked_recent, country, len(rating_items))


@dataclass
class SourceBundle:
    pop_ids: np.ndarray
    pop_scores: np.ndarray
    pop_rank: np.ndarray
    own_ids: np.ndarray
    own_map: dict[int, tuple[int, float]]
    bm25_ids: np.ndarray
    bm25_map: dict[int, tuple[int, float]]
    cofavorite_ids: np.ndarray
    cofavorite_map: dict[int, tuple[int, float]]
    affinity_ids: np.ndarray
    affinity_map: dict[int, tuple[int, float]]

    def arrays(self) -> list[np.ndarray]:
        return [self.pop_ids, self.own_ids, self.bm25_ids, self.cofavorite_ids, self.affinity_ids]


def ranked_map(items: np.ndarray, scores: np.ndarray) -> dict[int, tuple[int, float]]:
    return {int(item): (rank, float(score)) for rank, (item, score) in enumerate(zip(items, scores), 1)}


def make_sources(user: int, context: UserContext, popularity: PopularityContext, bm25: NeighborSnapshot, favorites: FavoriteTimeline, static: StaticStore) -> SourceBundle:
    own_mask = np.asarray([int(item) not in context.favorite_set for item in context.own_ids], dtype=bool)
    own_ids = context.own_ids[own_mask]
    own_scores = context.own_scores[own_mask]
    neighbor_scores: dict[int, float] = defaultdict(float)
    for seed_rank, seed in enumerate(context.liked_recent, 1):
        items, scores = bm25.get(int(seed))
        seed_weight = 1.0 / math.log2(seed_rank + 1)
        for item, score in zip(items, scores):
            value = int(item)
            if value not in context.favorite_set:
                neighbor_scores[value] += seed_weight * float(score)
    bm25_ranked = sorted(neighbor_scores, key=lambda item: (-neighbor_scores[item], item))[:500]
    bm25_ids = np.asarray(bm25_ranked, dtype=np.int32)
    bm25_scores = np.asarray([neighbor_scores[item] for item in bm25_ranked], dtype=np.float32)
    cofavorite_ids, cofavorite_scores = favorites.candidates(user, context.favorite_set)
    affinity_scores: dict[int, float] = defaultdict(float)
    style_total = max(sum(value[0] for value in context.style_stats.values()), 1.0)
    brewer_total = max(sum(value[0] for value in context.brewer_stats.values()), 1.0)
    top_styles = sorted(context.style_stats, key=lambda key: (-context.style_stats[key][0], key))[:4]
    top_brewers = sorted(context.brewer_stats, key=lambda key: (-context.brewer_stats[key][0], key))[:6]
    for style in top_styles:
        strength = context.style_stats[style][0] / style_total
        for rank, item in enumerate(popularity.style_lists.get(style, np.empty(0, dtype=np.int32)), 1):
            if int(item) not in context.favorite_set:
                affinity_scores[int(item)] += strength / math.log2(rank + 1)
    for brewer in top_brewers:
        strength = context.brewer_stats[brewer][0] / brewer_total
        for rank, item in enumerate(popularity.brewer_lists.get(brewer, np.empty(0, dtype=np.int32)), 1):
            if int(item) not in context.favorite_set:
                affinity_scores[int(item)] += strength / math.log2(rank + 1)
    affinity_ranked = sorted(affinity_scores, key=lambda item: (-affinity_scores[item], item))[:300]
    affinity_ids = np.asarray(affinity_ranked, dtype=np.int32)
    affinity_values = np.asarray([affinity_scores[item] for item in affinity_ranked], dtype=np.float32)
    return SourceBundle(
        popularity.popular_ids, popularity.popular_scores, popularity.popular_rank,
        own_ids, ranked_map(own_ids, own_scores), bm25_ids, ranked_map(bm25_ids, bm25_scores),
        cofavorite_ids, ranked_map(cofavorite_ids, cofavorite_scores), affinity_ids,
        ranked_map(affinity_ids, affinity_values),
    )


def source_features(items: np.ndarray, sources: SourceBundle) -> np.ndarray:
    size = len(items)
    values = np.zeros((size, 15), dtype=np.float32)
    ranks = sources.pop_rank[items]
    values[:, 0] = ranks > 0
    present = ranks > 0
    values[present, 1] = 1.0 / np.log2(ranks[present].astype(np.float32) + 1)
    pop_score_lookup = {int(item): float(score) for item, score in zip(sources.pop_ids, sources.pop_scores)}
    values[:, 2] = np.asarray([math.log1p(pop_score_lookup.get(int(item), 0.0)) for item in items], dtype=np.float32)
    for source_index, mapping in enumerate((sources.own_map, sources.bm25_map, sources.cofavorite_map, sources.affinity_map), 1):
        offset = source_index * 3
        for row, item in enumerate(items):
            match = mapping.get(int(item))
            if match is not None:
                rank, score = match
                values[row, offset] = 1
                values[row, offset + 1] = 1.0 / math.log2(rank + 1)
                values[row, offset + 2] = math.log1p(max(score, 0.0))
    return values


def feature_matrix(items: np.ndarray, origin: int, user_context: UserContext, sources: SourceBundle, timeline: GlobalTimeline, static: StaticStore) -> np.ndarray:
    items = np.asarray(items, dtype=np.int32)
    size = len(items)
    matrix = np.zeros((size, len(FEATURE_NAMES)), dtype=np.float32)
    matrix[:, 0:3] = np.log1p(timeline.rating_decay[:, items].T)
    matrix[:, 3:6] = np.log1p(timeline.favorite_decay[:, items].T)
    rating_count = timeline.rating_count[items].astype(np.float32)
    favorite_count = timeline.favorite_count[items].astype(np.float32)
    matrix[:, 6] = np.log1p(rating_count)
    matrix[:, 7] = np.log1p(favorite_count)
    mean = timeline.rating_sum[items] / np.maximum(rating_count, 1)
    variance = timeline.rating_sumsq[items] / np.maximum(rating_count, 1) - mean * mean
    matrix[:, 8] = mean
    matrix[:, 9] = np.sqrt(np.maximum(variance, 0))
    matrix[:, 10] = timeline.liked_count[items] / np.maximum(rating_count, 1)
    matrix[:, 11] = favorite_count / np.maximum(rating_count, 1)
    matrix[:, 12] = timeline.rating_decay[0, items] / np.maximum(timeline.rating_decay[1, items], 0.01)
    matrix[:, 13] = timeline.rating_decay[1, items] / np.maximum(timeline.rating_decay[2, items], 0.01)
    matrix[:, 14] = timeline.favorite_decay[0, items] / np.maximum(timeline.favorite_decay[1, items], 0.01)
    matrix[:, 15] = timeline.favorite_decay[1, items] / np.maximum(timeline.favorite_decay[2, items], 0.01)
    matrix[:, 16] = matrix[:, 12] - matrix[:, 13]
    matrix[:, 17] = matrix[:, 14] - matrix[:, 15]
    month = dt.datetime.fromtimestamp(origin, tz=dt.timezone.utc).month
    month_count = timeline.month_count[month - 1, items].astype(np.float32)
    matrix[:, 18] = np.log1p(month_count)
    matrix[:, 19] = month_count / np.maximum(rating_count, 1)
    age = (origin - static.created[items]).astype(np.float64) / 86400
    matrix[:, 20] = np.log1p(np.maximum(age, 0)).astype(np.float32)
    matrix[:, 21] = age < 0
    matrix[:, 22] = static.alcohol[items]
    matrix[:, 23] = static.ibu[items]
    matrix[:, 24] = static.ibu_missing[items]
    matrix[:, 25] = static.seasonal[items]
    matrix[:, 26] = static.one_off[items]
    matrix[:, 27] = static.alias[items]
    matrix[:, 28] = static.verified[items]
    matrix[:, 29] = static.retired[items]
    matrix[:, 30] = static.picture[items]
    matrix[:, 31] = static.brewer_country[items]
    matrix[:, 32:59] = user_context.user_features
    pair = np.zeros((size, 14), dtype=np.float32)
    rating_total = max(user_context.rating_count, 1)
    for row, item in enumerate(items):
        item_value = int(item)
        info = user_context.pair_info.get(item_value)
        if info is not None:
            pair[row, 0:5] = info[:5]
            pair[row, 12] = 1.0 / math.log2(info[5] + 1) if info[5] <= 500 else 0
        style_value = user_context.style_stats.get(int(static.style[item_value]))
        if style_value is not None:
            pair[row, 5] = style_value[0]
            pair[row, 6] = style_value[0] / rating_total
            pair[row, 7] = style_value[1]
        brewer_value = user_context.brewer_stats.get(int(static.brewer[item_value]))
        if brewer_value is not None:
            pair[row, 8] = brewer_value[0]
            pair[row, 9] = brewer_value[0] / rating_total
            pair[row, 10] = brewer_value[1]
        pair[row, 11] = user_context.country >= 0 and user_context.country == int(static.brewer_country[item_value])
        pair[row, 13] = item_value in user_context.favorite_set
    pair[:, 0] = np.log1p(pair[:, 0])
    pair[:, 4] = np.log1p(pair[:, 4])
    pair[:, 5] = np.log1p(pair[:, 5])
    pair[:, 8] = np.log1p(pair[:, 8])
    matrix[:, 59:73] = pair
    matrix[:, 73:88] = source_features(items, sources)
    angle = 2 * math.pi * month / 12
    matrix[:, 88] = math.sin(angle)
    matrix[:, 89] = math.cos(angle)
    return matrix


def reachable_labels(labels: np.ndarray, sources: SourceBundle, excluded: set[int]) -> list[int]:
    results = []
    for value in labels:
        item = int(value)
        if item in excluded:
            continue
        if sources.pop_rank[item] or item in sources.own_map or item in sources.bm25_map or item in sources.cofavorite_map or item in sources.affinity_map:
            results.append(item)
    return results


def balanced_negatives(sources: SourceBundle, positives: set[int], excluded: set[int], seed: int, count: int = 250) -> list[int]:
    selected: list[int] = []
    used = set(positives) | excluded
    for source_index, values in enumerate(sources.arrays()):
        size = len(values)
        if not size:
            continue
        offset = (seed * (source_index + 3) + 17 * source_index) % size
        step = 37 + 2 * source_index
        attempts = 0
        added = 0
        while attempts < size and added < 50:
            item = int(values[(offset + attempts * step) % size])
            attempts += 1
            if item in used:
                continue
            used.add(item)
            selected.append(item)
            added += 1
    for values in sources.arrays():
        for item_value in values:
            item = int(item_value)
            if item not in used:
                used.add(item)
                selected.append(item)
                if len(selected) == count:
                    return selected
    return selected[:count]


@dataclass
class TrainingMatrix:
    features: np.memmap
    labels: np.ndarray
    weights: np.ndarray
    groups: np.ndarray
    origins: np.ndarray
    total_labels: np.ndarray
    stats: dict


class MatrixWriter:
    def __init__(self, path: Path, max_rows: int):
        self.path = path
        self.max_rows = max_rows
        self.features = np.memmap(path, dtype=np.float32, mode="w+", shape=(max_rows, len(FEATURE_NAMES)))
        self.labels = np.empty(max_rows, dtype=np.uint8)
        self.weights = np.empty(max_rows, dtype=np.float32)
        self.groups: list[int] = []
        self.origins: list[int] = []
        self.total_labels: list[int] = []
        self.position = 0

    def append(self, features: np.ndarray, labels: np.ndarray, weight: float, origin: int, total_labels: int) -> None:
        stop = self.position + len(labels)
        self.features[self.position:stop] = features
        self.labels[self.position:stop] = labels
        self.weights[self.position:stop] = weight
        self.groups.append(len(labels))
        self.origins.append(origin)
        self.total_labels.append(total_labels)
        self.position = stop

    def finish(self, prefix: Path, stats: dict) -> TrainingMatrix:
        self.features.flush()
        del self.features
        os.truncate(self.path, self.position * len(FEATURE_NAMES) * np.dtype(np.float32).itemsize)
        np.save(prefix.with_suffix(".labels.npy"), self.labels[:self.position])
        np.save(prefix.with_suffix(".weights.npy"), self.weights[:self.position])
        np.save(prefix.with_suffix(".groups.npy"), np.asarray(self.groups, dtype=np.int32))
        np.save(prefix.with_suffix(".origins.npy"), np.asarray(self.origins, dtype=np.int64))
        np.save(prefix.with_suffix(".total_labels.npy"), np.asarray(self.total_labels, dtype=np.int16))
        metadata = {"rows": self.position, "features": len(FEATURE_NAMES), "stats": stats, "feature_names": FEATURE_NAMES}
        prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        return load_training_matrix(prefix)


def load_training_matrix(prefix: Path) -> TrainingMatrix:
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    shape = (int(metadata["rows"]), int(metadata["features"]))
    features = np.memmap(prefix.with_suffix(".features.dat"), dtype=np.float32, mode="r", shape=shape)
    return TrainingMatrix(
        features,
        np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r"),
        np.load(prefix.with_suffix(".weights.npy"), mmap_mode="r"),
        np.load(prefix.with_suffix(".groups.npy")),
        np.load(prefix.with_suffix(".origins.npy")),
        np.load(prefix.with_suffix(".total_labels.npy")),
        metadata["stats"],
    )


def build_training_matrix(chain: str, episodes: pd.DataFrame, timeline: GlobalTimeline, favorite_timeline: FavoriteTimeline, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore, bm25_store: BM25Store, cache: Path, fingerprint: str, debug: bool) -> TrainingMatrix:
    version = "v7_debug_last30" if debug else "v7_full_last30"
    key = f"ratebeer_rank_matrix_{chain}_{fingerprint}_{version}_n250_f90"
    directory = cache / "temporal_ranker_lane0" / "matrices"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / key
    feature_path = prefix.with_suffix(".features.dat")
    if prefix.with_suffix(".json").exists() and feature_path.exists():
        matrix = load_training_matrix(prefix)
        print(f"[ranker] reused matrix chain={chain} rows={len(matrix.labels)} groups={len(matrix.groups)}")
        return matrix
    if debug:
        keep_origins = sorted(episodes["origin"].unique())[-2:]
        episodes = pd.concat([
            episodes[episodes["origin"] == origin].head(250) for origin in keep_origins
        ], ignore_index=True)
    max_rows = len(episodes) * 250 + int(sum(len(value) for value in episodes["labels"]))
    writer = MatrixWriter(feature_path, max_rows)
    maximum_origin = timestamp_seconds(episodes["origin"].max())
    label_total = 0
    reachable_total = 0
    reached_episodes = 0
    history_strata = {"no_favorites": [0, 0], "one_to_five_favorites": [0, 0], "over_five_favorites": [0, 0]}
    started = time.time()
    origins = list(episodes.groupby("origin", sort=True))
    for origin_index, (origin_value, origin_frame) in enumerate(origins):
        origin = timestamp_seconds(origin_value)
        timeline.advance(origin)
        favorite_timeline.advance(origin)
        popularity = timeline.popularity_context()
        bm25 = bm25_store.get(origin)
        for row in origin_frame.itertuples(index=False):
            user = int(row.user_id)
            labels = np.asarray(row.labels, dtype=np.int32)
            user_context = make_user_context(user, origin, ratings, favorites, places, static)
            sources = make_sources(user, user_context, popularity, bm25, favorite_timeline, static)
            reachable = reachable_labels(labels, sources, user_context.favorite_set)
            label_total += len(labels)
            reachable_total += len(reachable)
            if len(user_context.favorite_set) == 0:
                stratum = "no_favorites"
            elif len(user_context.favorite_set) <= 5:
                stratum = "one_to_five_favorites"
            else:
                stratum = "over_five_favorites"
            history_strata[stratum][0] += len(labels)
            history_strata[stratum][1] += len(reachable)
            if not reachable:
                continue
            reached_episodes += 1
            positive_set = set(reachable)
            negatives = balanced_negatives(sources, positive_set, user_context.favorite_set, seed=(origin // 86400) ^ user)
            items = np.asarray(reachable + negatives, dtype=np.int32)
            labels_binary = np.zeros(len(items), dtype=np.uint8)
            labels_binary[:len(reachable)] = 1
            features = feature_matrix(items, origin, user_context, sources, timeline, static)
            recency_weight = math.exp(-math.log(2) * (maximum_origin - origin) / 86400 / 180) if chain == "B" else 1.0
            writer.append(features, labels_binary, recency_weight, origin, len(labels))
        if origin_index == 0 or (origin_index + 1) % 8 == 0 or origin_index + 1 == len(origins):
            rate = writer.position / max(time.time() - started, 1e-6)
            print(f"[ranker] matrix chain={chain} origins={origin_index + 1}/{len(origins)} rows={writer.position} rate={rate:.0f}/s")
    stats = {
        "episodes": len(episodes),
        "groups_with_reachable_positive": reached_episodes,
        "label_total": label_total,
        "reachable_labels": reachable_total,
        "candidate_recall": reachable_total / max(label_total, 1),
        "history_strata": {key: {"labels": value[0], "reachable": value[1], "recall": value[1] / max(value[0], 1)} for key, value in history_strata.items()},
        "rows_per_second": writer.position / max(time.time() - started, 1e-6),
    }
    matrix = writer.finish(prefix, stats)
    register_artifact(cache, {
        "name": f"RateBeer LambdaRank {chain} feature matrix",
        "path": str(prefix.relative_to(cache)),
        "description": f"Temporally censored 90-feature matrix with 250 source-balanced negatives per reachable episode; rows={len(matrix.labels)}.",
        "content_key": key,
        "rebuild_hint": "Run the full candidate; remove the matrix prefix files to force a rebuild.",
    })
    return matrix


def lgb_parameters() -> dict:
    return {
        "objective": "lambdarank",
        "metric": "None",
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 63,
        "device_type": "gpu",
        "gpu_use_dp": False,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "verbosity": -1,
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "data_random_seed": 1337,
    }


def group_average_precision(scores: np.ndarray, labels: np.ndarray, groups: np.ndarray, total_labels: np.ndarray) -> np.ndarray:
    values = np.zeros(len(groups), dtype=np.float64)
    position = 0
    for index, (size, denominator) in enumerate(zip(groups, total_labels)):
        stop = position + int(size)
        order = np.argsort(-scores[position:stop], kind="stable")[:10]
        hits = labels[position:stop][order]
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        values[index] = float(np.sum(precision * hits) / min(int(denominator), 10))
        position = stop
    return values


@dataclass
class FullFoldMatrix:
    features: np.memmap
    labels: np.ndarray
    heuristic: np.ndarray
    groups: np.ndarray
    total_labels: np.ndarray


def heuristic_scores(items: np.ndarray, context: UserContext, sources: SourceBundle) -> np.ndarray:
    scores = np.zeros(len(items), dtype=np.float32)
    ranks = sources.pop_rank[items]
    present = ranks > 0
    if np.any(present):
        scale = max(float(sources.pop_scores[0] - sources.pop_scores[-1]), 1e-6)
        scores[present] = (sources.pop_scores[ranks[present].astype(np.int64) - 1] - sources.pop_scores[-1]) / scale
    for index, item in enumerate(items):
        info = context.pair_info.get(int(item))
        if info is not None:
            _, best, mean, latest, age, rank = info
            scores[index] += 0.9 + 0.55 * max(0.0, best - 3.5) + 0.2 * max(0.0, mean - 3.5) + 0.35 * math.exp(-math.log(2) * age / 45)
            scores[index] += latest + (1.0 / math.log2(rank + 1) if rank <= 500 else 0.0)
    return scores


def load_full_fold(prefix: Path) -> FullFoldMatrix:
    metadata = json.loads(prefix.with_suffix(".json").read_text())
    shape = (int(metadata["rows"]), len(FEATURE_NAMES))
    features = np.memmap(prefix.with_suffix(".features.dat"), dtype=np.float32, mode="r", shape=shape)
    heuristic = np.load(prefix.with_suffix(".heuristic.npy"), mmap_mode="r")
    if metadata.get("heuristic_version") != "direct_rating_v2":
        heuristic = np.asarray(heuristic).copy() + features[:, 62] + features[:, 77]
    return FullFoldMatrix(
        features,
        np.load(prefix.with_suffix(".labels.npy"), mmap_mode="r"),
        heuristic,
        np.load(prefix.with_suffix(".groups.npy")),
        np.load(prefix.with_suffix(".total_labels.npy")),
    )


def build_full_fold_matrix(chain: str, evaluation_origin: int, episodes: pd.DataFrame, events: GlobalEvents, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore, bm25_store: BM25Store, cache: Path, fingerprint: str) -> FullFoldMatrix:
    day = dt.datetime.fromtimestamp(evaluation_origin, tz=dt.timezone.utc).date().isoformat()
    key = f"ratebeer_full_fold_{chain}_{day}_{fingerprint}_v3_last30_f90"
    directory = cache / "temporal_ranker_lane0" / "folds"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / key
    if prefix.with_suffix(".json").exists() and prefix.with_suffix(".features.dat").exists():
        return load_full_fold(prefix)
    origin_mask = episodes["origin"].map(timestamp_seconds) == evaluation_origin
    frame = episodes[origin_mask].reset_index(drop=True)
    timeline = GlobalTimeline(events, static)
    favorite_timeline = FavoriteTimeline(events)
    timeline.advance(evaluation_origin)
    favorite_timeline.advance(evaluation_origin)
    popularity = timeline.popularity_context()
    bm25 = bm25_store.get(evaluation_origin)
    maximum_rows = len(frame) * 3500
    feature_path = prefix.with_suffix(".features.dat")
    feature_store = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(maximum_rows, len(FEATURE_NAMES)))
    label_store = np.empty(maximum_rows, dtype=np.uint8)
    heuristic_store = np.empty(maximum_rows, dtype=np.float32)
    groups = []
    totals = []
    position = 0
    for row in frame.itertuples(index=False):
        user = int(row.user_id)
        labels = set(int(value) for value in row.labels)
        context = make_user_context(user, evaluation_origin, ratings, favorites, places, static)
        sources = make_sources(user, context, popularity, bm25, favorite_timeline, static)
        items = full_candidates(sources, context.favorite_set)
        stop = position + len(items)
        feature_store[position:stop] = feature_matrix(items, evaluation_origin, context, sources, timeline, static)
        label_store[position:stop] = np.asarray([int(item) in labels for item in items], dtype=np.uint8)
        heuristic_store[position:stop] = heuristic_scores(items, context, sources)
        groups.append(len(items))
        totals.append(len(labels))
        position = stop
    feature_store.flush()
    del feature_store
    os.truncate(feature_path, position * len(FEATURE_NAMES) * np.dtype(np.float32).itemsize)
    np.save(prefix.with_suffix(".labels.npy"), label_store[:position])
    np.save(prefix.with_suffix(".heuristic.npy"), heuristic_store[:position])
    np.save(prefix.with_suffix(".groups.npy"), np.asarray(groups, dtype=np.int32))
    np.save(prefix.with_suffix(".total_labels.npy"), np.asarray(totals, dtype=np.int16))
    prefix.with_suffix(".json").write_text(json.dumps({"rows": position, "groups": len(groups), "heuristic_version": "direct_rating_v2"}, indent=2))
    register_artifact(cache, {
        "name": f"RateBeer full-candidate purged fold {chain} {day}",
        "path": str(prefix.relative_to(cache)),
        "description": "Complete inference candidate features and heuristic scores for unbiased purged-fold selection.",
        "content_key": key,
        "rebuild_hint": "Run full round selection for the matching chain and origin.",
    })
    print(f"[ranker] full fold chain={chain} origin={day} groups={len(groups)} rows={position}")
    return load_full_fold(prefix)


def rank_percentiles(scores: np.ndarray, items: np.ndarray) -> np.ndarray:
    order = np.lexsort((items, -scores))
    values = np.empty(len(scores), dtype=np.float32)
    if len(scores) <= 1:
        values[:] = 1
    else:
        values[order] = 1.0 - np.arange(len(scores), dtype=np.float32) / (len(scores) - 1)
    return values


def blended_average_precision(model_scores: np.ndarray, fold: FullFoldMatrix, model_weight: float) -> np.ndarray:
    values = np.zeros(len(fold.groups), dtype=np.float64)
    position = 0
    for index, (size, denominator) in enumerate(zip(fold.groups, fold.total_labels)):
        stop = position + int(size)
        items = np.arange(int(size), dtype=np.int32)
        model_rank = rank_percentiles(model_scores[position:stop], items)
        heuristic_rank = rank_percentiles(fold.heuristic[position:stop], items)
        scores = model_weight * model_rank + (1 - model_weight) * heuristic_rank
        order = np.argsort(-scores, kind="stable")[:10]
        hits = fold.labels[position:stop][order]
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        values[index] = float(np.sum(precision * hits) / min(int(denominator), 10))
        position = stop
    return values


def select_rounds(chain: str, matrix: TrainingMatrix, episodes: pd.DataFrame, events: GlobalEvents, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore, bm25_store: BM25Store, cache: Path, fingerprint: str, debug: bool) -> tuple[int, float, dict]:
    if debug:
        return 50, 0.5, {"selected_rounds": 50, "model_weight": 0.5, "folds": []}
    unique_origins = np.unique(matrix.origins)
    fold_indices = [13, 15] if chain == "A" else [49, 73]
    fold_indices = [min(index, len(unique_origins) - 1) for index in fold_indices]
    candidates = (300, 600, 900)
    blend_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    results = {(rounds, weight): [] for rounds in candidates for weight in blend_weights}
    per_row = {(rounds, weight): [] for rounds in candidates for weight in blend_weights}
    boundaries = np.concatenate(([0], np.cumsum(matrix.groups)))
    fold_records = []
    for fold_index in fold_indices:
        evaluation_origin = int(unique_origins[fold_index])
        training_limit = evaluation_origin - 90 * 86400
        training_groups = int(np.searchsorted(matrix.origins, training_limit, side="right"))
        evaluation_start = int(np.searchsorted(matrix.origins, evaluation_origin, side="left"))
        evaluation_stop = int(np.searchsorted(matrix.origins, evaluation_origin, side="right"))
        if training_groups == 0 or evaluation_stop == evaluation_start:
            continue
        train_stop = int(boundaries[training_groups])
        dataset = lgb.Dataset(
            matrix.features[:train_stop], label=matrix.labels[:train_stop], weight=matrix.weights[:train_stop],
            group=matrix.groups[:training_groups], feature_name=FEATURE_NAMES, params={"max_bin": 63}, free_raw_data=True,
        )
        model = lgb.train(lgb_parameters(), dataset, num_boost_round=900, callbacks=[lgb.log_evaluation(0)])
        fold = build_full_fold_matrix(chain, evaluation_origin, episodes, events, ratings, favorites, places, static, bm25_store, cache, fingerprint)
        record = {"evaluation_origin": str(pd.Timestamp(evaluation_origin, unit="s")), "training_groups": training_groups, "evaluation_groups": len(fold.groups), "scores": {}}
        for rounds in candidates:
            predictions = model.predict(fold.features, num_iteration=rounds)
            for weight in blend_weights:
                values = blended_average_precision(predictions, fold, weight)
                results[(rounds, weight)].append(float(np.mean(values)))
                per_row[(rounds, weight)].append(values)
                record["scores"][f"{rounds}:{weight}"] = float(np.mean(values))
        fold_records.append(record)
        del model, dataset, fold
    summary = {}
    for rounds in candidates:
        for weight in blend_weights:
            key = (rounds, weight)
            rows = np.concatenate(per_row[key]) if per_row[key] else np.asarray([0.0])
            summary[key] = {
            "mean": float(np.mean(results[key])) if results[key] else 0.0,
            "worst": float(np.min(results[key])) if results[key] else 0.0,
            "standard_error": float(np.std(rows, ddof=1) / math.sqrt(len(rows))) if len(rows) > 1 else 0.0,
            }
    best_mean = max(value["mean"] for value in summary.values())
    eligible = []
    for key, value in summary.items():
        gap = best_mean - value["mean"]
        if gap <= 2 * value["standard_error"]:
            eligible.append(key)
    selected_rounds, selected_weight = min(eligible, key=lambda value: (value[1], value[0])) if eligible else max(summary, key=lambda value: (summary[value]["mean"], summary[value]["worst"], -value[1], -value[0]))
    diagnostics = {
        "selected_rounds": selected_rounds,
        "model_weight": selected_weight,
        "round_blend_candidates": {f"{key[0]}:{key[1]}": value for key, value in summary.items()},
        "folds": fold_records,
    }
    print(f"[ranker] full-candidate selection chain={chain} rounds={selected_rounds} model_weight={selected_weight}")
    return selected_rounds, selected_weight, diagnostics


def train_model(matrix: TrainingMatrix, rounds: int) -> lgb.Booster:
    dataset = lgb.Dataset(
        matrix.features, label=matrix.labels, weight=matrix.weights, group=matrix.groups,
        feature_name=FEATURE_NAMES, params={"max_bin": 63}, free_raw_data=True,
    )
    model = lgb.train(lgb_parameters(), dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    return model


def full_candidates(sources: SourceBundle, excluded: set[int], cap: int = 3500) -> np.ndarray:
    values: list[int] = []
    used = set(excluded)
    for source in sources.arrays():
        for raw in source:
            item = int(raw)
            if item not in used:
                used.add(item)
                values.append(item)
    if len(values) > cap:
        values = values[:cap]
    return np.asarray(values, dtype=np.int32)


def predict_queries(model: lgb.Booster, model_weight: float, table: pd.DataFrame, timeline: GlobalTimeline, favorite_timeline: FavoriteTimeline, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore, bm25_store: BM25Store, cache: Path, chain: str, floor: np.ndarray, debug: bool) -> np.ndarray:
    result = floor.copy()
    limit = min(50, len(table)) if debug else len(table)
    origin = timestamp_seconds(table["timestamp"].iloc[0])
    timeline.advance(origin)
    favorite_timeline.advance(origin)
    popularity = timeline.popularity_context()
    bm25 = bm25_store.get(origin)
    directory = cache / "temporal_ranker_lane0" / "inference"
    directory.mkdir(parents=True, exist_ok=True)
    feature_path = directory / f"inference_{chain}_{os.getpid()}.dat"
    max_rows = limit * 3500
    feature_store = np.memmap(feature_path, dtype=np.float32, mode="w+", shape=(max_rows, len(FEATURE_NAMES)))
    item_rows: list[np.ndarray] = []
    heuristic_rows: list[np.ndarray] = []
    offsets = [0]
    for index, row in enumerate(table.iloc[:limit].itertuples(index=False)):
        user = int(row.user_id)
        user_context = make_user_context(user, origin, ratings, favorites, places, static)
        sources = make_sources(user, user_context, popularity, bm25, favorite_timeline, static)
        items = full_candidates(sources, user_context.favorite_set)
        features = feature_matrix(items, origin, user_context, sources, timeline, static)
        start = offsets[-1]
        feature_store[start:start + len(items)] = features
        item_rows.append(items)
        heuristic_rows.append(heuristic_scores(items, user_context, sources))
        offsets.append(start + len(items))
    feature_store.flush()
    total = offsets[-1]
    predictions = np.empty(total, dtype=np.float32)
    for start in range(0, total, 200_000):
        stop = min(start + 200_000, total)
        predictions[start:stop] = model.predict(feature_store[start:stop])
    for index, items in enumerate(item_rows):
        start, stop = offsets[index], offsets[index + 1]
        model_rank = rank_percentiles(predictions[start:stop], items)
        heuristic_rank = rank_percentiles(heuristic_rows[index], items)
        scores = model_weight * model_rank + (1 - model_weight) * heuristic_rank
        order = np.lexsort((items, -scores))
        ranked = [int(value) for value in items[order[:10]]]
        if len(ranked) < 10:
            used = set(ranked)
            for item in popularity.popular_ids:
                value = int(item)
                if value not in used:
                    used.add(value)
                    ranked.append(value)
                    if len(ranked) == 10:
                        break
        result[index] = ranked[:10]
    del feature_store
    feature_path.unlink(missing_ok=True)
    print(f"[ranker] inference chain={chain} queries={limit} candidates={total}")
    return result


def run_chain(chain: str, episodes: pd.DataFrame, query_table: pd.DataFrame, floor: np.ndarray, events: GlobalEvents, ratings: GroupedEvents, favorites: GroupedEvents, places: GroupedEvents, static: StaticStore, bm25_store: BM25Store, cache: Path, fingerprint: str, debug: bool) -> tuple[np.ndarray, dict]:
    timeline = GlobalTimeline(events, static)
    favorite_timeline = FavoriteTimeline(events)
    matrix = build_training_matrix(chain, episodes, timeline, favorite_timeline, ratings, favorites, places, static, bm25_store, cache, fingerprint, debug)
    rounds, model_weight, selection = select_rounds(chain, matrix, episodes, events, ratings, favorites, places, static, bm25_store, cache, fingerprint, debug)
    started = time.time()
    model = train_model(matrix, rounds)
    print(f"[ranker] trained chain={chain} rows={len(matrix.labels)} groups={len(matrix.groups)} rounds={rounds} elapsed={time.time() - started:.1f}s")
    predictions = predict_queries(model, model_weight, query_table, timeline, favorite_timeline, ratings, favorites, places, static, bm25_store, cache, chain, floor, debug)
    diagnostics = {"matrix": matrix.stats, "selection": selection, "rounds": rounds, "model_weight": model_weight}
    del model, matrix
    return predictions, diagnostics


def run_temporal_ranker(connection: duckdb.DuckDBPyConnection, db_dir: Path, task_dir: Path, val_table: pd.DataFrame, test_table: pd.DataFrame, val_floor: np.ndarray, test_floor: np.ndarray, debug: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    started = time.time()
    cache = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    fingerprint = database_fingerprint(db_dir)
    episodes_a = make_episodes(connection, db_dir, cache, "A", "2018-02-14", "2018-06-03", fingerprint)
    episodes_b = make_episodes(connection, db_dir, cache, "B", "2018-05-09", "2019-10-02", fingerprint)
    users = np.unique(np.concatenate([
        episodes_a["user_id"].to_numpy(np.int64), episodes_b["user_id"].to_numpy(np.int64),
        val_table["user_id"].to_numpy(np.int64), test_table["user_id"].to_numpy(np.int64),
    ]))
    end_time = timestamp_seconds(test_table["timestamp"].max())
    static = StaticStore(connection, db_dir)
    ratings, favorites, places = load_user_events(connection, db_dir, users, end_time)
    seed_items = np.unique(ratings.items[ratings.scores >= 3.8])
    bm25_store = BM25Store(connection, db_dir, cache, seed_items, fingerprint)
    events = GlobalEvents(connection, db_dir)
    print(f"[ranker] setup users={len(users)} user_ratings={len(ratings.items)} elapsed={time.time() - started:.1f}s")
    val_predictions, diagnostics_a = run_chain("A", episodes_a, val_table, val_floor, events, ratings, favorites, places, static, bm25_store, cache, fingerprint, debug)
    print(f"[ranker] Model A validation predictions frozen elapsed={time.time() - started:.1f}s")
    test_predictions, diagnostics_b = run_chain("B", episodes_b, test_table, test_floor, events, ratings, favorites, places, static, bm25_store, cache, fingerprint, debug)
    diagnostics = {
        "feature_count": len(FEATURE_NAMES),
        "availability_legal_rows": 0,
        "model_a": diagnostics_a,
        "model_b": diagnostics_b,
        "elapsed_seconds": time.time() - started,
    }
    return val_predictions, test_predictions, diagnostics
