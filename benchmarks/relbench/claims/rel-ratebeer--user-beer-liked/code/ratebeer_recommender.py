import hashlib
import json
import math
import os
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


SOURCE_NAMES = (
    "global30",
    "global90",
    "global365",
    "brewer",
    "style",
    "parent",
    "country",
    "covis",
    "als",
    "bm25_knn",
    "new_release",
    "favorite",
    "availability",
    "geographic",
    "fallback",
)


SOURCE_QUOTAS = {
    "global30": 65,
    "global90": 100,
    "global365": 65,
    "brewer": 150,
    "style": 220,
    "parent": 80,
    "country": 40,
    "covis": 100,
    "als": 100,
    "bm25_knn": 60,
    "new_release": 70,
    "favorite": 25,
    "availability": 30,
    "geographic": 35,
    "fallback": 60,
}


RRF_WEIGHTS = np.asarray((3.0, 2.0, 1.0, 3.0, 2.5, 0.5, 0.5, 1.5, 1.5, 0.5, 1.0, 0.5, 0.25, 0.5, 0.5), dtype=np.float32)


@dataclass
class Settings:
    debug: bool
    candidate_cap: int
    anchors: int
    negatives: int
    factors: int
    iterations: int
    regularization: float
    maximum_rounds: int


@dataclass
class Metadata:
    n_items: int
    brewer: np.ndarray
    style: np.ndarray
    parent: np.ndarray
    country: np.ndarray
    alcohol: np.ndarray
    ibu: np.ndarray
    seasonal: np.ndarray
    one_off: np.ndarray
    retired: np.ndarray
    created_day: np.ndarray


@dataclass
class BeerState:
    origin: datetime
    values: np.ndarray
    global_lists: dict
    brewer_top: dict
    style_top: dict
    parent_top: dict
    country_top: dict
    new_top: list
    favorite_count: np.ndarray
    availability_count: np.ndarray
    availability_spread: np.ndarray
    availability_country: dict


@dataclass
class UserProfile:
    user_id: int
    rated: set
    anchors: np.ndarray
    anchor_days: np.ndarray
    brewer_count: dict
    brewer_last: dict
    style_count: dict
    style_last: dict
    parent_count: dict
    parent_last: dict
    country_count: dict
    country_last: dict
    preferred_brewers: list
    preferred_styles: list
    preferred_parents: list
    preferred_countries: list
    favorite: set
    geo_countries: list
    features: np.ndarray
    factor_vector: np.ndarray
    knn_vector: np.ndarray


@dataclass
class ChannelModel:
    reference: datetime
    factor: np.ndarray
    covis_path: Path | None


@dataclass
class OriginData:
    origin: datetime
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    full_group_scores: list
    diagnostics: dict


class Timer:
    def __init__(self):
        self.started = time.time()
        self.phase = self.started

    def log(self, name):
        now = time.time()
        print(f"[ratebeer] phase={name} phase_seconds={now-self.phase:.1f} elapsed_seconds={now-self.started:.1f}", flush=True)
        self.phase = now


def settings(debug):
    if debug:
        return Settings(True, 260, 16, 90, 96, 15, 0.05, 100)
    return Settings(False, 800, 50, 250, 96, 15, 0.05, 700)


def paths():
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    return root, root / "db", root / "tasks" / os.environ["RELBENCH_TASK"]


def connection():
    con = duckdb.connect()
    con.execute("SET threads=11")
    con.execute("SET memory_limit='80GB'")
    con.execute("SET preserve_insertion_order=false")
    return con


def integer_array(series, fill=-1, dtype=np.int32):
    return series.to_numpy(dtype=dtype, na_value=fill)


def load_metadata(con, db_dir):
    beers = con.execute(
        f"SELECT beer_id, brewer_id, style_id, alcohol_pct, ibu, is_seasonal, is_one_off, is_retired, created_at FROM read_parquet('{db_dir / 'beers.parquet'}') ORDER BY beer_id"
    ).df()
    styles = con.execute(
        f"SELECT style_id, parent_style_id FROM read_parquet('{db_dir / 'beer_styles.parquet'}')"
    ).df()
    brewers = con.execute(
        f"SELECT brewer_id, country_id FROM read_parquet('{db_dir / 'brewers.parquet'}')"
    ).df()
    n_items = int(beers["beer_id"].max()) + 1
    brewer = np.full(n_items, -1, dtype=np.int32)
    style = np.full(n_items, -1, dtype=np.int32)
    alcohol = np.full(n_items, -1, dtype=np.int16)
    ibu = np.full(n_items, -1, dtype=np.int16)
    seasonal = np.zeros(n_items, dtype=np.int8)
    one_off = np.zeros(n_items, dtype=np.int8)
    retired = np.zeros(n_items, dtype=np.int8)
    created_day = np.full(n_items, -100000, dtype=np.int32)
    idx = integer_array(beers["beer_id"], dtype=np.int64)
    brewer[idx] = integer_array(beers["brewer_id"])
    style[idx] = integer_array(beers["style_id"])
    av = beers["alcohol_pct"].fillna(-1).to_numpy(dtype=np.float32)
    iv = beers["ibu"].fillna(-1).to_numpy(dtype=np.float32)
    alcohol[idx] = np.where(av < 0, -1, np.minimum(20, np.floor(av / 1.5))).astype(np.int16)
    ibu[idx] = np.where(iv < 0, -1, np.minimum(20, np.floor(iv / 10))).astype(np.int16)
    seasonal[idx] = beers["is_seasonal"].fillna(False).to_numpy(dtype=np.int8)
    one_off[idx] = beers["is_one_off"].fillna(False).to_numpy(dtype=np.int8)
    retired[idx] = beers["is_retired"].fillna(False).to_numpy(dtype=np.int8)
    created_day[idx] = (pd.to_datetime(beers["created_at"]).astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int32)
    style_parent = dict(zip(integer_array(styles["style_id"]), integer_array(styles["parent_style_id"])))
    brewer_country = dict(zip(integer_array(brewers["brewer_id"]), integer_array(brewers["country_id"])))
    parent = np.asarray([style_parent.get(int(x), -1) for x in style], dtype=np.int32)
    country = np.asarray([brewer_country.get(int(x), -1) for x in brewer], dtype=np.int32)
    return Metadata(n_items, brewer, style, parent, country, alcohol, ibu, seasonal, one_off, retired, created_day)


def origin_day(origin):
    return int(pd.Timestamp(origin).value // 86_400_000_000_000)


def top_map(keys, scores, valid, limit):
    ids = np.flatnonzero(valid)
    if len(ids) == 0:
        return {}
    frame = pd.DataFrame({"key": keys[ids], "beer": ids, "score": scores[ids]})
    frame = frame[frame["key"] >= 0].sort_values(["key", "score", "beer"], ascending=[True, False, True])
    frame = frame.groupby("key", sort=False).head(limit)
    return {int(k): list(zip(g["beer"].astype(int), g["score"].astype(float))) for k, g in frame.groupby("key", sort=False)}


def ranked_global(score, eligible, limit):
    ids = np.flatnonzero(eligible)
    if len(ids) == 0:
        return []
    order = np.lexsort((ids, -score[ids]))[:limit]
    return [(int(ids[i]), float(score[ids[i]])) for i in order]


def build_beer_state(con, db_dir, meta, origin):
    day = origin_day(origin)
    query = f"""
        SELECT beer_id,
            count(*) AS ratings,
            count(*) FILTER (WHERE total_score >= 4.0) AS likes,
            count(*) FILTER (WHERE total_score >= 4.0 AND created_at > TIMESTAMP '{origin}' - INTERVAL 30 DAY) AS like30,
            count(*) FILTER (WHERE total_score >= 4.0 AND created_at > TIMESTAMP '{origin}' - INTERVAL 90 DAY) AS like90,
            count(*) FILTER (WHERE total_score >= 4.0 AND created_at > TIMESTAMP '{origin}' - INTERVAL 365 DAY) AS like365,
            count(*) FILTER (WHERE created_at > TIMESTAMP '{origin}' - INTERVAL 30 DAY) AS rating30,
            count(*) FILTER (WHERE created_at > TIMESTAMP '{origin}' - INTERVAL 90 DAY) AS rating90,
            count(*) FILTER (WHERE created_at > TIMESTAMP '{origin}' - INTERVAL 365 DAY) AS rating365,
            approx_count_distinct(user_id) AS users,
            avg(total_score) AS mean_score,
            epoch(min(created_at))/86400.0 AS first_day,
            epoch(max(created_at))/86400.0 AS last_day
        FROM read_parquet('{db_dir / 'beer_ratings.parquet'}')
        WHERE created_at <= TIMESTAMP '{origin}'
        GROUP BY beer_id
    """
    stats = con.execute(query).fetchnumpy()
    values = np.zeros((meta.n_items, 12), dtype=np.float32)
    ids = stats["beer_id"].astype(np.int64)
    for j, name in enumerate(("ratings", "likes", "like30", "like90", "like365", "rating30", "rating90", "rating365", "users", "mean_score", "first_day", "last_day")):
        values[ids, j] = np.asarray(stats[name], dtype=np.float32)
    base_rate = float(values[:, 1].sum() / max(1.0, values[:, 0].sum()))
    bayes = (values[:, 1] + 20.0 * base_rate) / (values[:, 0] + 20.0)
    momentum = values[:, 2] / (1.0 + values[:, 4] / 12.0)
    score30 = np.log1p(values[:, 2]) + 0.15 * np.log1p(values[:, 3]) + 0.08 * bayes
    score90 = np.log1p(values[:, 3]) + 0.20 * np.log1p(values[:, 4]) + 0.10 * bayes + 0.03 * momentum
    score365 = np.log1p(values[:, 4]) + 0.15 * np.log1p(values[:, 1]) + 0.12 * bayes
    eligible = (meta.created_day <= day) & (values[:, 0] > 0)
    global_lists = {
        "global30": ranked_global(score30, eligible, 5000),
        "global90": ranked_global(score90, eligible, 5000),
        "global365": ranked_global(score365, eligible, 5000),
    }
    group_score = score90 + 0.25 * score365
    brewer_top = top_map(meta.brewer, group_score, eligible, 100)
    style_top = top_map(meta.style, group_score, eligible, 100)
    parent_top = top_map(meta.parent, group_score, eligible, 100)
    country_top = top_map(meta.country, group_score, eligible, 100)
    new_eligible = eligible & (meta.created_day >= day - 365)
    new_score = score30 + 0.5 * score90 + 0.2 * bayes + 0.08 * np.maximum(0.0, 365 - (day - meta.created_day)) / 365
    new_top = ranked_global(new_score, new_eligible, 2000)
    favorite_count = np.zeros(meta.n_items, dtype=np.float32)
    fav = con.execute(
        f"SELECT beer_id, count(*) n FROM read_parquet('{db_dir / 'favorites.parquet'}') WHERE created_at <= TIMESTAMP '{origin}' GROUP BY beer_id"
    ).fetchnumpy()
    if len(fav["beer_id"]):
        favorite_count[fav["beer_id"].astype(np.int64)] = np.asarray(fav["n"], dtype=np.float32)
    availability_count = np.zeros(meta.n_items, dtype=np.float32)
    availability_spread = np.zeros(meta.n_items, dtype=np.float32)
    av = con.execute(
        f"SELECT beer_id, count(*) n, approx_count_distinct(coalesce(country_id,-1)) spread FROM read_parquet('{db_dir / 'availability.parquet'}') WHERE beer_id IS NOT NULL AND created_at <= TIMESTAMP '{origin}' GROUP BY beer_id"
    ).fetchnumpy()
    if len(av["beer_id"]):
        aidx = av["beer_id"].astype(np.int64)
        availability_count[aidx] = np.asarray(av["n"], dtype=np.float32)
        availability_spread[aidx] = np.asarray(av["spread"], dtype=np.float32)
    avc = con.execute(
        f"SELECT coalesce(a.country_id,p.country_id) country_id, a.beer_id, count(*) n FROM read_parquet('{db_dir / 'availability.parquet'}') a LEFT JOIN read_parquet('{db_dir / 'places.parquet'}') p USING(place_id) WHERE a.beer_id IS NOT NULL AND a.created_at <= TIMESTAMP '{origin}' GROUP BY 1,2 QUALIFY row_number() OVER(PARTITION BY coalesce(a.country_id,p.country_id) ORDER BY n DESC,a.beer_id)<=100"
    ).df()
    availability_country = {}
    if len(avc):
        avc = avc.dropna(subset=["country_id"])
        availability_country = {int(k): list(zip(g["beer_id"].astype(int), g["n"].astype(float))) for k, g in avc.groupby("country_id", sort=False)}
    return BeerState(origin, values, global_lists, brewer_top, style_top, parent_top, country_top, new_top, favorite_count, availability_count, availability_spread, availability_country)


def register_artifact(cache, name, path, description, key, hint):
    import fcntl
    registry = cache / "artifacts.json"
    lock = cache / ".artifacts.lock"
    with lock.open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except Exception:
            records = []
        rel = str(path.relative_to(cache))
        if not any(x.get("path") == rel for x in records):
            records.append({"name": name, "path": rel, "description": description, "content_key": key, "rebuild_hint": hint})
            temp = registry.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(records, indent=2))
            os.replace(temp, registry)
        fcntl.flock(guard, fcntl.LOCK_UN)


def build_covis(con, db_dir, cache, reference):
    stamp = reference.strftime("%Y%m%d")
    key = f"ratebeer-directed-next-like-v2-{stamp}"
    target = cache / f"lane0_covis_{stamp}_v2.parquet"
    if not target.exists():
        query = f"""
            COPY (
                WITH seq AS (
                    SELECT user_id, beer_id AS x, created_at AS tx,
                        lead(beer_id,1) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS y1,
                        lead(created_at,1) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS ty1,
                        lead(beer_id,2) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS y2,
                        lead(created_at,2) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS ty2,
                        lead(beer_id,3) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS y3,
                        lead(created_at,3) OVER(PARTITION BY user_id ORDER BY created_at,rating_id) AS ty3
                    FROM read_parquet('{db_dir / 'beer_ratings.parquet'}')
                    WHERE total_score >= 4.0 AND created_at <= TIMESTAMP '{reference}'
                ), pairs AS (
                    SELECT x,y1 AS y,tx,ty1 AS ty FROM seq
                    UNION ALL SELECT x,y2,tx,ty2 FROM seq
                    UNION ALL SELECT x,y3,tx,ty3 FROM seq
                ), agg AS (
                    SELECT x,y,count(*) AS n,
                        sum(exp(-datediff('day',tx,ty)/14.0)+exp(-datediff('day',tx,ty)/45.0)+exp(-datediff('day',tx,ty)/180.0)) AS score
                    FROM pairs
                    WHERE y IS NOT NULL AND x <> y AND ty <= tx + INTERVAL 90 DAY
                    GROUP BY x,y
                )
                SELECT x,y,n,score FROM agg
                QUALIFY row_number() OVER(PARTITION BY x ORDER BY score DESC,n DESC,y)<=40
            ) TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
        con.execute(query)
    register_artifact(cache, f"directed co-visitation {stamp}", target, "Top directed next-like neighbors with 14/45/180-day lag weights, censored at the reference date", key, "Run main.py full; generated from sanitized beer_ratings")
    return target


def build_factor(con, db_dir, cache, meta, reference, cfg):
    stamp = reference.strftime("%Y%m%d")
    key = f"ratebeer-bm25-als-f{cfg.factors}-i{cfg.iterations}-r{cfg.regularization}-{stamp}-v2"
    target = cache / f"lane0_item_factor_{stamp}_v2.npy"
    if target.exists():
        return np.load(target, mmap_mode="r")
    import scipy.sparse as sp
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight
    try:
        from implicit.gpu.als import AlternatingLeastSquares as GPUAlternatingLeastSquares
        GPUAlternatingLeastSquares(factors=4, iterations=1)
        print("[ratebeer] implicit_gpu_smoke=available", flush=True)
    except Exception as exc:
        print(f"[ratebeer] implicit_gpu_smoke=unavailable fallback=cpu error={type(exc).__name__}", flush=True)
    arr = con.execute(
        f"SELECT DISTINCT user_id,beer_id FROM read_parquet('{db_dir / 'beer_ratings.parquet'}') WHERE total_score>=4.0 AND created_at<=TIMESTAMP '{reference}'"
    ).fetchnumpy()
    users = arr["user_id"].astype(np.int32)
    items = arr["beer_id"].astype(np.int32)
    n_users = int(users.max()) + 1
    matrix = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)), shape=(n_users, meta.n_items))
    weighted = bm25_weight(matrix, K1=100, B=0.8).tocsr().astype(np.float32)
    model = AlternatingLeastSquares(
        factors=cfg.factors,
        regularization=cfg.regularization,
        iterations=cfg.iterations,
        random_state=1337,
        num_threads=11,
    )
    model.fit(weighted, show_progress=False)
    factor = np.asarray(model.item_factors, dtype=np.float32)
    norm = np.linalg.norm(factor, axis=1, keepdims=True)
    factor /= np.maximum(norm, 1e-8)
    np.save(target, factor)
    register_artifact(cache, f"BM25 ALS item factors {stamp}", target, "Likes-only BM25-weighted implicit ALS item factors, 96 factors and 15 CPU iterations", key, "Install implicit 0.7.2 and run main.py full")
    return np.load(target, mmap_mode="r")


def build_channel_model(con, db_dir, cache, meta, reference, cfg):
    if cfg.debug:
        return ChannelModel(reference, np.zeros((meta.n_items, cfg.factors), dtype=np.float32), None)
    covis_path = build_covis(con, db_dir, cache, reference)
    factor = build_factor(con, db_dir, cache, meta, reference, cfg)
    return ChannelModel(reference, factor, covis_path)


def group_counts(values, days):
    if len(values) == 0:
        return {}, {}, []
    unique, count = np.unique(values[values >= 0], return_counts=True)
    counts = {int(k): int(v) for k, v in zip(unique, count)}
    last = {}
    for key, day in zip(values, days):
        if key >= 0 and day > last.get(int(key), -100000):
            last[int(key)] = int(day)
    preferred = sorted(counts, key=lambda x: (-counts[x], -last[x], x))
    return counts, last, preferred


def entropy(values):
    values = values[values >= 0]
    if len(values) == 0:
        return 0.0
    _, counts = np.unique(values, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def recent_first(values, preferred, limit):
    output = []
    seen = set()
    for value in values:
        value = int(value)
        if value >= 0 and value not in seen:
            seen.add(value)
            output.append(value)
    for value in preferred:
        value = int(value)
        if value >= 0 and value not in seen:
            seen.add(value)
            output.append(value)
    return output[:limit]


def user_feature_vector(scores, days, liked_items, meta, day):
    n = len(scores)
    likes = scores >= 4.0
    d = np.sort(days)
    gaps = np.diff(d) if len(d) > 1 else np.asarray([0], dtype=np.float32)
    liked = liked_items
    return np.asarray([
        math.log1p(n),
        math.log1p(np.sum(days > day - 30)),
        math.log1p(np.sum(days > day - 90)),
        math.log1p(np.sum(days > day - 365)),
        float(likes.mean()) if n else 0.0,
        float(scores.mean()) if n else 0.0,
        float(np.mean(scores < 2.0)) if n else 0.0,
        float(np.mean((scores >= 2.0) & (scores < 3.0))) if n else 0.0,
        float(np.mean((scores >= 3.0) & (scores < 4.0))) if n else 0.0,
        float(np.mean((scores >= 4.0) & (scores < 4.5))) if n else 0.0,
        float(np.mean(scores >= 4.5)) if n else 0.0,
        float(day - days.max()) if n else 9999.0,
        float(gaps.mean()),
        float(gaps.std()),
        entropy(meta.style[liked]),
        entropy(meta.brewer[liked]),
        entropy(meta.country[liked]),
        float(len(np.unique(meta.style[liked])) / max(1, len(liked))),
        float(len(np.unique(meta.brewer[liked])) / max(1, len(liked))),
        math.log1p(len(liked)),
    ], dtype=np.float32)


def load_user_profiles(con, db_dir, meta, state, model, users, origin, cfg):
    user_frame = pd.DataFrame({"user_id": np.asarray(users, dtype=np.int64)})
    con.register("lane0_seed_users", user_frame)
    hist = con.execute(
        f"SELECT r.user_id,r.beer_id,r.total_score,epoch(r.created_at)/86400.0 AS day FROM read_parquet('{db_dir / 'beer_ratings.parquet'}') r JOIN lane0_seed_users s USING(user_id) WHERE r.created_at<=TIMESTAMP '{origin}' ORDER BY r.user_id,day DESC,r.beer_id"
    ).fetchnumpy()
    fav = con.execute(
        f"SELECT f.user_id,f.beer_id FROM read_parquet('{db_dir / 'favorites.parquet'}') f JOIN lane0_seed_users s USING(user_id) WHERE f.created_at<=TIMESTAMP '{origin}'"
    ).fetchnumpy()
    geo = con.execute(
        f"SELECT pr.user_id,p.country_id,count(*) n,max(epoch(pr.created_at)/86400.0) last_day FROM read_parquet('{db_dir / 'place_ratings.parquet'}') pr JOIN lane0_seed_users s USING(user_id) JOIN read_parquet('{db_dir / 'places.parquet'}') p USING(place_id) WHERE pr.created_at<=TIMESTAMP '{origin}' GROUP BY pr.user_id,p.country_id ORDER BY pr.user_id,n DESC,last_day DESC"
    ).fetchnumpy()
    favorite_map = defaultdict(set)
    for uid, beer in zip(fav["user_id"], fav["beer_id"]):
        favorite_map[int(uid)].add(int(beer))
    geo_map = defaultdict(list)
    for uid, country in zip(geo["user_id"], geo["country_id"]):
        if country is not None and not pd.isna(country):
            geo_map[int(uid)].append(int(country))
    h_users = np.asarray(np.ma.filled(hist["user_id"], -1), dtype=np.int64)
    cuts = np.flatnonzero(np.r_[True, h_users[1:] != h_users[:-1], True])
    slices = {int(h_users[cuts[i]]): (cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)} if len(h_users) else {}
    day = origin_day(origin)
    factor = model.factor
    profiles = []
    for uid in users:
        uid = int(uid)
        start, end = slices.get(uid, (0, 0))
        items = np.asarray(np.ma.filled(hist["beer_id"][start:end], -1), dtype=np.int64)
        scores = np.asarray(np.ma.filled(hist["total_score"][start:end], np.nan), dtype=np.float32)
        days = np.asarray(np.ma.filled(hist["day"][start:end], -100000), dtype=np.int32)
        valid_history = (items >= 0) & np.isfinite(scores)
        items = items[valid_history]
        scores = scores[valid_history]
        days = days[valid_history]
        liked_mask = scores >= 4.0
        liked_items_all = items[liked_mask]
        liked_days_all = days[liked_mask]
        seen = set()
        anchors = []
        anchor_days = []
        for beer, bday in zip(liked_items_all, liked_days_all):
            beer = int(beer)
            if beer not in seen:
                seen.add(beer)
                anchors.append(beer)
                anchor_days.append(int(bday))
                if len(anchors) >= cfg.anchors:
                    break
        anchors = np.asarray(anchors, dtype=np.int64)
        anchor_days = np.asarray(anchor_days, dtype=np.int32)
        bcount, blast, bpref = group_counts(meta.brewer[liked_items_all], liked_days_all)
        scount, slast, spref = group_counts(meta.style[liked_items_all], liked_days_all)
        pcount, plast, ppref = group_counts(meta.parent[liked_items_all], liked_days_all)
        ccount, clast, cpref = group_counts(meta.country[liked_items_all], liked_days_all)
        if len(anchors):
            known = anchors[np.linalg.norm(np.asarray(factor[anchors]), axis=1) > 0]
        else:
            known = anchors
        if len(known):
            kdays = anchor_days[:len(known)] if len(known) == len(anchors) else np.asarray([day] * len(known))
            weights = np.exp(-np.maximum(0, day - kdays) / 180.0).astype(np.float32)
            vector = np.average(np.asarray(factor[known]), axis=0, weights=weights).astype(np.float32)
            vector /= max(1e-8, float(np.linalg.norm(vector)))
            knn = np.asarray(factor[known[0]], dtype=np.float32).copy()
        else:
            vector = np.zeros(factor.shape[1], dtype=np.float32)
            knn = np.zeros(factor.shape[1], dtype=np.float32)
        profiles.append(UserProfile(
            uid,
            set(int(x) for x in items),
            anchors,
            anchor_days,
            bcount,
            blast,
            scount,
            slast,
            pcount,
            plast,
            ccount,
            clast,
            recent_first(meta.brewer[anchors], bpref, 12),
            recent_first(meta.style[anchors], spref, 16),
            recent_first(meta.parent[anchors], ppref, 10),
            recent_first(meta.country[anchors], cpref, 6),
            favorite_map[uid],
            geo_map[uid][:3],
            user_feature_vector(scores, days, liked_items_all, meta, day),
            vector,
            knn,
        ))
    con.unregister("lane0_seed_users")
    return profiles


def factor_recommendations(profiles, model, topk):
    if not profiles or not np.any(model.factor):
        empty = [[] for _ in profiles]
        return empty, empty
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        items = torch.as_tensor(np.asarray(model.factor), dtype=torch.float32, device=device)
        output_a = []
        output_k = []
        batch = 64 if device.type == "cuda" else 16
        for start in range(0, len(profiles), batch):
            part = profiles[start:start + batch]
            a = torch.as_tensor(np.stack([x.factor_vector for x in part]), dtype=torch.float32, device=device)
            k = torch.as_tensor(np.stack([x.knn_vector for x in part]), dtype=torch.float32, device=device)
            score_a = a @ items.T
            score_k = k @ items.T
            va, ia = torch.topk(score_a, k=topk, dim=1, sorted=True)
            vk, ik = torch.topk(score_k, k=topk, dim=1, sorted=True)
            va = va.cpu().numpy()
            ia = ia.cpu().numpy()
            vk = vk.cpu().numpy()
            ik = ik.cpu().numpy()
            output_a.extend([list(zip(row_i.astype(int), row_v.astype(float))) for row_i, row_v in zip(ia, va)])
            output_k.extend([list(zip(row_i.astype(int), row_v.astype(float))) for row_i, row_v in zip(ik, vk)])
            del score_a, score_k, va, ia, vk, ik
        del items
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return output_a, output_k
    except Exception as exc:
        print(f"[ratebeer] factor_recommendation_fallback={type(exc).__name__}", flush=True)
        import faiss
        index = faiss.IndexFlatIP(model.factor.shape[1])
        index.add(np.asarray(model.factor, dtype=np.float32))
        av = np.stack([x.factor_vector for x in profiles]).astype(np.float32)
        kv = np.stack([x.knn_vector for x in profiles]).astype(np.float32)
        da, ia = index.search(av, topk)
        dk, ik = index.search(kv, topk)
        return [list(zip(i.astype(int), d.astype(float))) for i, d in zip(ia, da)], [list(zip(i.astype(int), d.astype(float))) for i, d in zip(ik, dk)]


def load_covis(con, model, profiles):
    if model.covis_path is None:
        return {}
    anchors = np.unique(np.concatenate([p.anchors for p in profiles if len(p.anchors)])) if any(len(p.anchors) for p in profiles) else np.asarray([], dtype=np.int64)
    if len(anchors) == 0:
        return {}
    con.register("lane0_anchor_items", pd.DataFrame({"x": anchors}))
    frame = con.execute(f"SELECT c.x,c.y,c.score FROM read_parquet('{model.covis_path}') c JOIN lane0_anchor_items a USING(x) ORDER BY c.x,c.score DESC,c.y").df()
    con.unregister("lane0_anchor_items")
    return {int(k): list(zip(g["y"].astype(int), g["score"].astype(float))) for k, g in frame.groupby("x", sort=False)}


def merge_group_source(preferred, mapping, counts, limit_each):
    scores = defaultdict(float)
    for key in preferred:
        affinity = 1.0 + math.log1p(counts.get(key, 0))
        for beer, score in mapping.get(key, [])[:limit_each]:
            scores[int(beer)] = max(scores[int(beer)], affinity * float(score))
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))


def candidate_sources(profile, state, meta, covis, als, knn):
    day = origin_day(state.origin)
    covis_score = defaultdict(float)
    for anchor, aday in zip(profile.anchors, profile.anchor_days):
        anchor_weight = math.exp(-max(0, day - int(aday)) / 180.0)
        for beer, score in covis.get(int(anchor), [])[:20]:
            covis_score[int(beer)] += anchor_weight * float(score)
    geo_country = profile.geo_countries if profile.geo_countries else profile.preferred_countries
    availability = merge_group_source(geo_country[:2], state.availability_country, {x: 1 for x in geo_country}, 60)
    geographic = merge_group_source(geo_country[:2], state.country_top, {x: 1 for x in geo_country}, 60)
    favorites = [(int(x), float(state.favorite_count[int(x)] + 1)) for x in sorted(profile.favorite)]
    return {
        "global30": state.global_lists["global30"][:220],
        "global90": state.global_lists["global90"][:260],
        "global365": state.global_lists["global365"][:260],
        "brewer": merge_group_source(profile.preferred_brewers, state.brewer_top, profile.brewer_count, 55),
        "style": merge_group_source(profile.preferred_styles, state.style_top, profile.style_count, 55),
        "parent": merge_group_source(profile.preferred_parents, state.parent_top, profile.parent_count, 50),
        "country": merge_group_source(profile.preferred_countries, state.country_top, profile.country_count, 45),
        "covis": sorted(covis_score.items(), key=lambda x: (-x[1], x[0]))[:180],
        "als": als,
        "bm25_knn": knn,
        "new_release": state.new_top[:180],
        "favorite": favorites,
        "availability": availability[:100],
        "geographic": geographic[:100],
        "fallback": state.global_lists["global90"],
    }


def make_candidates(profile, sources, meta, cfg):
    selected = {}
    ranks = {}
    scores = {}
    filtered = {}
    for source_id, source in enumerate(SOURCE_NAMES):
        rows = []
        for rank, pair in enumerate(sources.get(source, []), 1):
            beer = int(pair[0])
            score = float(pair[1])
            if beer < 0 or beer >= meta.n_items or beer in profile.rated:
                continue
            rows.append((beer, score, rank))
        filtered[source] = rows
        for beer, score, rank in rows:
            key = (beer, source_id)
            ranks[key] = rank
            scores[key] = score
    positions = {s: 0 for s in SOURCE_NAMES}
    used = {s: 0 for s in SOURCE_NAMES}
    exhausted = set()
    while len(selected) < cfg.candidate_cap and len(exhausted) < len(SOURCE_NAMES):
        choices = [s for s in SOURCE_NAMES if s not in exhausted]
        source = min(choices, key=lambda s: (used[s] / SOURCE_QUOTAS[s], SOURCE_NAMES.index(s)))
        rows = filtered[source]
        pos = positions[source]
        while pos < len(rows):
            beer = rows[pos][0]
            pos += 1
            if beer in selected or meta.created_day[beer] > origin_day(state_origin_holder[0]):
                continue
            selected[beer] = len(selected)
            used[source] += 1
            break
        positions[source] = pos
        if pos >= len(rows) or used[source] >= SOURCE_QUOTAS[source]:
            exhausted.add(source)
    if len(selected) < cfg.candidate_cap:
        beer = 0
        while len(selected) < cfg.candidate_cap and beer < meta.n_items:
            if beer not in profile.rated and meta.created_day[beer] <= origin_day(state_origin_holder[0]):
                selected[beer] = len(selected)
            beer += 1
    beers = np.fromiter(selected.keys(), dtype=np.int64)
    rank_matrix = np.zeros((len(beers), len(SOURCE_NAMES)), dtype=np.float32)
    score_matrix = np.zeros((len(beers), len(SOURCE_NAMES)), dtype=np.float32)
    for i, beer in enumerate(beers):
        for source_id in range(len(SOURCE_NAMES)):
            rank = ranks.get((int(beer), source_id), 0)
            score = scores.get((int(beer), source_id), 0.0)
            rank_matrix[i, source_id] = float(rank)
            score_matrix[i, source_id] = float(score)
    return beers, rank_matrix, score_matrix


state_origin_holder = [datetime(2000, 1, 1)]


def dictionary_values(mapping, keys, default=0.0):
    return np.asarray([mapping.get(int(x), default) for x in keys], dtype=np.float32)


def build_features(profile, beers, ranks, scores, state, meta, model):
    n = len(beers)
    v = state.values[beers]
    brewer = meta.brewer[beers]
    style = meta.style[beers]
    parent = meta.parent[beers]
    country = meta.country[beers]
    br_count = dictionary_values(profile.brewer_count, brewer)
    st_count = dictionary_values(profile.style_count, style)
    pa_count = dictionary_values(profile.parent_count, parent)
    co_count = dictionary_values(profile.country_count, country)
    total_likes = max(1.0, math.expm1(float(profile.features[-1])))
    day = origin_day(state.origin)
    br_rec = day - dictionary_values(profile.brewer_last, brewer, -10000)
    st_rec = day - dictionary_values(profile.style_last, style, -10000)
    pa_rec = day - dictionary_values(profile.parent_last, parent, -10000)
    co_rec = day - dictionary_values(profile.country_last, country, -10000)
    source_present = ranks > 0
    reciprocal = np.where(source_present, 1.0 / np.maximum(1.0, ranks), 0.0)
    best_rank = np.min(np.where(source_present, ranks, 1e6), axis=1)
    best_rank[best_rank >= 1e6] = 0
    base_rate = float(v[:, 1].sum() / max(1.0, v[:, 0].sum())) if n else 0.0
    bayes = (v[:, 1] + 20.0 * base_rate) / (v[:, 0] + 20.0)
    momentum = v[:, 2] / (1.0 + v[:, 4] / 12.0)
    if np.any(profile.factor_vector):
        als_score = np.asarray(model.factor[beers]) @ profile.factor_vector
        knn_score = np.asarray(model.factor[beers]) @ profile.knn_vector
    else:
        als_score = np.zeros(n, dtype=np.float32)
        knn_score = np.zeros(n, dtype=np.float32)
    stable = np.column_stack([
        meta.alcohol[beers],
        meta.ibu[beers],
        meta.seasonal[beers],
        meta.one_off[beers],
        meta.retired[beers],
        np.maximum(0, day - meta.created_day[beers]),
    ]).astype(np.float32)
    relations = np.column_stack([
        np.log1p(state.favorite_count[beers]),
        np.log1p(state.availability_count[beers]),
        np.log1p(state.availability_spread[beers]),
        np.asarray([int(x) in profile.favorite for x in beers], dtype=np.float32),
        np.asarray([int(x) in profile.geo_countries for x in country], dtype=np.float32),
    ])
    beer_features = np.column_stack([
        np.log1p(v[:, :9]),
        v[:, 9],
        bayes,
        momentum,
        np.maximum(0, day - v[:, 10]),
        np.maximum(0, day - v[:, 11]),
    ])
    pair = np.column_stack([
        np.log1p(br_count), br_count / total_likes, br_rec,
        np.log1p(st_count), st_count / total_likes, st_rec,
        np.log1p(pa_count), pa_count / total_likes, pa_rec,
        np.log1p(co_count), co_count / total_likes, co_rec,
        als_score, knn_score, np.zeros(n, dtype=np.float32),
    ])
    source_features = np.column_stack([
        np.where(source_present, np.log1p(ranks), 0.0),
        np.sign(scores) * np.log1p(np.abs(scores)),
        source_present.sum(axis=1),
        reciprocal.sum(axis=1),
        best_rank,
    ])
    users = np.tile(profile.features, (n, 1))
    return np.nan_to_num(np.column_stack([source_features, beer_features, users, pair, stable, relations]), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


def history_bucket(profile):
    n = int(round(math.expm1(float(profile.features[0]))))
    if n <= 5:
        return "1-5"
    if n <= 25:
        return "6-25"
    if n <= 199:
        return "26-199"
    return "200+"


def diagnostic_template():
    return {
        "groups": 0,
        "labels": 0,
        "hit50": 0,
        "hit100": 0,
        "hit200": 0,
        "hit800": 0,
        "oracle_sum": 0.0,
        "channel_hits": {x: 0 for x in SOURCE_NAMES},
        "history": {x: {"groups": 0, "labels": 0, "hits": 0} for x in ("1-5", "6-25", "26-199", "200+")},
    }


def update_diagnostics(diag, profile, truth, beers, ranks):
    truth = set(int(x) for x in truth)
    diag["groups"] += 1
    diag["labels"] += len(truth)
    for k, key in ((50, "hit50"), (100, "hit100"), (200, "hit200"), (800, "hit800")):
        diag[key] += len(truth.intersection(int(x) for x in beers[:k]))
    retrieved = len(truth.intersection(int(x) for x in beers))
    diag["oracle_sum"] += retrieved / max(1, min(10, len(truth)))
    previous = set()
    for source_id, source in enumerate(SOURCE_NAMES):
        source_items = set(int(beers[i]) for i in np.flatnonzero(ranks[:, source_id] > 0))
        now = truth.intersection(source_items)
        diag["channel_hits"][source] += len(now - previous)
        previous.update(now)
    bucket = history_bucket(profile)
    diag["history"][bucket]["groups"] += 1
    diag["history"][bucket]["labels"] += len(truth)
    diag["history"][bucket]["hits"] += retrieved


def balanced_negative_indices(beers, ranks, truth, maximum):
    truth_array = np.fromiter(truth, dtype=np.int64)
    negative_mask = ~np.isin(beers, truth_array)
    chosen = set()
    per_source = max(4, maximum // (2 * len(SOURCE_NAMES)))
    for source_id in range(len(SOURCE_NAMES)):
        available = np.flatnonzero(negative_mask & (ranks[:, source_id] > 0))
        if len(available):
            positions = np.linspace(0, len(available) - 1, min(per_source, len(available)), dtype=np.int64)
            chosen.update(int(available[x]) for x in positions)
    available = np.flatnonzero(negative_mask)
    if len(chosen) < maximum and len(available):
        positions = np.linspace(0, len(available) - 1, min(maximum * 2, len(available)), dtype=np.int64)
        for position in positions:
            chosen.add(int(available[position]))
            if len(chosen) >= maximum:
                break
    return np.asarray(sorted(chosen)[:maximum], dtype=np.int64)


def finalize_diagnostics(diag, origin):
    labels = max(1, diag["labels"])
    result = {
        "origin": str(origin),
        "groups": diag["groups"],
        "labels": diag["labels"],
        "recall50": diag["hit50"] / labels,
        "recall100": diag["hit100"] / labels,
        "recall200": diag["hit200"] / labels,
        "recall800": diag["hit800"] / labels,
        "oracle_map10": diag["oracle_sum"] / max(1, diag["groups"]),
        "marginal_recall": {k: v / labels for k, v in diag["channel_hits"].items()},
        "history": {},
    }
    for key, value in diag["history"].items():
        result["history"][key] = {
            "groups": value["groups"],
            "labels": value["labels"],
            "recall800": value["hits"] / max(1, value["labels"]),
        }
    return result


def build_origin_data(con, db_dir, meta, state, model, groups, cfg, diagnostics_only=False, evaluate_full=False):
    users = groups["user_id"].to_numpy(dtype=np.int64)
    profiles = load_user_profiles(con, db_dir, meta, state, model, users, state.origin, cfg)
    covis = load_covis(con, model, profiles)
    factor_k = 450 if not cfg.debug else 50
    als_rows, knn_rows = factor_recommendations(profiles, model, factor_k)
    truth_rows = groups["beer_id"].tolist()
    x_parts = []
    y_parts = []
    sizes = []
    group_scores = []
    diag = diagnostic_template()
    state_origin_holder[0] = state.origin
    for index, profile in enumerate(profiles):
        sources = candidate_sources(profile, state, meta, covis, als_rows[index], knn_rows[index])
        beers, ranks, source_scores = make_candidates(profile, sources, meta, cfg)
        truth = set(int(x) for x in truth_rows[index])
        update_diagnostics(diag, profile, truth, beers, ranks)
        truth_array = np.fromiter(truth, dtype=np.int64)
        positives = np.flatnonzero(np.isin(beers, truth_array))
        negatives = balanced_negative_indices(beers, ranks, truth, cfg.negatives)
        chosen = np.sort(np.concatenate([positives, negatives]))
        if evaluate_full:
            full_features = build_features(profile, beers, ranks, source_scores, state, meta, model)
            features = full_features[chosen]
        else:
            full_features = None
            features = build_features(profile, beers[chosen], ranks[chosen], source_scores[chosen], state, meta, model)
        labels = np.isin(beers[chosen], truth_array).astype(np.int8)
        if labels.sum() > 0 and not diagnostics_only:
            x_parts.append(features)
            y_parts.append(labels)
            sizes.append(len(labels))
        if evaluate_full:
            group_scores.append((full_features, np.isin(beers, truth_array).astype(np.int8), beers, max(1, min(10, len(truth)))))
        else:
            group_scores.append((features, labels, beers[chosen], max(1, min(10, len(truth)))))
    width = x_parts[0].shape[1] if x_parts else build_features(profiles[0], np.asarray([state.global_lists["global90"][0][0]]), np.zeros((1, len(SOURCE_NAMES)), dtype=np.float32), np.zeros((1, len(SOURCE_NAMES)), dtype=np.float32), state, meta, model).shape[1]
    x = np.concatenate(x_parts) if x_parts else np.empty((0, width), dtype=np.float32)
    y = np.concatenate(y_parts) if y_parts else np.empty(0, dtype=np.int8)
    return OriginData(state.origin, x, y, np.asarray(sizes, dtype=np.int32), group_scores, finalize_diagnostics(diag, state.origin))


def train_ranker(data, rounds):
    x = np.concatenate([d.x for d in data])
    y = np.concatenate([d.y for d in data])
    groups = np.concatenate([d.groups for d in data])
    params = {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "lambdarank_truncation_level": 13,
        "verbosity": -1,
        "num_threads": 11,
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "deterministic": True,
        "force_col_wise": True,
    }
    dataset = lgb.Dataset(x, label=y, group=groups, free_raw_data=True)
    model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    return model


def rrf_prior(features):
    ranks = np.expm1(features[:, :len(SOURCE_NAMES)])
    present = ranks > 0
    return (present * RRF_WEIGHTS / (20.0 + ranks)).sum(axis=1)


def rank_percentiles(values, beers):
    order = np.lexsort((beers, -values))
    result = np.empty(len(values), dtype=np.float32)
    result[order] = (len(values) - np.arange(len(values), dtype=np.float32)) / max(1, len(values))
    return result


def average_precision(labels, scores, beers, denominator):
    order = np.lexsort((beers, -scores))[:10]
    ranked = labels[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float((precision * ranked).sum() / denominator)


def map_groups(model, origin_data, iteration, blend=0.0):
    values = []
    for features, labels, beers, denominator in origin_data.full_group_scores:
        if labels.sum() == 0:
            values.append(0.0)
            continue
        pred = model.predict(features, num_iteration=iteration)
        if blend > 0:
            pred = (1.0 - blend) * rank_percentiles(pred, beers) + blend * rank_percentiles(rrf_prior(features), beers)
        values.append(average_precision(labels, pred, beers, denominator))
    return float(np.mean(values)) if values else 0.0


def map_groups_blends(model, origin_data, iteration, blends):
    values = {blend: [] for blend in blends}
    for features, labels, beers, denominator in origin_data.full_group_scores:
        if labels.sum() == 0:
            for blend in blends:
                values[blend].append(0.0)
            continue
        prediction = rank_percentiles(model.predict(features, num_iteration=iteration), beers)
        prior = rank_percentiles(rrf_prior(features), beers)
        for blend in blends:
            score = (1.0 - blend) * prediction + blend * prior
            values[blend].append(average_precision(labels, score, beers, denominator))
    return {blend: float(np.mean(score)) if score else 0.0 for blend, score in values.items()}


def select_rounds(train_data, heldout, cfg):
    model = train_ranker(train_data, cfg.maximum_rounds)
    candidates = [x for x in (200, 350, 500, 700) if x <= cfg.maximum_rounds]
    if not candidates:
        candidates = [cfg.maximum_rounds]
    blends = (0.0, 0.25, 0.5, 0.75)
    rows = []
    for rounds in candidates:
        origin_scores = [map_groups_blends(model, data, rounds, blends) for data in heldout]
        for blend in blends:
            scores = [score[blend] for score in origin_scores]
            rows.append({"rounds": rounds, "blend": blend, "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))})
    best_mean = max(x["mean"] for x in rows)
    tied = []
    for row in rows:
        standard_error = row["std"] / math.sqrt(max(1, len(row["scores"])))
        if row["mean"] >= best_mean - max(0.001, standard_error):
            tied.append(row)
    best = min(tied, key=lambda x: (-x["blend"], x["rounds"]))
    print(f"[ratebeer] forward_round_selection={json.dumps(rows, separators=(',', ':'))} selected_rounds={best['rounds']} selected_blend={best['blend']}", flush=True)
    return int(best["rounds"]), float(best["blend"]), rows


def predict_groups(model_ranker, con, db_dir, meta, state, channel_model, groups, cfg, blend, limit_users=None):
    users = groups["user_id"].to_numpy(dtype=np.int64)
    process_n = len(users) if limit_users is None else min(len(users), limit_users)
    profiles = load_user_profiles(con, db_dir, meta, state, channel_model, users[:process_n], state.origin, cfg)
    covis = load_covis(con, channel_model, profiles)
    als_rows, knn_rows = factor_recommendations(profiles, channel_model, 450 if not cfg.debug else 50)
    output = np.empty((len(users), 10), dtype=np.int64)
    fallback = [beer for beer, _ in state.global_lists["global90"]]
    if len(fallback) < 10:
        fallback.extend([x for x in range(meta.n_items) if x not in fallback][:10 - len(fallback)])
    for i in range(len(users)):
        if i >= process_n:
            output[i] = np.asarray(fallback[:10], dtype=np.int64)
            continue
        profile = profiles[i]
        sources = candidate_sources(profile, state, meta, covis, als_rows[i], knn_rows[i])
        state_origin_holder[0] = state.origin
        beers, ranks, source_scores = make_candidates(profile, sources, meta, cfg)
        features = build_features(profile, beers, ranks, source_scores, state, meta, channel_model)
        pred = model_ranker.predict(features)
        if blend > 0:
            pred = (1.0 - blend) * rank_percentiles(pred, beers) + blend * rank_percentiles(rrf_prior(features), beers)
        order = np.lexsort((beers, -pred))[:10]
        output[i] = beers[order]
    return output


def exact_reconstructed_groups(con, db_dir, origin):
    frame = con.execute(f"""
        SELECT br.user_id, list(DISTINCT br.beer_id ORDER BY br.beer_id) AS beer_id
        FROM read_parquet('{db_dir / 'beer_ratings.parquet'}') br
        WHERE br.created_at > TIMESTAMP '{origin}'
          AND br.created_at <= TIMESTAMP '{origin}' + INTERVAL 90 DAY
          AND br.user_id IS NOT NULL AND br.beer_id IS NOT NULL
          AND br.total_score >= 4.0
          AND EXISTS (
              SELECT 1 FROM read_parquet('{db_dir / 'beer_ratings.parquet'}') br2
              WHERE br2.user_id=br.user_id
                AND br2.created_at > TIMESTAMP '{origin}' - INTERVAL 90 DAY
                AND br2.created_at <= TIMESTAMP '{origin}'
          )
        GROUP BY br.user_id
        ORDER BY br.user_id
    """).df()
    frame.insert(0, "timestamp", pd.Timestamp(origin))
    return frame


def table_frame(table):
    return table.df.copy()


def validate_predictions(values, n_items):
    assert values.ndim == 2 and values.shape[1] == 10
    assert np.issubdtype(values.dtype, np.integer)
    assert np.all((values >= 0) & (values < n_items))
    assert all(len(set(row.tolist())) == 10 for row in values)


def append_feature_result(cache, diagnostics):
    import fcntl
    path = cache / "features_history.md"
    lines = [
        "",
        "### Multi-channel candidate bank internal gate",
        "- run/experiment: generic_exp_0 lane 0 | status: TESTED-KEPT",
        "- what: 800-item union of temporal popularity, relational loyalty, directed co-visitation, BM25-ALS, item-neighbor, new-release, favorites, availability, and geography channels.",
        f"- outcome: internal origin diagnostics {json.dumps(diagnostics, separators=(',', ':'))}",
        "- takeaway: Retained all channels for fixed forward-selected LambdaRank fitting; per-channel marginal recall identifies discovery contribution after deterministic source order.",
    ]
    lock = cache / ".features_history.lock"
    with lock.open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        with path.open("a") as handle:
            handle.write("\n".join(lines) + "\n")
        fcntl.flock(guard, fcntl.LOCK_UN)


def run(debug, task, output_dir, cache):
    warnings.filterwarnings("ignore")
    cfg = settings(debug)
    timer = Timer()
    root, db_dir, task_dir = paths()
    con = connection()
    meta = load_metadata(con, db_dir)
    timer.log("metadata")
    train = table_frame(task.get_table("train", mask_input_cols=False))
    val = table_frame(task.get_table("val", mask_input_cols=False))
    test = table_frame(task.get_table("test"))
    train["timestamp"] = pd.to_datetime(train["timestamp"])
    val["timestamp"] = pd.to_datetime(val["timestamp"])
    test["timestamp"] = pd.to_datetime(test["timestamp"])
    if debug:
        origin = pd.Timestamp("2018-06-03").to_pydatetime()
        reference = origin
        channel_a = build_channel_model(con, db_dir, cache, meta, reference, cfg)
        frame = train[train["timestamp"] == pd.Timestamp(origin)].head(300).reset_index(drop=True)
        state = build_beer_state(con, db_dir, meta, origin)
        data = build_origin_data(con, db_dir, meta, state, channel_a, frame, cfg)
        ranker = train_ranker([data], cfg.maximum_rounds)
        val_state = build_beer_state(con, db_dir, meta, pd.Timestamp("2018-09-01").to_pydatetime())
        selected_blend = 0.0
        val_pred = predict_groups(ranker, con, db_dir, meta, val_state, channel_a, val, cfg, selected_blend, 300)
        test_state = build_beer_state(con, db_dir, meta, pd.Timestamp("2020-01-01").to_pydatetime())
        test_pred = predict_groups(ranker, con, db_dir, meta, test_state, channel_a, test, cfg, selected_blend, 300)
        diagnostics = [data.diagnostics]
        selected_rounds = cfg.maximum_rounds
        selection = []
        timer.log("debug_pipeline")
    else:
        origins_a = [pd.Timestamp(x).to_pydatetime() for x in (
            "2016-12-10", "2017-03-10", "2017-06-08", "2017-09-06", "2017-12-05", "2018-03-05", "2018-06-03"
        )]
        channel_a = build_channel_model(con, db_dir, cache, meta, origins_a[0], cfg)
        timer.log("model_a_channels")
        data_a = []
        for origin in origins_a:
            channel_origin = build_channel_model(con, db_dir, cache, meta, origin, cfg)
            frame = train[train["timestamp"] == pd.Timestamp(origin)].reset_index(drop=True)
            state = build_beer_state(con, db_dir, meta, origin)
            evaluate_full = origin in origins_a[-2:]
            data_a.append(build_origin_data(con, db_dir, meta, state, channel_origin, frame, cfg, evaluate_full=evaluate_full))
            timer.log(f"model_a_origin_{origin.date()}")
        selected_rounds, selected_blend, selection = select_rounds(data_a[:-2], data_a[-2:], cfg)
        ranker_a = train_ranker(data_a, selected_rounds)
        timer.log("model_a_fit")
        val_origin = pd.Timestamp("2018-09-01").to_pydatetime()
        channel_val = build_channel_model(con, db_dir, cache, meta, val_origin, cfg)
        val_state = build_beer_state(con, db_dir, meta, val_origin)
        val_pred = predict_groups(ranker_a, con, db_dir, meta, val_state, channel_val, val, cfg, selected_blend)
        timer.log("model_a_validation_inference")
        diagnostics_a = [x.diagnostics for x in data_a]
        del data_a
        del ranker_a
        import gc
        gc.collect()
        origins_b = [pd.Timestamp(x).to_pydatetime() for x in (
            "2018-09-01", "2018-11-30", "2019-02-28", "2019-05-29", "2019-08-27"
        )]
        channel_b = build_channel_model(con, db_dir, cache, meta, origins_b[0], cfg)
        timer.log("model_b_channels")
        data_b = []
        for index, origin in enumerate(origins_b):
            channel_origin = build_channel_model(con, db_dir, cache, meta, origin, cfg)
            frame = val.reset_index(drop=True) if index == 0 else exact_reconstructed_groups(con, db_dir, origin)
            state = build_beer_state(con, db_dir, meta, origin)
            data_b.append(build_origin_data(con, db_dir, meta, state, channel_origin, frame, cfg))
            timer.log(f"model_b_origin_{origin.date()}")
        ranker_b = train_ranker(data_b, selected_rounds)
        timer.log("model_b_fit")
        test_origin = pd.Timestamp("2020-01-01").to_pydatetime()
        channel_test = build_channel_model(con, db_dir, cache, meta, test_origin, cfg)
        test_state = build_beer_state(con, db_dir, meta, test_origin)
        test_pred = predict_groups(ranker_b, con, db_dir, meta, test_state, channel_test, test, cfg, selected_blend)
        timer.log("model_b_test_inference")
        diagnostics = diagnostics_a + [x.diagnostics for x in data_b]
        append_feature_result(cache, diagnostics)
    validate_predictions(val_pred, meta.n_items)
    validate_predictions(test_pred, meta.n_items)
    np.save(output_dir / "val_predictions.npy", val_pred.astype(np.int64))
    np.save(output_dir / "test_predictions.npy", test_pred.astype(np.int64))
    metrics = {
        "debug": debug,
        "model_a_never_used_validation_labels": True,
        "model_b_used_legal_supervision_through_test_cutoff": not debug,
        "selected_rounds": selected_rounds,
        "selected_rrf_blend": selected_blend,
        "forward_selection": selection,
        "candidate_diagnostics": diagnostics,
        "val_shape": list(val_pred.shape),
        "test_shape": list(test_pred.shape),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    timer.log("saved")
    print(f"[ratebeer] diagnostics={json.dumps(metrics, separators=(',', ':'))}", flush=True)
    return val_pred, test_pred
