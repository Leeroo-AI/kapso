from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit
from scipy import sparse

warnings.filterwarnings("ignore")


# Configuration

N_CUSTOMERS = 1_850_193
N_PRODUCTS = 506_012
HISTORY_CAP = 50
SOURCE_ITEMS = 15
SUCCESSORS = 30
CANDIDATE_CAP = 600
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
CACHE_VERSION = "lane0_staged_ltr_v3"
DAY_ZERO = np.datetime64("1970-01-01", "D")
POPULARITY_WEIGHTS = (1.0, 0.25, 0.05, 0.2)
DIRECTIONAL_MIX = (10.0, 0.04)


def elapsed(start: float, phase: str) -> None:
    print(f"[solution] {phase}: {time.time() - start:.1f}s", flush=True)


def cutoff_day(cutoff: str) -> int:
    return int((np.datetime64(cutoff, "D") - DAY_ZERO).astype(np.int64))


def cache_paths() -> tuple[Path, Path]:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    root = shared / CACHE_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return shared, root


def register_artifact(name: str, path: Path, description: str, content_key: str) -> None:
    shared, _ = cache_paths()
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
            relative = str(path.relative_to(shared))
            if not any(x.get("path") == relative for x in records):
                records.append(
                    {
                        "name": name,
                        "path": relative,
                        "description": description,
                        "content_key": content_key,
                        "rebuild_hint": "Run python main.py; the candidate extends this cache automatically.",
                    }
                )
                registry.write_text(json.dumps(records, indent=2))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def database_paths() -> dict[str, Path]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    task = root / "tasks" / os.environ["RELBENCH_TASK"]
    return {
        "review": root / "db" / "review.parquet",
        "product": root / "db" / "product.parquet",
        "customer": root / "db" / "customer.parquet",
        "train": task / "train.parquet",
        "val": task / "val.parquet",
        "test": task / "test.parquet",
    }


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"set threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    con.execute("set memory_limit='80GB'")
    con.execute("set preserve_insertion_order=false")
    return con


# Static data

@dataclass
class StaticData:
    brand_code: np.ndarray
    category_code: np.ndarray
    price: np.ndarray
    brand_size: np.ndarray
    category_size: np.ndarray
    name_code: np.ndarray
    name_size: np.ndarray
    product_text: pd.DataFrame | None


def normalize_names(values: pd.Series) -> np.ndarray:
    normalized = values.fillna("").str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
    return pd.factorize(normalized, sort=True)[0].astype(np.int32)


def load_static(include_text: bool) -> StaticData:
    paths = database_paths()
    con = connection()
    columns = "product_id,category,brand,price" + (",title,description" if include_text else "")
    product = con.sql(f"select {columns} from read_parquet('{paths['product']}') order by product_id").df()
    brand = product["brand"].fillna("").str.strip().replace("", "__missing__")
    category = product["category"].map(
        lambda x: str(x[-1]) if isinstance(x, np.ndarray) and len(x) else "__missing__"
    )
    brand_code, brand_values = pd.factorize(brand, sort=True)
    category_code, category_values = pd.factorize(category, sort=True)
    price = product["price"].to_numpy(np.float32)
    finite_price = price[np.isfinite(price)]
    price[~np.isfinite(price)] = np.median(finite_price)
    customer = con.sql(
        f"select customer_id,customer_name from read_parquet('{paths['customer']}') order by customer_id"
    ).df()
    name_code = normalize_names(customer["customer_name"])
    name_size = np.bincount(name_code, minlength=int(name_code.max()) + 1).astype(np.int32)
    product_text = None
    if include_text:
        product_text = pd.DataFrame(
            {
                "title": product["title"].fillna(""),
                "brand": brand,
                "category": product["category"].map(
                    lambda x: " > ".join(map(str, x)) if isinstance(x, np.ndarray) else ""
                ),
                "description": product["description"].fillna(""),
            }
        )
    return StaticData(
        brand_code=brand_code.astype(np.int32),
        category_code=category_code.astype(np.int32),
        price=price,
        brand_size=np.bincount(brand_code, minlength=len(brand_values)).astype(np.int32),
        category_size=np.bincount(category_code, minlength=len(category_values)).astype(np.int32),
        name_code=name_code,
        name_size=name_size,
        product_text=product_text,
    )


# Snapshot artifacts

@dataclass
class ItemStats:
    count30: np.ndarray
    count91: np.ndarray
    count365: np.ndarray
    count_all: np.ndarray
    five_all: np.ndarray
    rating_mean: np.ndarray
    verified_share: np.ndarray
    first_day: np.ndarray
    last_day: np.ndarray
    seasonal: np.ndarray
    popularity: np.ndarray
    brand_top: np.ndarray
    category_top: np.ndarray


@dataclass
class UserStats:
    count91: np.ndarray
    count365: np.ndarray
    count_all: np.ndarray
    five_share: np.ndarray
    rating_mean: np.ndarray
    verified_share: np.ndarray


@dataclass
class Histories:
    product: np.ndarray
    day: np.ndarray
    rating: np.ndarray
    verified: np.ndarray


@dataclass
class Directional:
    product: np.ndarray
    weight: np.ndarray
    normalized: np.ndarray
    successor_day: np.ndarray
    support: np.ndarray


@njit(cache=True)
def fill_group_top(order: np.ndarray, codes: np.ndarray, n_groups: int, width: int) -> np.ndarray:
    result = np.full((n_groups, width), -1, np.int32)
    counts = np.zeros(n_groups, np.int32)
    for product in order:
        code = codes[product]
        pos = counts[code]
        if pos < width:
            result[code, pos] = product
            counts[code] += 1
    return result


def build_item_stats(cutoff: str, static: StaticData) -> ItemStats:
    paths = database_paths()
    con = connection()
    q = f"""
        select product_id,
        count(*) filter(where review_time>'{cutoff}'::timestamp-interval '30 days') c30,
        count(*) filter(where review_time>'{cutoff}'::timestamp-interval '91 days') c91,
        count(*) filter(where review_time>'{cutoff}'::timestamp-interval '365 days') c365,
        count(*) filter(where rating=5 and review_time>'{cutoff}'::timestamp-interval '30 days') f5_30,
        count(*) filter(where rating=5 and review_time>'{cutoff}'::timestamp-interval '91 days') f5_91,
        count(*) filter(where rating=5 and review_time>'{cutoff}'::timestamp-interval '365 days') f5_365,
        count(*) c_all,
        count(*) filter(where rating=5) f5,
        avg(rating) rating_mean,
        avg(verified::integer) verified_share,
        date_diff('day','1970-01-01'::timestamp,min(review_time)) first_day,
        date_diff('day','1970-01-01'::timestamp,max(review_time)) last_day,
        count(*) filter(where rating=5 and review_time>'{cutoff}'::timestamp-interval '365 days' and review_time<='{cutoff}'::timestamp-interval '274 days') seasonal
        from read_parquet('{paths['review']}')
        where review_time<='{cutoff}'::timestamp
        group by product_id
    """
    frame = con.sql(q).df()
    arrays: dict[str, np.ndarray] = {}
    integer_columns = ["c30", "c91", "c365", "f5_30", "f5_91", "f5_365", "c_all", "f5", "first_day", "last_day", "seasonal"]
    float_columns = ["rating_mean", "verified_share"]
    ids = frame["product_id"].to_numpy(np.int64)
    for column in integer_columns:
        values = np.zeros(N_PRODUCTS, np.int32)
        values[ids] = frame[column].to_numpy(np.int32)
        arrays[column] = values
    for column in float_columns:
        values = np.zeros(N_PRODUCTS, np.float32)
        values[ids] = frame[column].to_numpy(np.float32)
        arrays[column] = values
    score = (
        POPULARITY_WEIGHTS[0] * arrays["f5_91"].astype(np.float64)
        + POPULARITY_WEIGHTS[1] * arrays["f5_365"]
        + POPULARITY_WEIGHTS[2] * arrays["f5"]
        + POPULARITY_WEIGHTS[3] * arrays["seasonal"]
    )
    order = np.lexsort((np.arange(N_PRODUCTS), -arrays["last_day"], -score)).astype(np.int32)
    brand_top = fill_group_top(order, static.brand_code, len(static.brand_size), 30)
    category_top = fill_group_top(order, static.category_code, len(static.category_size), 40)
    return ItemStats(
        count30=arrays["c30"],
        count91=arrays["c91"],
        count365=arrays["c365"],
        count_all=arrays["c_all"],
        five_all=arrays["f5"],
        rating_mean=arrays["rating_mean"],
        verified_share=arrays["verified_share"],
        first_day=arrays["first_day"],
        last_day=arrays["last_day"],
        seasonal=arrays["seasonal"],
        popularity=order[:2000],
        brand_top=brand_top,
        category_top=category_top,
    )


def build_user_stats(cutoff: str) -> UserStats:
    paths = database_paths()
    con = connection()
    q = f"""
        select customer_id,
        count(*) filter(where review_time>'{cutoff}'::timestamp-interval '91 days') c91,
        count(*) filter(where review_time>'{cutoff}'::timestamp-interval '365 days') c365,
        count(*) c_all,
        avg((rating=5)::integer) five_share,
        avg(rating) rating_mean,
        avg(verified::integer) verified_share
        from read_parquet('{paths['review']}')
        where review_time<='{cutoff}'::timestamp
        group by customer_id
    """
    frame = con.sql(q).df()
    ids = frame["customer_id"].to_numpy(np.int64)
    result = {}
    for column in ["c91", "c365", "c_all"]:
        values = np.zeros(N_CUSTOMERS, np.int32)
        values[ids] = frame[column].to_numpy(np.int32)
        result[column] = values
    for column in ["five_share", "rating_mean", "verified_share"]:
        values = np.zeros(N_CUSTOMERS, np.float32)
        values[ids] = frame[column].to_numpy(np.float32)
        result[column] = values
    return UserStats(
        count91=result["c91"],
        count365=result["c365"],
        count_all=result["c_all"],
        five_share=result["five_share"],
        rating_mean=result["rating_mean"],
        verified_share=result["verified_share"],
    )


def load_histories(customers: np.ndarray, cutoff: str) -> Histories:
    paths = database_paths()
    con = connection()
    seed = pd.DataFrame({"row_id": np.arange(len(customers), dtype=np.int32), "customer_id": customers})
    con.register("seed_customers", seed)
    q = f"""
        select row_id,product_id,
        date_diff('day','1970-01-01'::timestamp,review_time)::integer event_day,
        rating,verified,
        row_number() over(partition by row_id order by review_time desc,product_id) history_rank
        from seed_customers join read_parquet('{paths['review']}') using(customer_id)
        where review_time<='{cutoff}'::timestamp
        qualify history_rank<={HISTORY_CAP}
    """
    frame = con.sql(q).df()
    shape = (len(customers), HISTORY_CAP)
    product = np.full(shape, -1, np.int32)
    day = np.zeros(shape, np.int32)
    rating = np.zeros(shape, np.float32)
    verified = np.zeros(shape, np.float32)
    rows = frame["row_id"].to_numpy(np.int64)
    ranks = frame["history_rank"].to_numpy(np.int64) - 1
    product[rows, ranks] = frame["product_id"].to_numpy(np.int32)
    day[rows, ranks] = frame["event_day"].to_numpy(np.int32)
    rating[rows, ranks] = frame["rating"].to_numpy(np.float32)
    verified[rows, ranks] = frame["verified"].fillna(False).to_numpy(np.float32)
    return Histories(product=product, day=day, rating=rating, verified=verified)


def transition_directory(cutoff: str) -> Path:
    _, root = cache_paths()
    return root / f"directional_target_only_span49_{cutoff}"


def build_directional(cutoff: str) -> Directional:
    directory = transition_directory(cutoff)
    done = directory / "done.json"
    names = ["product", "weight", "normalized", "successor_day", "support"]
    if done.exists() and all((directory / f"{x}.npy").exists() for x in names):
        return Directional(*(np.load(directory / f"{x}.npy", mmap_mode="r") for x in names))
    directory.mkdir(parents=True, exist_ok=True)
    paths = database_paths()
    parquet = directory / "pairs.parquet"
    con = connection()
    q = f"""
        copy (
        with recent as (
            select customer_id,product_id,review_time,rating,
            row_number() over(partition by customer_id order by review_time desc,product_id) rn
            from read_parquet('{paths['review']}') where review_time<='{cutoff}'::timestamp
            qualify rn<={HISTORY_CAP}
        ), pairs as (
            select a.product_id src,b.product_id dst,
            sum(pow(0.5,date_diff('day',a.review_time,b.review_time)/365.0)) pair_weight,
            count(*) support,
            date_diff('day','1970-01-01'::timestamp,max(b.review_time)) successor_day
            from recent a join recent b
            on a.customer_id=b.customer_id and b.rn<a.rn and a.rn-b.rn<=49
            where a.product_id<>b.product_id and b.rating=5
            group by src,dst
        ), ranked as (
            select *,pair_weight/sum(pair_weight) over(partition by src) normalized_score,
            row_number() over(partition by src order by pair_weight desc,successor_day desc,dst) rank
            from pairs
        )
        select * from ranked where rank<={SUCCESSORS}
        ) to '{parquet}' (format parquet,compression zstd)
    """
    con.execute(q)
    frame = con.sql(f"select * from read_parquet('{parquet}')").df()
    shape = (N_PRODUCTS, SUCCESSORS)
    product = np.full(shape, -1, np.int32)
    weight = np.zeros(shape, np.float32)
    normalized = np.zeros(shape, np.float32)
    successor_day = np.zeros(shape, np.int32)
    support = np.zeros(shape, np.float32)
    src = frame["src"].to_numpy(np.int64)
    rank = frame["rank"].to_numpy(np.int64) - 1
    product[src, rank] = frame["dst"].to_numpy(np.int32)
    weight[src, rank] = frame["pair_weight"].to_numpy(np.float32)
    normalized[src, rank] = frame["normalized_score"].to_numpy(np.float32)
    successor_day[src, rank] = frame["successor_day"].to_numpy(np.int32)
    support[src, rank] = frame["support"].to_numpy(np.float32)
    for name, array in zip(names, [product, weight, normalized, successor_day, support]):
        np.save(directory / f"{name}.npy", array)
    done.write_text(json.dumps({"cutoff": cutoff, "history_cap": HISTORY_CAP, "successor_span": 49, "target_five_star_only": True}))
    parquet.unlink(missing_ok=True)
    register_artifact(
        f"Directional transitions {cutoff}",
        directory,
        "Cutoff-safe decayed directional top-30 item successors.",
        f"{CACHE_VERSION}:directional:{cutoff}",
    )
    return Directional(*(np.load(directory / f"{x}.npy", mmap_mode="r") for x in names))


# Representation artifacts

def embedding_paths() -> tuple[Path, Path]:
    _, root = cache_paths()
    key = hashlib.sha256(f"bge-small:{MODEL_REVISION}:metadata-v1:256".encode()).hexdigest()[:16]
    return root / f"product_embeddings_{key}.npy", root / f"product_faiss_{key}.index"


def build_embeddings(static: StaticData) -> np.ndarray:
    embedding_path, _ = embedding_paths()
    if embedding_path.exists():
        return np.load(embedding_path, mmap_mode="r")
    from sentence_transformers import SentenceTransformer

    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-en-v1.5" / "snapshots"
    model_path = snapshots / MODEL_REVISION
    model = SentenceTransformer(str(model_path), device="cuda")
    model.max_seq_length = 256
    text = static.product_text
    values = (
        text["title"].str.slice(0, 400)
        + " [SEP] "
        + text["brand"].str.slice(0, 200)
        + " [SEP] "
        + text["category"].str.slice(0, 300)
        + " [SEP] "
        + text["description"].str.slice(0, 1600)
    ).tolist()
    encoded = model.encode(
        values,
        batch_size=512,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
        device="cuda",
    ).astype(np.float16)
    np.save(embedding_path, encoded)
    register_artifact(
        "BGE product metadata embeddings",
        embedding_path,
        "Normalized FP16 title, brand-group, category-path, and description embeddings.",
        f"{CACHE_VERSION}:bge:{MODEL_REVISION}:metadata-v1",
    )
    return np.load(embedding_path, mmap_mode="r")


def build_semantic_index(embeddings: np.ndarray) -> faiss.Index:
    _, index_path = embedding_paths()
    if index_path.exists():
        index = faiss.read_index(str(index_path))
        index.nprobe = 32
        return index
    dimension = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, 2048, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(1337)
    sample = rng.choice(len(embeddings), size=120_000, replace=False)
    index.train(np.asarray(embeddings[sample], dtype=np.float32))
    for start in range(0, len(embeddings), 20_000):
        index.add(np.asarray(embeddings[start : start + 20_000], dtype=np.float32))
    index.nprobe = 32
    faiss.write_index(index, str(index_path))
    register_artifact(
        "BGE product Faiss index",
        index_path,
        "IVF inner-product retrieval index for normalized product embeddings.",
        f"{CACHE_VERSION}:faiss:{MODEL_REVISION}:ivf2048",
    )
    return index


@njit(cache=True)
def merge_semantic(long_ids: np.ndarray, long_scores: np.ndarray, short_ids: np.ndarray, short_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(long_ids)
    output = np.full((n, 125), -1, np.int32)
    scores = np.zeros((n, 125), np.float32)
    stamps = np.zeros(N_PRODUCTS, np.int32)
    positions = np.zeros(N_PRODUCTS, np.int16)
    for row in range(n):
        size = 0
        stamp = row + 1
        for rank in range(100):
            for source in range(2):
                product = long_ids[row, rank] if source == 0 else short_ids[row, rank]
                value = long_scores[row, rank] if source == 0 else short_scores[row, rank]
                if product < 0:
                    continue
                if stamps[product] == stamp:
                    pos = positions[product]
                    if value > scores[row, pos]:
                        scores[row, pos] = value
                elif size < 125:
                    stamps[product] = stamp
                    positions[product] = size
                    output[row, size] = product
                    scores[row, size] = value
                    size += 1
    return output, scores


def semantic_candidates(histories: Histories, embeddings: np.ndarray, index: faiss.Index) -> tuple[np.ndarray, np.ndarray]:
    n = len(histories.product)
    long_ids = np.full((n, 100), -1, np.int32)
    short_ids = np.full((n, 100), -1, np.int32)
    long_scores = np.zeros((n, 100), np.float32)
    short_scores = np.zeros((n, 100), np.float32)
    for start in range(0, n, 3000):
        stop = min(start + 3000, n)
        products = histories.product[start:stop]
        ratings = histories.rating[start:stop]
        long_profile = np.zeros((stop - start, embeddings.shape[1]), np.float32)
        short_profile = np.zeros_like(long_profile)
        long_count = np.zeros(stop - start, np.float32)
        short_count = np.zeros(stop - start, np.float32)
        for rank in range(HISTORY_CAP):
            valid = (products[:, rank] >= 0) & (ratings[:, rank] == 5)
            rows = np.flatnonzero(valid)
            if len(rows):
                long_profile[rows] += np.asarray(embeddings[products[rows, rank]], dtype=np.float32)
                long_count[rows] += 1
                short_rows = rows[short_count[rows] < 5]
                if len(short_rows):
                    short_profile[short_rows] += np.asarray(embeddings[products[short_rows, rank]], dtype=np.float32)
                    short_count[short_rows] += 1
        warm_long = long_count > 0
        warm_short = short_count > 0
        long_profile[warm_long] /= long_count[warm_long, None]
        short_profile[warm_short] /= short_count[warm_short, None]
        faiss.normalize_L2(long_profile)
        faiss.normalize_L2(short_profile)
        if warm_long.any():
            values, ids = index.search(long_profile[warm_long], 100)
            target = np.flatnonzero(warm_long) + start
            long_ids[target] = ids.astype(np.int32)
            long_scores[target] = values.astype(np.float32)
        if warm_short.any():
            values, ids = index.search(short_profile[warm_short], 100)
            target = np.flatnonzero(warm_short) + start
            short_ids[target] = ids.astype(np.int32)
            short_scores[target] = values.astype(np.float32)
    return merge_semantic(long_ids, long_scores, short_ids, short_scores)


def als_directory(cutoff: str) -> Path:
    _, root = cache_paths()
    return root / f"als96_{cutoff}"


def five_star_matrix(cutoff: str) -> sparse.csr_matrix:
    paths = database_paths()
    con = connection()
    q = f"""
        select customer_id,product_id,
        (1.0+0.35*verified::integer)*pow(0.5,date_diff('day',review_time,'{cutoff}'::timestamp)/365.0) weight
        from read_parquet('{paths['review']}')
        where review_time<='{cutoff}'::timestamp and rating=5
    """
    frame = con.sql(q).fetchnumpy()
    matrix = sparse.csr_matrix(
        (frame["weight"].astype(np.float32), (frame["customer_id"], frame["product_id"])),
        shape=(N_CUSTOMERS, N_PRODUCTS),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    return matrix


def build_als(cutoff: str) -> tuple[np.ndarray, np.ndarray]:
    directory = als_directory(cutoff)
    user_path = directory / "user_factors.npy"
    item_path = directory / "item_factors.npy"
    if user_path.exists() and item_path.exists():
        return np.load(user_path, mmap_mode="r"), np.load(item_path, mmap_mode="r")
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight

    directory.mkdir(parents=True, exist_ok=True)
    matrix = five_star_matrix(cutoff)
    weighted = bm25_weight(matrix, K1=100, B=0.8).tocsr().astype(np.float32)
    model = AlternatingLeastSquares(
        factors=96,
        iterations=12,
        regularization=0.02,
        alpha=1.0,
        random_state=1337,
        num_threads=int(os.environ.get("OMP_NUM_THREADS", "11")),
    )
    model.fit(weighted, show_progress=False)
    np.save(user_path, model.user_factors.astype(np.float32))
    np.save(item_path, model.item_factors.astype(np.float32))
    register_artifact(
        f"Five-star ALS snapshot {cutoff}",
        directory,
        "CPU ALS factors over verified-weighted, time-decayed, BM25-normalized five-star events.",
        f"{CACHE_VERSION}:als96x12:{cutoff}",
    )
    return np.load(user_path, mmap_mode="r"), np.load(item_path, mmap_mode="r")


def als_candidates(customers: np.ndarray, user_factors: np.ndarray, item_factors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((len(customers), 175), -1, np.int32)
    scores = np.zeros((len(customers), 175), np.float32)
    factors = np.asarray(item_factors, dtype=np.float32)
    quantizer = faiss.IndexFlatIP(factors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, factors.shape[1], 2048, faiss.METRIC_INNER_PRODUCT)
    sample = np.linspace(0, len(factors) - 1, 120_000, dtype=np.int64)
    index.train(factors[sample])
    index.add(factors)
    index.nprobe = 32
    queries = np.asarray(user_factors[customers], dtype=np.float32)
    cold = np.linalg.norm(queries, axis=1) == 0
    for start in range(0, len(customers), 10_000):
        stop = min(start + 10_000, len(customers))
        batch_scores, batch_ids = index.search(queries[start:stop], 175)
        ids[start:stop] = batch_ids.astype(np.int32)
        scores[start:stop] = batch_scores.astype(np.float32)
    ids[cold] = -1
    scores[cold] = 0
    return ids, scores


# Candidate generation

@njit(cache=True)
def directional_candidates(hist_product: np.ndarray, hist_day: np.ndarray, cutoff: int, successors: np.ndarray, weights: np.ndarray, normalized: np.ndarray, normalized_weight: float, rank_penalty: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(hist_product)
    output = np.full((n, 200), -1, np.int32)
    output_score = np.zeros((n, 200), np.float32)
    stamps = np.zeros(N_PRODUCTS, np.int32)
    positions = np.zeros(N_PRODUCTS, np.int16)
    candidate_ids = np.empty(SOURCE_ITEMS * SUCCESSORS, np.int32)
    candidate_scores = np.empty(SOURCE_ITEMS * SUCCESSORS, np.float32)
    for row in range(n):
        size = 0
        stamp = row + 1
        for history_rank in range(SOURCE_ITEMS):
            source = hist_product[row, history_rank]
            if source < 0:
                continue
            source_decay = pow(0.5, max(0, cutoff - hist_day[row, history_rank]) / 180.0)
            for rank in range(SUCCESSORS):
                product = successors[source, rank]
                if product < 0:
                    break
                value = source_decay * (weights[source, rank] + normalized_weight * normalized[source, rank]) / (1.0 + rank_penalty * rank)
                if stamps[product] == stamp:
                    candidate_scores[positions[product]] += value
                else:
                    stamps[product] = stamp
                    positions[product] = size
                    candidate_ids[size] = product
                    candidate_scores[size] = value
                    size += 1
        if size:
            order = np.argsort(-candidate_scores[:size])
            width = min(200, size)
            for rank in range(width):
                pos = order[rank]
                output[row, rank] = candidate_ids[pos]
                output_score[row, rank] = candidate_scores[pos]
    return output, output_score


@njit(cache=True)
def affinity_candidates(hist_product: np.ndarray, hist_rating: np.ndarray, brand_code: np.ndarray, category_code: np.ndarray, brand_top: np.ndarray, category_top: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(hist_product)
    output = np.full((n, 100), -1, np.int32)
    output_score = np.zeros((n, 100), np.float32)
    stamps = np.zeros(N_PRODUCTS, np.int32)
    positions = np.zeros(N_PRODUCTS, np.int16)
    ids = np.empty(150, np.int32)
    scores = np.empty(150, np.float32)
    for row in range(n):
        preferred_brands = np.full(5, -1, np.int32)
        brand_counts = np.zeros(5, np.int16)
        preferred_categories = np.full(3, -1, np.int32)
        category_counts = np.zeros(3, np.int16)
        for h in range(HISTORY_CAP):
            source = hist_product[row, h]
            if source < 0 or hist_rating[row, h] != 5:
                continue
            b = brand_code[source]
            c = category_code[source]
            found = -1
            for j in range(5):
                if preferred_brands[j] == b:
                    found = j
                    break
            if found >= 0:
                brand_counts[found] += 1
            else:
                for j in range(5):
                    if preferred_brands[j] < 0:
                        preferred_brands[j] = b
                        brand_counts[j] = 1
                        break
            found = -1
            for j in range(3):
                if preferred_categories[j] == c:
                    found = j
                    break
            if found >= 0:
                category_counts[found] += 1
            else:
                for j in range(3):
                    if preferred_categories[j] < 0:
                        preferred_categories[j] = c
                        category_counts[j] = 1
                        break
        size = 0
        stamp = row + 1
        for pref in range(5):
            code = preferred_brands[pref]
            if code < 0:
                continue
            for rank in range(30):
                product = brand_top[code, rank]
                if product < 0:
                    break
                value = brand_counts[pref] / (60.0 + pref + rank)
                if stamps[product] == stamp:
                    scores[positions[product]] += value
                else:
                    stamps[product] = stamp
                    positions[product] = size
                    ids[size] = product
                    scores[size] = value
                    size += 1
        if size:
            order = np.argsort(-scores[:size])
            width = min(100, size)
            for rank in range(width):
                pos = order[rank]
                output[row, rank] = ids[pos]
                output_score[row, rank] = scores[pos]
    return output, output_score


@njit(cache=True)
def union_candidates(cf_ids: np.ndarray, cf_scores: np.ndarray, als_ids: np.ndarray, als_scores: np.ndarray, sem_ids: np.ndarray, sem_scores: np.ndarray, affinity_ids: np.ndarray, affinity_scores: np.ndarray, popularity: np.ndarray, use_als: bool, use_semantic: bool, seen_history: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(cf_ids)
    ids = np.full((n, CANDIDATE_CAP), -1, np.int32)
    ranks = np.full((n, CANDIDATE_CAP, 5), 1000, np.int16)
    scores = np.zeros((n, CANDIDATE_CAP, 5), np.float32)
    flags = np.zeros((n, CANDIDATE_CAP), np.int16)
    stamps = np.zeros(N_PRODUCTS, np.int32)
    positions = np.zeros(N_PRODUCTS, np.int16)
    seen_stamps = np.zeros(N_PRODUCTS, np.int32)
    widths = np.array([200, 175, 125, 100, 125], np.int32)
    for row in range(n):
        stamp = row + 1
        for h in range(HISTORY_CAP):
            product = seen_history[row, h]
            if product >= 0:
                seen_stamps[product] = stamp
        size = 0
        for rank in range(200):
            for source in range(5):
                if rank >= widths[source] or (source == 1 and not use_als) or (source == 2 and not use_semantic):
                    continue
                if source == 0:
                    product = cf_ids[row, rank]
                    value = cf_scores[row, rank]
                elif source == 1:
                    product = als_ids[row, rank]
                    value = als_scores[row, rank]
                elif source == 2:
                    product = sem_ids[row, rank]
                    value = sem_scores[row, rank]
                elif source == 3:
                    product = affinity_ids[row, rank]
                    value = affinity_scores[row, rank]
                else:
                    product = popularity[rank]
                    value = 1.0 / (rank + 1.0)
                if product < 0 or seen_stamps[product] == stamp:
                    continue
                if stamps[product] == stamp:
                    pos = positions[product]
                    ranks[row, pos, source] = rank + 1
                    scores[row, pos, source] = value
                    flags[row, pos] |= 1 << source
                elif size < CANDIDATE_CAP:
                    stamps[product] = stamp
                    positions[product] = size
                    ids[row, size] = product
                    ranks[row, size, source] = rank + 1
                    scores[row, size, source] = value
                    flags[row, size] = 1 << source
                    size += 1
        if size < CANDIDATE_CAP:
            for product in popularity:
                if size >= CANDIDATE_CAP:
                    break
                if seen_stamps[product] == stamp or stamps[product] == stamp:
                    continue
                stamps[product] = stamp
                positions[product] = size
                ids[row, size] = product
                ranks[row, size, 4] = size + 1
                scores[row, size, 4] = 1.0 / (size + 1.0)
                flags[row, size] = 1 << 4
                size += 1
    return ids, ranks, scores, flags


@dataclass
class SourceCandidates:
    cf_ids: np.ndarray
    cf_scores: np.ndarray
    als_ids: np.ndarray
    als_scores: np.ndarray
    sem_ids: np.ndarray
    sem_scores: np.ndarray
    affinity_ids: np.ndarray
    affinity_scores: np.ndarray


@dataclass
class Snapshot:
    cutoff: str
    customers: np.ndarray
    histories: Histories
    item: ItemStats
    user: UserStats
    sources: SourceCandidates


def build_snapshot(cutoff: str, customers: np.ndarray, static: StaticData, embeddings: np.ndarray | None, semantic_index: faiss.Index | None, heavy: bool) -> Snapshot:
    start = time.time()
    histories = load_histories(customers, cutoff)
    item = build_item_stats(cutoff, static)
    user = build_user_stats(cutoff)
    directional = build_directional(cutoff)
    cf_ids, cf_scores = directional_candidates(
        histories.product,
        histories.day,
        cutoff_day(cutoff),
        directional.product,
        directional.weight,
        directional.normalized,
        DIRECTIONAL_MIX[0],
        DIRECTIONAL_MIX[1],
    )
    affinity_ids, affinity_scores = affinity_candidates(
        histories.product,
        histories.rating,
        static.brand_code,
        static.category_code,
        item.brand_top,
        item.category_top,
    )
    if heavy:
        user_factors, item_factors = build_als(cutoff)
        als_ids, als_scores = als_candidates(customers, user_factors, item_factors)
        sem_ids, sem_scores = semantic_candidates(histories, embeddings, semantic_index)
    else:
        als_ids = np.full((len(customers), 175), -1, np.int32)
        als_scores = np.zeros((len(customers), 175), np.float32)
        sem_ids = np.full((len(customers), 125), -1, np.int32)
        sem_scores = np.zeros((len(customers), 125), np.float32)
    elapsed(start, f"snapshot {cutoff} for {len(customers)} seeds")
    return Snapshot(
        cutoff=cutoff,
        customers=customers,
        histories=histories,
        item=item,
        user=user,
        sources=SourceCandidates(
            cf_ids=cf_ids,
            cf_scores=cf_scores,
            als_ids=als_ids,
            als_scores=als_scores,
            sem_ids=sem_ids,
            sem_scores=sem_scores,
            affinity_ids=affinity_ids,
            affinity_scores=affinity_scores,
        ),
    )


# Features and ranking

FEATURE_NAMES = [
    "cf_inv_rank", "cf_score", "cf_flag", "als_inv_rank", "als_score", "als_flag",
    "semantic_inv_rank", "semantic_score", "semantic_flag", "affinity_inv_rank", "affinity_score", "affinity_flag",
    "pop_inv_rank", "pop_score", "pop_flag", "source_count", "rrf",
    "item_count30", "item_count91", "item_count365", "item_count_all", "item_five_rate",
    "item_velocity30", "item_velocity91", "item_rating_mean", "item_verified_share", "item_price",
    "item_first_age", "item_last_age", "item_seasonal", "brand_group_size", "category_size",
    "user_count91", "user_count365", "user_count_all", "user_five_share", "user_rating_mean",
    "user_verified_share", "history_count", "history_recency", "history_price_mean", "history_price_std",
    "distinct_brand_groups", "distinct_categories", "name_frequency", "name_cohort_activity",
    "brand_affinity", "category_affinity", "price_distance", "seen_product", "history_brand_count",
    "history_category_count", "candidate_recency", "candidate_age", "cf_pop_interaction", "als_semantic_interaction",
]


def feature_matrix(snapshot: Snapshot, static: StaticData, ids: np.ndarray, ranks: np.ndarray, scores: np.ndarray, flags: np.ndarray) -> np.ndarray:
    batch, width = ids.shape
    products = ids.reshape(-1)
    rows = np.repeat(np.arange(batch), width)
    customers = snapshot.customers[rows]
    cutoff = cutoff_day(snapshot.cutoff)
    matrix = np.zeros((len(products), len(FEATURE_NAMES)), np.float32)
    cursor = 0
    for source in range(5):
        rank = ranks[:, :, source].reshape(-1)
        score = scores[:, :, source].reshape(-1)
        present = rank < 1000
        matrix[:, cursor] = np.where(present, 1.0 / rank, 0)
        matrix[:, cursor + 1] = score
        matrix[:, cursor + 2] = present
        cursor += 3
    source_count = np.bitwise_and(flags.reshape(-1)[:, None], (1 << np.arange(5))).astype(bool).sum(axis=1)
    matrix[:, 15] = source_count
    matrix[:, 16] = np.sum(np.where(ranks.reshape(-1, 5) < 1000, 1.0 / (60.0 + ranks.reshape(-1, 5)), 0), axis=1)
    item = snapshot.item
    matrix[:, 17] = np.log1p(item.count30[products])
    matrix[:, 18] = np.log1p(item.count91[products])
    matrix[:, 19] = np.log1p(item.count365[products])
    matrix[:, 20] = np.log1p(item.count_all[products])
    matrix[:, 21] = item.five_all[products] / np.maximum(item.count_all[products], 1)
    matrix[:, 22] = (item.count30[products] + 1) / (item.count91[products] / 3.0 + 1)
    matrix[:, 23] = (item.count91[products] + 1) / (item.count365[products] / 4.0 + 1)
    matrix[:, 24] = item.rating_mean[products]
    matrix[:, 25] = item.verified_share[products]
    matrix[:, 26] = np.log1p(static.price[products])
    matrix[:, 27] = np.log1p(np.maximum(cutoff - item.first_day[products], 0))
    matrix[:, 28] = np.log1p(np.maximum(cutoff - item.last_day[products], 0))
    matrix[:, 29] = np.log1p(item.seasonal[products])
    matrix[:, 30] = np.log1p(static.brand_size[static.brand_code[products]])
    matrix[:, 31] = np.log1p(static.category_size[static.category_code[products]])
    user = snapshot.user
    matrix[:, 32] = np.log1p(user.count91[customers])
    matrix[:, 33] = np.log1p(user.count365[customers])
    matrix[:, 34] = np.log1p(user.count_all[customers])
    matrix[:, 35] = user.five_share[customers]
    matrix[:, 36] = user.rating_mean[customers]
    matrix[:, 37] = user.verified_share[customers]
    hist = snapshot.histories.product
    valid = hist >= 0
    hist_count = valid.sum(axis=1)
    hist_recency = np.where(valid[:, 0], cutoff - snapshot.histories.day[:, 0], 10000)
    hist_price = np.where(valid, static.price[np.maximum(hist, 0)], np.nan)
    hist_price_mean = np.nanmean(hist_price, axis=1)
    hist_price_std = np.nanstd(hist_price, axis=1)
    hist_price_mean = np.nan_to_num(hist_price_mean, nan=float(np.median(static.price)))
    hist_price_std = np.nan_to_num(hist_price_std)
    distinct_brand = np.zeros(batch, np.float32)
    distinct_category = np.zeros(batch, np.float32)
    brand_affinity = np.zeros((batch, width), np.float32)
    category_affinity = np.zeros((batch, width), np.float32)
    brand_count = np.zeros((batch, width), np.float32)
    category_count = np.zeros((batch, width), np.float32)
    seen = np.zeros((batch, width), np.float32)
    for row in range(batch):
        hp = hist[row, valid[row]]
        if len(hp) == 0:
            continue
        brands, bcounts = np.unique(static.brand_code[hp], return_counts=True)
        categories, ccounts = np.unique(static.category_code[hp], return_counts=True)
        distinct_brand[row] = len(brands)
        distinct_category[row] = len(categories)
        candidate_brands = static.brand_code[ids[row]]
        candidate_categories = static.category_code[ids[row]]
        for code, count in zip(brands, bcounts):
            mask = candidate_brands == code
            brand_affinity[row, mask] = 1
            brand_count[row, mask] = count
        for code, count in zip(categories, ccounts):
            mask = candidate_categories == code
            category_affinity[row, mask] = 1
            category_count[row, mask] = count
        seen[row] = np.isin(ids[row], hp)
    name_code = static.name_code[snapshot.customers]
    cohort_sum = np.bincount(static.name_code, weights=user.count_all, minlength=len(static.name_size))
    cohort_activity = cohort_sum[name_code] / np.maximum(static.name_size[name_code], 1)
    repeated = lambda x: np.repeat(x, width)
    matrix[:, 38] = repeated(hist_count)
    matrix[:, 39] = np.log1p(repeated(hist_recency))
    matrix[:, 40] = repeated(np.log1p(hist_price_mean))
    matrix[:, 41] = repeated(np.log1p(hist_price_std))
    matrix[:, 42] = repeated(distinct_brand)
    matrix[:, 43] = repeated(distinct_category)
    matrix[:, 44] = repeated(np.where(static.name_size[name_code] >= 5, np.log1p(static.name_size[name_code]), 0))
    matrix[:, 45] = repeated(np.where(static.name_size[name_code] >= 5, np.log1p(cohort_activity), 0))
    matrix[:, 46] = brand_affinity.reshape(-1)
    matrix[:, 47] = category_affinity.reshape(-1)
    matrix[:, 48] = np.log1p(np.abs(static.price[products] - repeated(hist_price_mean)))
    matrix[:, 49] = seen.reshape(-1)
    matrix[:, 50] = brand_count.reshape(-1)
    matrix[:, 51] = category_count.reshape(-1)
    matrix[:, 52] = np.log1p(np.maximum(cutoff - item.last_day[products], 0))
    matrix[:, 53] = np.log1p(np.maximum(cutoff - item.first_day[products], 0))
    matrix[:, 54] = matrix[:, 1] * matrix[:, 18]
    matrix[:, 55] = matrix[:, 4] * matrix[:, 7]
    return matrix


def candidate_arrays(snapshot: Snapshot, use_als: bool, use_semantic: bool, start: int, stop: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = snapshot.sources
    return union_candidates(
        source.cf_ids[start:stop], source.cf_scores[start:stop],
        source.als_ids[start:stop], source.als_scores[start:stop],
        source.sem_ids[start:stop], source.sem_scores[start:stop],
        source.affinity_ids[start:stop], source.affinity_scores[start:stop],
        snapshot.item.popularity, use_als, use_semantic, snapshot.histories.product[start:stop],
    )


def flatten_labels(labels: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.fromiter((len(x) for x in labels), dtype=np.int32, count=len(labels))
    offsets = np.empty(len(labels) + 1, np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    return np.concatenate(labels.to_numpy()).astype(np.int32), offsets


@njit(cache=True)
def recall_widths(pool: np.ndarray, labels: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    totals = np.zeros(3, np.float64)
    stamps = np.zeros(N_PRODUCTS, np.int32)
    positions = np.zeros(N_PRODUCTS, np.int16)
    requested = np.array([100, 300, 600], np.int32)
    for row in range(len(pool)):
        stamp = row + 1
        for rank in range(pool.shape[1]):
            product = pool[row, rank]
            if product >= 0:
                stamps[product] = stamp
                positions[product] = rank + 1
        denominator = offsets[row + 1] - offsets[row]
        for pos in range(offsets[row], offsets[row + 1]):
            product = labels[pos]
            if stamps[product] == stamp:
                rank = positions[product]
                for index in range(3):
                    if rank <= requested[index]:
                        totals[index] += 1.0 / denominator
    return totals / len(pool)


def source_recall(snapshot: Snapshot, labels: pd.Series) -> dict[str, dict[int, float]]:
    pools = {
        "directional": snapshot.sources.cf_ids,
        "als": snapshot.sources.als_ids,
        "semantic": snapshot.sources.sem_ids,
        "affinity": snapshot.sources.affinity_ids,
        "popularity": np.broadcast_to(snapshot.item.popularity[:600], (len(labels), 600)),
    }
    flat, offsets = flatten_labels(labels)
    result = {}
    for name, values in pools.items():
        measured = recall_widths(values, flat, offsets)
        result[name] = {width: float(value) for width, value in zip([100, 300, 600], measured)}
    return result


def union_recall(snapshot: Snapshot, labels: pd.Series, use_als: bool, use_semantic: bool) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(labels), 2000):
        stop = min(start + 2000, len(labels))
        ids, _, _, _ = candidate_arrays(snapshot, use_als, use_semantic, start, stop)
        flat, offsets = flatten_labels(labels.iloc[start:stop])
        total += recall_widths(ids, flat, offsets)[2] * (stop - start)
        count += stop - start
    return total / count


def training_matrix(snapshot: Snapshot, static: StaticData, labels: pd.Series, use_als: bool, use_semantic: bool, maximum_groups: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_parts = []
    target_parts = []
    weight_parts = []
    groups = []
    accepted = 0
    for start in range(0, len(labels), 1000):
        stop = min(start + 1000, len(labels))
        ids, ranks, scores, flags = candidate_arrays(snapshot, use_als, use_semantic, start, stop)
        local_snapshot = Snapshot(
            snapshot.cutoff,
            snapshot.customers[start:stop],
            Histories(*(x[start:stop] for x in [snapshot.histories.product, snapshot.histories.day, snapshot.histories.rating, snapshot.histories.verified])),
            snapshot.item,
            snapshot.user,
            snapshot.sources,
        )
        features = feature_matrix(local_snapshot, static, ids, ranks, scores, flags).reshape(stop - start, CANDIDATE_CAP, -1)
        for row, truth in enumerate(labels.iloc[start:stop]):
            positive = np.flatnonzero(np.isin(ids[row], np.asarray(truth)))
            if len(positive) == 0:
                continue
            negative = []
            for source in range(5):
                choices = np.flatnonzero(((flags[row] & (1 << source)) != 0) & ~np.isin(np.arange(CANDIDATE_CAP), positive))
                if len(choices):
                    positions = np.linspace(0, len(choices) - 1, min(8, len(choices)), dtype=np.int32)
                    for choice in choices[positions]:
                        if choice not in negative:
                            negative.append(int(choice))
            if len(negative) < 40:
                choices = np.flatnonzero(~np.isin(np.arange(CANDIDATE_CAP), np.r_[positive, negative]))
                positions = np.linspace(0, len(choices) - 1, min(40 - len(negative), len(choices)), dtype=np.int32)
                negative.extend(choices[positions].tolist())
            negative = np.asarray(negative[:40], np.int32)
            selected = np.r_[positive, negative]
            target = np.r_[np.ones(len(positive), np.float32), np.zeros(len(negative), np.float32)]
            feature_parts.append(features[row, selected])
            target_parts.append(target)
            weight_parts.append(np.full(len(selected), 1.0 / len(selected), np.float32))
            groups.append(len(selected))
            accepted += 1
            if accepted >= maximum_groups:
                break
        if accepted >= maximum_groups:
            break
    return (
        np.concatenate(feature_parts),
        np.concatenate(target_parts),
        np.concatenate(weight_parts),
        np.asarray(groups, np.int32),
    )


def train_ranker(train_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], valid_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None, trees: int | None) -> tuple[lgb.Booster, int]:
    x, y, weight, groups = train_data
    dataset = lgb.Dataset(x, label=y, weight=weight, group=groups, feature_name=FEATURE_NAMES, free_raw_data=True)
    valid_sets = None
    callbacks = [lgb.log_evaluation(0)]
    rounds = trees or 800
    if valid_data is not None:
        vx, vy, vw, vg = valid_data
        valid = lgb.Dataset(vx, label=vy, weight=vw, group=vg, feature_name=FEATURE_NAMES, reference=dataset, free_raw_data=True)
        valid_sets = [valid]
        callbacks.append(lgb.early_stopping(40, verbose=False))
    params = {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": [10],
        "lambdarank_truncation_level": 10,
        "num_leaves": 255,
        "learning_rate": 0.05,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": 1337,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "force_col_wise": True,
    }
    model = lgb.train(params, dataset, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    selected = model.best_iteration if model.best_iteration > 0 else rounds
    return model, selected


def predict(snapshot: Snapshot, static: StaticData, model: lgb.Booster | None, use_als: bool, use_semantic: bool) -> np.ndarray:
    result = np.empty((len(snapshot.customers), 10), np.int64)
    for start in range(0, len(snapshot.customers), 1000):
        stop = min(start + 1000, len(snapshot.customers))
        ids, ranks, scores, flags = candidate_arrays(snapshot, use_als, use_semantic, start, stop)
        local_snapshot = Snapshot(
            snapshot.cutoff,
            snapshot.customers[start:stop],
            Histories(*(x[start:stop] for x in [snapshot.histories.product, snapshot.histories.day, snapshot.histories.rating, snapshot.histories.verified])),
            snapshot.item,
            snapshot.user,
            snapshot.sources,
        )
        features = feature_matrix(local_snapshot, static, ids, ranks, scores, flags)
        if model is None:
            prediction = features[:, 16]
        else:
            prediction = model.predict(features, num_iteration=model.best_iteration or model.current_iteration())
        prediction = prediction.reshape(stop - start, CANDIDATE_CAP)
        best_source = ranks.min(axis=2)
        for row in range(stop - start):
            order = np.lexsort((ids[row], best_source[row], -prediction[row]))
            result[start + row] = ids[row, order[:10]]
    return result


@njit(cache=True)
def rank_directional_rrf(ids: np.ndarray, ranks: np.ndarray, scores: np.ndarray) -> np.ndarray:
    result = np.empty((len(ids), 10), np.int64)
    for row in range(len(ids)):
        top_score = np.full(10, -1e30, np.float32)
        top_rank = np.full(10, 1000, np.int16)
        top_product = np.full(10, -1, np.int32)
        for candidate in range(ids.shape[1]):
            product = ids[row, candidate]
            best_rank = 1000
            value = 0.0
            for source in range(5):
                rank = ranks[row, candidate, source]
                if rank < best_rank:
                    best_rank = rank
                if source == 0 and rank < 1000:
                    value += 1.0 / (60.0 + rank)
                elif source == 3 and rank < 1000:
                    value += 4.0 / (60.0 + rank)
            value += 0.02 * np.log1p(scores[row, candidate, 0])
            position = 10
            for rank in range(10):
                if value > top_score[rank] or (
                    value == top_score[rank]
                    and (best_rank < top_rank[rank] or (best_rank == top_rank[rank] and product < top_product[rank]))
                ):
                    position = rank
                    break
            if position < 10:
                for rank in range(9, position, -1):
                    top_score[rank] = top_score[rank - 1]
                    top_rank[rank] = top_rank[rank - 1]
                    top_product[rank] = top_product[rank - 1]
                top_score[position] = value
                top_rank[position] = best_rank
                top_product[position] = product
        result[row] = top_product
    return result


def predict_rrf(snapshot: Snapshot, use_als: bool, use_semantic: bool) -> np.ndarray:
    result = np.empty((len(snapshot.customers), 10), np.int64)
    for start in range(0, len(snapshot.customers), 5000):
        stop = min(start + 5000, len(snapshot.customers))
        ids, ranks, scores, _ = candidate_arrays(snapshot, use_als, use_semantic, start, stop)
        result[start:stop] = rank_directional_rrf(ids, ranks, scores)
    return result


# Debug

def debug_predictions(val_customers: np.ndarray, test_customers: np.ndarray, static: StaticData) -> tuple[np.ndarray, np.ndarray]:
    val_snapshot = build_snapshot("2015-10-01", val_customers[:3000], static, None, None, False)
    test_snapshot = build_snapshot("2016-01-01", test_customers[:3000], static, None, None, False)
    val_row = val_snapshot.item.popularity[:10].astype(np.int64)
    test_row = test_snapshot.item.popularity[:10].astype(np.int64)
    val = np.tile(val_row, (len(val_customers), 1))
    test = np.tile(test_row, (len(test_customers), 1))
    val[:3000] = predict(val_snapshot, static, None, False, False)
    test[:3000] = predict(test_snapshot, static, None, False, False)
    return val, test
