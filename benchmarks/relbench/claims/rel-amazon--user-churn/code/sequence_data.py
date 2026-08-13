from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numba
import numpy as np
import pyarrow.parquet as pq


INDEX_VERSION = "lane2_sequence_index_v3"
EPOCH_DAY = np.datetime64("1970-01-01", "D")


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / INDEX_VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]


def atomic_save(path: Path, array: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(temporary, path)


def task_arrays(split: str) -> dict[str, np.ndarray]:
    path = data_root() / "tasks" / os.environ["RELBENCH_TASK"] / f"{split}.parquet"
    names = ["timestamp", "customer_id"]
    if split != "test":
        names.append("churn")
    table = pq.read_table(path, columns=names)
    timestamp = table["timestamp"].to_numpy().astype("datetime64[D]")
    result = {
        "day": (timestamp - EPOCH_DAY).astype(np.int32),
        "customer": table["customer_id"].to_numpy().astype(np.int32),
    }
    result["target"] = (
        table["churn"].to_numpy().astype(np.int64)
        if split != "test"
        else np.full(len(timestamp), -1, dtype=np.int64)
    )
    return result


def selected_cutoffs() -> np.ndarray:
    train = task_arrays("train")["day"]
    latest = np.unique(train)[-8:]
    return np.concatenate((latest, np.unique(task_arrays("val")["day"]), np.unique(task_arrays("test")["day"]))).astype(np.int32)


def prepare_index() -> Path:
    root = cache_root()
    metadata_path = root / "index_metadata.json"
    required = [
        "customer.npy",
        "day.npy",
        "product.npy",
        "rating.npy",
        "verified.npy",
        "text_missing.npy",
        "doc_hash.npy",
        "gap.npy",
        "multiplicity.npy",
        "offsets.npy",
        "product_price.npy",
        "product_category.npy",
        "product_brand.npy",
        "product_text_missing.npy",
        "product_popularity.npy",
        "cutoffs.npy",
    ]
    if metadata_path.exists() and all((root / name).exists() for name in required):
        return root
    started = time.time()
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    connection.execute("SET preserve_insertion_order=false")
    review_path = data_root() / "db" / "review.parquet"
    query = f"""
        SELECT
            customer_id::INTEGER AS customer,
            date_diff('day', TIMESTAMP '1970-01-01', review_time)::INTEGER AS day,
            product_id::INTEGER AS product,
            rating::FLOAT AS rating,
            verified::UTINYINT AS verified,
            (review_text IS NULL OR length(trim(review_text)) = 0)::UTINYINT AS text_missing,
            hash(coalesce(summary, ''), coalesce(review_text, ''))::UBIGINT AS doc_hash
        FROM read_parquet('{review_path}')
        ORDER BY customer_id, review_time, product_id, doc_hash
    """
    arrays = connection.execute(query).fetchnumpy()
    customer = arrays["customer"].astype(np.int32, copy=False)
    day = arrays["day"].astype(np.int32, copy=False)
    product = arrays["product"].astype(np.int32, copy=False)
    rating = arrays["rating"].astype(np.float32, copy=False)
    verified = arrays["verified"].astype(np.uint8, copy=False)
    text_missing = arrays["text_missing"].astype(np.uint8, copy=False)
    doc_hash = arrays["doc_hash"].astype(np.uint64, copy=False)
    first_customer = np.empty(len(customer), dtype=np.bool_)
    first_customer[0] = True
    first_customer[1:] = customer[1:] != customer[:-1]
    gap = np.zeros(len(customer), dtype=np.int16)
    valid_gap = ~first_customer
    gap[valid_gap] = np.minimum(day[valid_gap] - day[np.flatnonzero(valid_gap) - 1], np.iinfo(np.int16).max).astype(np.int16)
    first_timestamp = first_customer.copy()
    first_timestamp[1:] |= day[1:] != day[:-1]
    starts = np.flatnonzero(first_timestamp)
    ends = np.append(starts[1:], len(customer))
    multiplicity = np.repeat(np.minimum(ends - starts, 255).astype(np.uint8), ends - starts)
    customer_count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{data_root() / 'db' / 'customer.parquet'}')").fetchone()[0])
    counts = np.bincount(customer, minlength=customer_count)
    offsets = np.empty(customer_count + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    for name, value in {
        "customer": customer,
        "day": day,
        "product": product,
        "rating": rating,
        "verified": verified,
        "text_missing": text_missing,
        "doc_hash": doc_hash,
        "gap": gap,
        "multiplicity": multiplicity,
        "offsets": offsets,
    }.items():
        atomic_save(root / f"{name}.npy", value)
    product_path = data_root() / "db" / "product.parquet"
    product_query = f"""
        SELECT
            product_id::INTEGER AS product,
            log(1 + greatest(coalesce(price, 0), 0))::FLOAT AS price,
            (hash(category) % 1024)::INTEGER AS category,
            (hash(coalesce(brand, '')) % 65536)::INTEGER AS brand,
            (description IS NULL OR length(trim(description)) = 0)::UTINYINT AS text_missing
        FROM read_parquet('{product_path}')
        ORDER BY product_id
    """
    products = connection.execute(product_query).fetchnumpy()
    for name in ["price", "category", "brand", "text_missing"]:
        atomic_save(root / f"product_{name}.npy", np.asarray(products[name]))
    cutoffs = selected_cutoffs()
    popularity = np.empty((len(cutoffs), len(products["product"])), dtype=np.int32)
    for index, cutoff in enumerate(cutoffs):
        popularity[index] = np.bincount(product[day <= cutoff], minlength=popularity.shape[1]).astype(np.int32)
    atomic_save(root / "product_popularity.npy", popularity)
    atomic_save(root / "cutoffs.npy", cutoffs)
    atomic_json(
        metadata_path,
        {
            "version": INDEX_VERSION,
            "reviews": int(len(customer)),
            "customers": customer_count,
            "products": int(len(products["product"])),
            "duplicate_customer_times": int((multiplicity > 1).sum()),
            "seconds": time.time() - started,
        },
    )
    print(f"[index] sorted {len(customer):,} reviews in {time.time() - started:.1f}s", flush=True)
    return root


def load_index() -> dict[str, np.ndarray]:
    root = prepare_index()
    names = [
        "customer",
        "day",
        "product",
        "rating",
        "verified",
        "text_missing",
        "doc_hash",
        "gap",
        "multiplicity",
        "offsets",
        "product_price",
        "product_category",
        "product_brand",
        "product_text_missing",
        "product_popularity",
        "cutoffs",
    ]
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


@numba.njit(parallel=True, cache=True)
def locate_seeds(
    customers: np.ndarray,
    cutoffs: np.ndarray,
    offsets: np.ndarray,
    event_days: np.ndarray,
    event_ratings: np.ndarray,
    event_verified: np.ndarray,
    event_missing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(customers)
    last = np.empty(size, dtype=np.int64)
    auxiliary = np.empty(size, dtype=np.int64)
    context = np.empty((size, 16), dtype=np.float32)
    recent = np.empty(size, dtype=np.int32)
    for row in numba.prange(size):
        customer = customers[row]
        cutoff = cutoffs[row]
        low = offsets[customer]
        high = offsets[customer + 1]
        left = low
        right = high
        while left < right:
            middle = (left + right) // 2
            if event_days[middle] <= cutoff:
                left = middle + 1
            else:
                right = middle
        end = left - 1
        last[row] = end
        depth = end - low + 1
        next_index = end + 1
        next_gap = 100000
        if next_index < high:
            next_gap = event_days[next_index] - cutoff
        if next_gap <= 7:
            auxiliary[row] = 0
        elif next_gap <= 30:
            auxiliary[row] = 1
        elif next_gap <= 60:
            auxiliary[row] = 2
        elif next_gap <= 91:
            auxiliary[row] = 3
        else:
            auxiliary[row] = 4
        bounds = np.empty(5, dtype=np.int64)
        windows = (7, 30, 91, 182, 365)
        for window_index in range(5):
            threshold = cutoff - windows[window_index]
            lo2 = low
            hi2 = end + 1
            while lo2 < hi2:
                middle = (lo2 + hi2) // 2
                if event_days[middle] <= threshold:
                    lo2 = middle + 1
                else:
                    hi2 = middle
            bounds[window_index] = lo2
        c7 = end - bounds[0] + 1
        c30 = end - bounds[1] + 1
        c91 = end - bounds[2] + 1
        c182 = end - bounds[3] + 1
        c365 = end - bounds[4] + 1
        recent[row] = c30
        recency = cutoff - event_days[end]
        gap_start = max(low + 1, end - 7)
        gap_sum = 0.0
        gap_count = 0
        rating_sum = 0.0
        verified_sum = 0.0
        extreme_sum = 0.0
        text_sum = 0.0
        mark_start = max(low, end - 7)
        for position in range(mark_start, end + 1):
            rating_sum += event_ratings[position]
            verified_sum += event_verified[position]
            extreme_sum += 1.0 if event_ratings[position] == 1.0 or event_ratings[position] == 5.0 else 0.0
            text_sum += 1.0 - event_missing[position]
        for position in range(gap_start, end + 1):
            gap_sum += event_days[position] - event_days[position - 1]
            gap_count += 1
        mark_count = end - mark_start + 1
        mean_gap = gap_sum / max(gap_count, 1)
        span = event_days[end] - event_days[low]
        context[row, 0] = np.log1p(depth)
        context[row, 1] = np.log1p(recency)
        context[row, 2] = np.log1p(c7)
        context[row, 3] = np.log1p(c30)
        context[row, 4] = np.log1p(c91)
        context[row, 5] = np.log1p(c182)
        context[row, 6] = np.log1p(c365)
        context[row, 7] = np.log1p(mean_gap)
        context[row, 8] = event_ratings[end] / 5.0
        context[row, 9] = rating_sum / (5.0 * mark_count)
        context[row, 10] = verified_sum / mark_count
        context[row, 11] = extreme_sum / mark_count
        context[row, 12] = text_sum / mark_count
        context[row, 13] = np.log1p(span)
        context[row, 14] = recency / (mean_gap + 1.0)
        context[row, 15] = 1.0 if depth > 32 else 0.0
    return last, auxiliary, context, recent


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float32)
    result[order] = (np.arange(len(values), dtype=np.float32) + 0.5) / max(len(values), 1)
    return result


@dataclass
class SeedSet:
    name: str
    customer: np.ndarray
    day: np.ndarray
    target: np.ndarray
    auxiliary: np.ndarray
    origin: np.ndarray
    last: np.ndarray
    depth: np.ndarray
    context: np.ndarray

    def subset(self, rows: np.ndarray, name: str | None = None) -> "SeedSet":
        return SeedSet(
            name or self.name,
            self.customer[rows],
            self.day[rows],
            self.target[rows],
            self.auxiliary[rows],
            self.origin[rows],
            self.last[rows],
            self.depth[rows],
            self.context[rows],
        )


def build_seed_set(split: str, index: dict[str, np.ndarray], latest_train_only: bool = True) -> SeedSet:
    raw = task_arrays(split)
    if split == "train" and latest_train_only:
        allowed = np.unique(raw["day"])[-8:]
        rows = np.flatnonzero(np.isin(raw["day"], allowed))
        raw = {name: value[rows] for name, value in raw.items()}
    cutoffs = np.asarray(index["cutoffs"])
    origin = np.searchsorted(cutoffs, raw["day"]).astype(np.int16)
    if np.any(cutoffs[origin] != raw["day"]):
        raise RuntimeError(f"unregistered origin in {split}")
    last, auxiliary, base_context, recent = locate_seeds(
        raw["customer"],
        raw["day"],
        np.asarray(index["offsets"]),
        np.asarray(index["day"]),
        np.asarray(index["rating"]),
        np.asarray(index["verified"]),
        np.asarray(index["text_missing"]),
    )
    if np.any(last < np.asarray(index["offsets"])[raw["customer"]]):
        raise RuntimeError(f"missing seed history in {split}")
    if np.any(np.asarray(index["day"])[last] > raw["day"]):
        raise RuntimeError(f"temporal safety failure in {split}")
    depth = last - np.asarray(index["offsets"])[raw["customer"]] + 1
    rank_columns = []
    for day in np.unique(raw["day"]):
        mask = raw["day"] == day
        group = np.flatnonzero(mask)
        columns = [base_context[group, 1], base_context[group, 0], recent[group], base_context[group, 14]]
        local = np.column_stack([percentile_rank(np.asarray(column)) for column in columns])
        if not rank_columns:
            ranks = np.empty((len(raw["day"]), 4), dtype=np.float32)
        ranks[group] = local
    context = np.concatenate((base_context, ranks), axis=1).astype(np.float32)
    return SeedSet(
        split,
        raw["customer"],
        raw["day"],
        raw["target"],
        auxiliary,
        origin,
        last,
        depth.astype(np.int32),
        context,
    )
