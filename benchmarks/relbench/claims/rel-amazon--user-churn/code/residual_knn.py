from __future__ import annotations

import fcntl
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import faiss
import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


VERSION = "lane3_residual_knn_v1"
FOLDS = (20, 24, 30)
FANOUTS = (1, 4, 8)
BUCKETS = 8
STATES = 12
PRODUCTS = 506012
GROUPS = 65521
CUSTOMERS = 1850193
HALF_LIFE = 730.0
RAW_STRENGTH = 50.0
RESIDUAL_STRENGTH = 25.0
BASE_COLUMNS = (
    "recency_days",
    "n_7",
    "n_14",
    "n_30",
    "n_60",
    "n_91",
    "n_182",
    "n_365",
    "n_730",
    "n_all",
    "days_active_91",
    "products_91",
    "products_365",
    "tenure_days",
    "gap_1",
    "gap_2",
    "gap_3",
    "rating_91",
    "rating_all",
    "verified_91",
    "text_length_91",
    "price_91",
    "context_product_volume_1",
    "context_product_momentum_1",
)
GRAPH_NAMES = tuple(
    f"graph_{name}_{fanout}"
    for fanout in FANOUTS
    for name in (
        "raw_mean",
        "raw_min",
        "raw_max",
        "raw_std",
        "state_eb_mean",
        "residual_mean",
        "decayed_residual_mean",
        "label_count_mean",
        "coverage",
        "last_rate",
        "own_disagreement",
    )
)
KNN_NAMES = (
    "knn_inverse_rate",
    "knn_age_decay_rate",
    "knn_top20_rate",
    "knn_top100_rate",
    "knn_min_distance",
    "knn_mean_distance20",
    "knn_mean_distance100",
    "knn_entropy",
    "knn_mean_age20",
    "knn_mean_age100",
    "knn_coverage",
    "knn_neighbors",
    "knn_fallback",
)


def cache_root() -> Path:
    path = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def task_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "tasks" / os.environ["RELBENCH_TASK"]


def feature_root() -> Path:
    return Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane1_churn_state_product_v1" / "features"


def announce(phase: str, started: float, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[lane3-residual] phase={phase} elapsed={time.time() - started:.1f}s{suffix}", flush=True)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default))
    os.replace(temporary, path)


def register_artifact(name: str, path: Path, description: str, content_key: str) -> None:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(registry.read_text()) if registry.exists() else []
        relative = str(path.relative_to(shared))
        if not any(row.get("path") == relative for row in rows):
            rows.append(
                {
                    "name": name,
                    "path": relative,
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py; every stage checks its versioned artifact before rebuilding.",
                }
            )
            temporary = registry.with_name(f"artifacts.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(rows, indent=2, sort_keys=True))
            os.replace(temporary, registry)
        fcntl.flock(lock, fcntl.LOCK_UN)


def origin_days() -> np.ndarray:
    rows = duckdb.sql(
        f"SELECT DISTINCT timestamp FROM read_parquet('{task_root() / 'train.parquet'}') ORDER BY timestamp"
    ).fetchnumpy()["timestamp"]
    return (rows.astype("datetime64[s]").astype(np.int64) // 86400).astype(np.int32)


def split_labels(split: str) -> np.ndarray:
    return duckdb.sql(
        f"SELECT churn FROM read_parquet('{task_root() / f'{split}.parquet'}', file_row_number=true) ORDER BY file_row_number"
    ).fetchnumpy()["churn"].astype(np.int8)


def feature_files(split: str, debug: bool = False) -> list[Path]:
    if split == "train":
        paths = sorted(feature_root().glob("train_*.parquet"))
        if debug:
            return [path for path in paths if "debug" in path.name]
        return [path for path in paths if "debug" not in path.name]
    return [feature_root() / f"{split}_00.parquet"]


@dataclass
class SeedData:
    split: str
    row_id: np.ndarray
    customer: np.ndarray
    origin: np.ndarray
    day: np.ndarray
    label: np.ndarray | None
    base: np.ndarray
    recency: np.ndarray
    activity: np.ndarray
    own_rate: np.ndarray
    product_text: np.ndarray
    semantic: np.ndarray


def load_seed_data(split: str, origins: np.ndarray, debug: bool = False) -> SeedData:
    columns = ["row_id", "customer_id", "origin_index", *BASE_COLUMNS, "history_label_mean"]
    frames = [pd.read_parquet(path, columns=columns) for path in feature_files(split, debug)]
    frame = pd.concat(frames, ignore_index=True)
    frame.sort_values("row_id", inplace=True, kind="stable")
    row_id = frame["row_id"].to_numpy(np.int64)
    if not debug and not np.array_equal(row_id, np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError(f"{split} lane-1 seed rows are not in original task order")
    customer = frame["customer_id"].to_numpy(np.int32)
    origin = frame["origin_index"].to_numpy(np.int16)
    base = frame[list(BASE_COLUMNS)].to_numpy(np.float32)
    recency = frame["recency_days"].to_numpy(np.float32)
    activity = frame["n_91"].to_numpy(np.float32)
    own_rate = frame["history_label_mean"].to_numpy(np.float32)
    if split == "train":
        all_labels = split_labels("train")
        label = all_labels[row_id]
        day = origins[origin]
    elif split == "val":
        label = split_labels("val")
        day = np.full(len(frame), int(pd.Timestamp("2015-10-01").timestamp() // 86400), dtype=np.int32)
    else:
        label = None
        day = np.full(len(frame), int(pd.Timestamp("2016-01-01").timestamp() // 86400), dtype=np.int32)
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    product_suffix = f"seed_{split}{'_debug' if debug and split == 'train' else ''}.npy"
    product_path = shared / "lane1_product_text_minilm_v1" / product_suffix
    if debug and not product_path.exists():
        product_path = shared / "lane1_product_text_minilm_v1" / "seed_train.npy"
        product_text = np.load(product_path, mmap_mode="r")[row_id]
    else:
        product_text = np.load(product_path, mmap_mode="r")
    semantic_path = shared / "lane2_cutoff_transformer_v1" / f"semantic_seed_{split}_ordered_v1.npy"
    if debug:
        semantic = np.load(semantic_path, mmap_mode="r")[row_id]
    else:
        semantic = np.load(semantic_path, mmap_mode="r")
    return SeedData(
        split=split,
        row_id=row_id,
        customer=customer,
        origin=origin,
        day=day,
        label=label,
        base=base,
        recency=recency,
        activity=activity,
        own_rate=own_rate,
        product_text=product_text,
        semantic=semantic,
    )


@njit(cache=True)
def recent_distinct_products(pointer, days, products, customers, seed_day, fanout):
    output = np.full((len(customers), fanout), -1, dtype=np.int32)
    for row in range(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        low = left
        high = right
        while low < high:
            middle = (low + high) // 2
            if days[middle] <= seed_day:
                low = middle + 1
            else:
                high = middle
        index = low - 1
        count = 0
        while index >= left and count < fanout:
            product = products[index]
            duplicate = False
            for previous in range(count):
                if output[row, previous] == product:
                    duplicate = True
                    break
            if not duplicate:
                output[row, count] = product
                count += 1
            index -= 1
    return output


def ensure_customer_history(debug: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    suffix = "debug" if debug else "full"
    root = cache_root() / f"customer_history_{suffix}"
    root.mkdir(parents=True, exist_ok=True)
    pointer_path = root / "pointer.npy"
    day_path = root / "day.npy"
    product_path = root / "product.npy"
    if all(path.exists() for path in (pointer_path, day_path, product_path)):
        return (
            np.load(pointer_path, mmap_mode="r"),
            np.load(day_path, mmap_mode="r"),
            np.load(product_path, mmap_mode="r"),
        )
    events = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane1_churn_state_product_v1" / ("events_debug.parquet" if debug else "events.parquet")
    frame = duckdb.sql(
        f"SELECT customer_id, CAST(epoch(review_time) / 86400 AS INTEGER) event_day, product_id FROM read_parquet('{events}') ORDER BY customer_id, review_time, product_id"
    ).fetchdf()
    customer = frame.pop("customer_id").to_numpy(np.int32)
    days = frame.pop("event_day").to_numpy(np.int32)
    products = frame.pop("product_id").to_numpy(np.int32)
    counts = np.bincount(customer, minlength=CUSTOMERS)
    pointer = np.empty(CUSTOMERS + 1, dtype=np.int64)
    pointer[0] = 0
    np.cumsum(counts, out=pointer[1:])
    for path, values in ((pointer_path, pointer), (day_path, days), (product_path, products)):
        temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npy")
        np.save(temporary, values)
        os.replace(temporary, path)
    register_artifact(
        f"{VERSION} customer product history {suffix}",
        root,
        "Customer-ordered review-day and product arrays for causal last-product fanouts.",
        f"{VERSION}:customer_history:{suffix}",
    )
    return (
        np.load(pointer_path, mmap_mode="r"),
        np.load(day_path, mmap_mode="r"),
        np.load(product_path, mmap_mode="r"),
    )


def product_metadata() -> tuple[np.ndarray, np.ndarray]:
    product = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db" / "product.parquet"
    frame = duckdb.sql(
        f"SELECT product_id, CAST(coalesce(hash(category) % 65521, 0) AS INTEGER) category_hash, CAST(coalesce(hash(brand) % 65521, 0) AS INTEGER) brand_hash FROM read_parquet('{product}') ORDER BY product_id"
    ).fetchdf()
    return frame["category_hash"].to_numpy(np.int32), frame["brand_hash"].to_numpy(np.int32)


def customer_buckets(customers: np.ndarray) -> np.ndarray:
    values = customers.astype(np.uint64)
    values ^= values >> np.uint64(16)
    values *= np.uint64(0x7FEB352D)
    values ^= values >> np.uint64(15)
    return (values & np.uint64(7)).astype(np.uint8)


def state_cells(recency: np.ndarray, activity: np.ndarray) -> np.ndarray:
    recency_cell = np.where(recency <= 14, 0, np.where(recency <= 30, 1, np.where(recency <= 60, 2, 3)))
    activity_cell = np.where(activity <= 1, 0, np.where(activity <= 3, 1, 2))
    return (recency_cell * 3 + activity_cell).astype(np.uint8)


@njit(cache=True)
def capture_residuals(states, labels, state_count, state_sum):
    output = np.empty((len(labels), BUCKETS), dtype=np.float32)
    for row in range(len(labels)):
        state = states[row]
        for bucket in range(BUCKETS):
            count = state_count[bucket, state]
            total = state_sum[bucket, state]
            if count <= 0:
                count = 0.0
                total = 0.0
                for other_state in range(STATES):
                    count += state_count[bucket, other_state]
                    total += state_sum[bucket, other_state]
            prior = total / count if count > 0 else 0.5
            output[row, bucket] = labels[row] - prior
    return output


@njit(cache=True)
def add_sources(
    recent_products,
    source_buckets,
    source_states,
    labels,
    residuals,
    source_weight,
    product_category,
    product_brand,
    state_count,
    state_sum,
    product_count,
    product_sum,
    product_residual,
    product_decay_count,
    product_decay_residual,
    category_count,
    category_sum,
    category_residual,
    category_decay_count,
    category_decay_residual,
    brand_count,
    brand_sum,
    brand_residual,
    brand_decay_count,
    brand_decay_residual,
):
    for row in range(len(labels)):
        source_bucket = source_buckets[row]
        state = source_states[row]
        label = labels[row]
        for bucket in range(BUCKETS):
            if bucket == source_bucket:
                continue
            residual = residuals[row, bucket]
            state_count[bucket, state] += 1.0
            state_sum[bucket, state] += label
            for position in range(recent_products.shape[1]):
                product = recent_products[row, position]
                if product < 0:
                    break
                category = product_category[product]
                brand = product_brand[product]
                product_count[bucket, product] += 1.0
                product_sum[bucket, product] += label
                product_residual[bucket, product] += residual
                product_decay_count[bucket, product] += source_weight
                product_decay_residual[bucket, product] += source_weight * residual
                category_count[bucket, category] += 1.0
                category_sum[bucket, category] += label
                category_residual[bucket, category] += residual
                category_decay_count[bucket, category] += source_weight
                category_decay_residual[bucket, category] += source_weight * residual
                brand_count[bucket, brand] += 1.0
                brand_sum[bucket, brand] += label
                brand_residual[bucket, brand] += residual
                brand_decay_count[bucket, brand] += source_weight
                brand_decay_residual[bucket, brand] += source_weight * residual


@njit(cache=True)
def emit_graph_features(
    recent_products,
    query_buckets,
    query_states,
    own_rate,
    product_category,
    product_brand,
    state_count,
    state_sum,
    product_count,
    product_sum,
    product_residual,
    product_decay_count,
    product_decay_residual,
    category_count,
    category_residual,
    category_decay_count,
    category_decay_residual,
    brand_count,
    brand_residual,
    brand_decay_count,
    brand_decay_residual,
):
    output = np.empty((len(query_buckets), len(FANOUTS) * 11), dtype=np.float32)
    for row in range(len(query_buckets)):
        bucket = query_buckets[row]
        state = query_states[row]
        state_n = state_count[bucket, state]
        state_y = state_sum[bucket, state]
        if state_n <= 0:
            state_n = 0.0
            state_y = 0.0
            for other_state in range(STATES):
                state_n += state_count[bucket, other_state]
                state_y += state_sum[bucket, other_state]
        state_prior = state_y / state_n if state_n > 0 else 0.5
        for fanout_index in range(len(FANOUTS)):
            fanout = FANOUTS[fanout_index]
            raw_total = 0.0
            raw_square = 0.0
            raw_minimum = 2.0
            raw_maximum = -1.0
            state_total = 0.0
            residual_total = 0.0
            decay_total = 0.0
            count_total = 0.0
            covered = 0.0
            last_rate = state_prior
            take = 0
            for position in range(fanout):
                product = recent_products[row, position]
                if product < 0:
                    break
                take += 1
                category = product_category[product]
                brand = product_brand[product]
                count = product_count[bucket, product]
                raw = product_sum[bucket, product] / count if count > 0 else state_prior
                state_rate = (product_sum[bucket, product] + RAW_STRENGTH * state_prior) / (count + RAW_STRENGTH)
                category_n = category_count[bucket, category]
                brand_n = brand_count[bucket, brand]
                category_value = category_residual[bucket, category] / category_n if category_n > 0 else 0.0
                brand_value = brand_residual[bucket, brand] / brand_n if brand_n > 0 else 0.0
                parent_n = category_n + brand_n
                parent = (category_value * category_n + brand_value * brand_n) / parent_n if parent_n > 0 else 0.0
                residual_value = (product_residual[bucket, product] + RESIDUAL_STRENGTH * parent) / (count + RESIDUAL_STRENGTH)
                category_decay_n = category_decay_count[bucket, category]
                brand_decay_n = brand_decay_count[bucket, brand]
                category_decay_value = category_decay_residual[bucket, category] / category_decay_n if category_decay_n > 0 else 0.0
                brand_decay_value = brand_decay_residual[bucket, brand] / brand_decay_n if brand_decay_n > 0 else 0.0
                parent_decay_n = category_decay_n + brand_decay_n
                parent_decay = (category_decay_value * category_decay_n + brand_decay_value * brand_decay_n) / parent_decay_n if parent_decay_n > 0 else 0.0
                decay_n = product_decay_count[bucket, product]
                decay_residual = (product_decay_residual[bucket, product] + RESIDUAL_STRENGTH * parent_decay) / (decay_n + RESIDUAL_STRENGTH)
                residual_rate = min(1.0, max(0.0, state_prior + residual_value))
                decay_rate = min(1.0, max(0.0, state_prior + decay_residual))
                raw_total += raw
                raw_square += raw * raw
                raw_minimum = min(raw_minimum, raw)
                raw_maximum = max(raw_maximum, raw)
                state_total += state_rate
                residual_total += residual_rate
                decay_total += decay_rate
                count_total += count
                covered += count > 0
                if position == 0:
                    last_rate = residual_rate
            if take <= 0:
                take = 1
                raw_total = state_prior
                raw_square = state_prior * state_prior
                raw_minimum = state_prior
                raw_maximum = state_prior
                state_total = state_prior
                residual_total = state_prior
                decay_total = state_prior
            raw_mean = raw_total / take
            base = fanout_index * 11
            output[row, base] = raw_mean
            output[row, base + 1] = raw_minimum
            output[row, base + 2] = raw_maximum
            output[row, base + 3] = math.sqrt(max(0.0, raw_square / take - raw_mean * raw_mean))
            output[row, base + 4] = state_total / take
            output[row, base + 5] = residual_total / take
            output[row, base + 6] = decay_total / take
            output[row, base + 7] = count_total / take
            output[row, base + 8] = covered / take
            output[row, base + 9] = last_rate
            output[row, base + 10] = residual_total / take - own_rate[row] if np.isfinite(own_rate[row]) else 0.0
    return output


def accumulator_arrays() -> dict[str, np.ndarray]:
    arrays = {
        "state_count": np.zeros((BUCKETS, STATES), dtype=np.float64),
        "state_sum": np.zeros((BUCKETS, STATES), dtype=np.float64),
    }
    for prefix, size in (("product", PRODUCTS), ("category", GROUPS), ("brand", GROUPS)):
        arrays[f"{prefix}_count"] = np.zeros((BUCKETS, size), dtype=np.float64)
        arrays[f"{prefix}_sum"] = np.zeros((BUCKETS, size), dtype=np.float64)
        arrays[f"{prefix}_residual"] = np.zeros((BUCKETS, size), dtype=np.float64)
        arrays[f"{prefix}_decay_count"] = np.zeros((BUCKETS, size), dtype=np.float64)
        arrays[f"{prefix}_decay_residual"] = np.zeros((BUCKETS, size), dtype=np.float64)
    return arrays


def capture_payload(data: SeedData, indices: np.ndarray, day: int, products: np.ndarray, arrays: dict) -> dict:
    states = state_cells(data.recency[indices], data.activity[indices])
    buckets = customer_buckets(data.customer[indices])
    labels = data.label[indices].astype(np.float32)
    residuals = capture_residuals(states, labels, arrays["state_count"], arrays["state_sum"])
    return {
        "available": int(day + 91),
        "day": int(day),
        "products": products,
        "buckets": buckets,
        "states": states,
        "labels": labels,
        "residuals": residuals,
    }


def apply_payload(payload: dict, arrays: dict, product_category: np.ndarray, product_brand: np.ndarray) -> None:
    source_weight = float(math.exp(math.log(2.0) * payload["day"] / HALF_LIFE))
    add_sources(
        payload["products"],
        payload["buckets"],
        payload["states"],
        payload["labels"],
        payload["residuals"],
        source_weight,
        product_category,
        product_brand,
        arrays["state_count"],
        arrays["state_sum"],
        arrays["product_count"],
        arrays["product_sum"],
        arrays["product_residual"],
        arrays["product_decay_count"],
        arrays["product_decay_residual"],
        arrays["category_count"],
        arrays["category_sum"],
        arrays["category_residual"],
        arrays["category_decay_count"],
        arrays["category_decay_residual"],
        arrays["brand_count"],
        arrays["brand_sum"],
        arrays["brand_residual"],
        arrays["brand_decay_count"],
        arrays["brand_decay_residual"],
    )


def query_graph(data: SeedData, indices: np.ndarray, products: np.ndarray, arrays: dict, product_category: np.ndarray, product_brand: np.ndarray) -> np.ndarray:
    return emit_graph_features(
        products,
        customer_buckets(data.customer[indices]),
        state_cells(data.recency[indices], data.activity[indices]),
        data.own_rate[indices],
        product_category,
        product_brand,
        arrays["state_count"],
        arrays["state_sum"],
        arrays["product_count"],
        arrays["product_sum"],
        arrays["product_residual"],
        arrays["product_decay_count"],
        arrays["product_decay_residual"],
        arrays["category_count"],
        arrays["category_residual"],
        arrays["category_decay_count"],
        arrays["category_decay_residual"],
        arrays["brand_count"],
        arrays["brand_residual"],
        arrays["brand_decay_count"],
        arrays["brand_decay_residual"],
    )


def ensure_graph_features(train: SeedData, val: SeedData | None, test: SeedData | None, origins: np.ndarray, debug: bool, started: float) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, dict]:
    suffix = "debug" if debug else "full"
    paths = {
        "train": cache_root() / f"graph_train_{suffix}.npy",
        "val": cache_root() / f"graph_val_{suffix}.npy",
        "test": cache_root() / f"graph_test_{suffix}.npy",
    }
    required = [paths["train"]] if debug else list(paths.values())
    diagnostic_path = cache_root() / f"graph_diagnostics_{suffix}.json"
    if all(path.exists() for path in required) and diagnostic_path.exists():
        diagnostics = json.loads(diagnostic_path.read_text())
        return (
            np.load(paths["train"], mmap_mode="r"),
            None if debug else np.load(paths["val"], mmap_mode="r"),
            None if debug else np.load(paths["test"], mmap_mode="r"),
            diagnostics,
        )
    build_started = time.time()
    pointer, event_days, event_products = ensure_customer_history(debug)
    product_category, product_brand = product_metadata()
    arrays = accumulator_arrays()
    temporary_paths = {}
    train_temp = paths["train"].with_name(f"graph_train_{suffix}.{os.getpid()}.tmp.npy")
    train_output = np.lib.format.open_memmap(train_temp, mode="w+", dtype=np.float32, shape=(len(train.customer), len(GRAPH_NAMES)))
    temporary_paths["train"] = train_temp
    if not debug:
        val_temp = paths["val"].with_name(f"graph_val_{suffix}.{os.getpid()}.tmp.npy")
        test_temp = paths["test"].with_name(f"graph_test_{suffix}.{os.getpid()}.tmp.npy")
        val_output = np.lib.format.open_memmap(val_temp, mode="w+", dtype=np.float32, shape=(len(val.customer), len(GRAPH_NAMES)))
        test_output = np.lib.format.open_memmap(test_temp, mode="w+", dtype=np.float32, shape=(len(test.customer), len(GRAPH_NAMES)))
        temporary_paths["val"] = val_temp
        temporary_paths["test"] = test_temp
    pending = []
    origin_rows = []
    for origin in np.unique(train.origin):
        indices = np.flatnonzero(train.origin == origin)
        day = int(origins[int(origin)])
        remaining = []
        for payload in pending:
            if payload["available"] <= day:
                apply_payload(payload, arrays, product_category, product_brand)
            else:
                remaining.append(payload)
        pending = remaining
        products = recent_distinct_products(pointer, event_days, event_products, train.customer[indices], day, 8)
        phase_started = time.time()
        train_output[indices] = query_graph(train, indices, products, arrays, product_category, product_brand)
        pending.append(capture_payload(train, indices, day, products, arrays))
        origin_rows.append(
            {
                "origin": int(origin),
                "rows": int(len(indices)),
                "available_sources": float(arrays["state_count"][0].sum()),
                "seconds": time.time() - phase_started,
            }
        )
    if not debug:
        val_day = int(val.day[0])
        remaining = []
        for payload in pending:
            if payload["available"] <= val_day:
                apply_payload(payload, arrays, product_category, product_brand)
            else:
                remaining.append(payload)
        pending = remaining
        val_indices = np.arange(len(val.customer), dtype=np.int64)
        val_products = recent_distinct_products(pointer, event_days, event_products, val.customer, val_day, 8)
        val_output[:] = query_graph(val, val_indices, val_products, arrays, product_category, product_brand)
        pending.append(capture_payload(val, val_indices, val_day, val_products, arrays))
        test_day = int(test.day[0])
        for payload in pending:
            if payload["available"] <= test_day:
                apply_payload(payload, arrays, product_category, product_brand)
        test_indices = np.arange(len(test.customer), dtype=np.int64)
        test_products = recent_distinct_products(pointer, event_days, event_products, test.customer, test_day, 8)
        test_output[:] = query_graph(test, test_indices, test_products, arrays, product_category, product_brand)
        val_output.flush()
        test_output.flush()
    train_output.flush()
    del train_output
    if not debug:
        del val_output, test_output
    gc.collect()
    for split, temporary in temporary_paths.items():
        os.replace(temporary, paths[split])
        register_artifact(
            f"{VERSION} residual graph {split} {suffix}",
            paths[split],
            "Eight-bucket customer-excluded state-residual product propagation features.",
            f"{VERSION}:graph:{split}:{suffix}",
        )
    total_rows = len(train.customer) + (0 if debug else len(val.customer) + len(test.customer))
    seconds = time.time() - build_started
    diagnostics = {
        "version": VERSION,
        "rows": total_rows,
        "seconds": seconds,
        "rows_per_minute": 60.0 * total_rows / max(seconds, 1e-6),
        "feature_count": len(GRAPH_NAMES),
        "origin_rows": origin_rows,
        "customer_exclusion_buckets": BUCKETS,
        "raw_eb_strength": RAW_STRENGTH,
        "residual_eb_strength": RESIDUAL_STRENGTH,
        "decay_half_life_days": HALF_LIFE,
        "source_maturity_days": 91,
    }
    write_json(diagnostic_path, diagnostics)
    announce("graph_features", started, f"rows={total_rows} rate={diagnostics['rows_per_minute']:.0f}/min")
    return (
        np.load(paths["train"], mmap_mode="r"),
        None if debug else np.load(paths["val"], mmap_mode="r"),
        None if debug else np.load(paths["test"], mmap_mode="r"),
        diagnostics,
    )


def rank_values(values: np.ndarray) -> np.ndarray:
    return (rankdata(np.asarray(values), method="average") / len(values)).astype(np.float32)


def run0014_oof(train: SeedData) -> np.ndarray:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    lane1 = np.load(shared / "lane1_widened_gates_v1" / "oof_939409618911d095.npy", mmap_mode="r")
    champion = np.load(shared / "lane2_champion_reproduction_v1" / "predictions_full.npz")
    output = np.full(len(train.customer), np.nan, dtype=np.float32)
    for fold in FOLDS:
        transformer = np.load(shared / "lane2_cutoff_transformer_v1" / f"fold_{fold}_full_ordered_v2.npz")
        indices = transformer["hold_indices"]
        lane2 = 0.8 * rank_values(champion["oof"][indices]) + 0.2 * rank_values(transformer["full"])
        output[indices] = 0.6 * rank_values(lane1[indices]) + 0.4 * rank_values(lane2)
    return output


def temporal_weights(origins: np.ndarray) -> np.ndarray:
    latest = float(np.max(origins))
    return np.clip(np.exp(-0.025 * (latest - origins)), 0.55, 1.0).astype(np.float32)


def train_graph_model(x: np.ndarray, y: np.ndarray, origins: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=500,
        learning_rate=0.035,
        num_leaves=63,
        min_child_samples=1000,
        colsample_bytree=0.8,
        reg_lambda=8.0,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        random_state=1337,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(x, y, sample_weight=temporal_weights(origins), callbacks=[lgb.log_evaluation(0)])
    return model


def graph_matrix(data: SeedData, graph: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(data.base, dtype=np.float32), np.asarray(graph, dtype=np.float32)], axis=1)


def load_lane1_modules():
    code = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0012" / "code"
    if str(code) not in sys.path:
        sys.path.insert(0, str(code))
    import renewal
    import product_text

    return renewal, product_text


def ensure_widened_matrices(started: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = {split: cache_root() / f"widened_{split}_v3.npy" for split in ("train", "val", "test")}
    metadata_path = cache_root() / "widened_v3.json"
    if all(path.exists() for path in paths.values()) and metadata_path.exists():
        return np.load(paths["train"], mmap_mode="r"), np.load(paths["val"], mmap_mode="r"), np.load(paths["test"], mmap_mode="r"), json.loads(metadata_path.read_text())
    renewal, product_text = load_lane1_modules()
    train_paths, _ = renewal.ensure_features("train")
    val_paths, _ = renewal.ensure_features("val")
    test_paths, _ = renewal.ensure_features("test")
    train_frame = renewal.load_feature_frame(train_paths)
    val_frame = renewal.load_feature_frame(val_paths)
    test_frame = renewal.load_feature_frame(test_paths)
    labels = renewal.labels_frame()
    y_train = labels.set_index("row_id")["churn"].loc[train_frame["row_id"]].to_numpy(np.int8)
    val_labels = renewal.validation_labels()
    phase_started = time.time()
    train_frame, val_frame, test_frame = renewal.add_causal_relational_targets(train_frame, y_train, val_frame, val_labels, test_frame)
    train_frame = renewal.add_renewal_transforms(train_frame)
    val_frame = renewal.add_renewal_transforms(val_frame)
    test_frame = renewal.add_renewal_transforms(test_frame)
    train_frame, val_frame, test_frame = renewal.add_customer_name_features(train_frame, val_frame, test_frame)
    train_frame, _ = product_text.add_product_text_features("train", train_frame)
    val_frame, _ = product_text.add_product_text_features("val", val_frame)
    test_frame, _ = product_text.add_product_text_features("test", test_frame)
    selected = renewal.feature_columns(train_frame, ("state", "product", "product_text"))
    feature_importance_path = cache_root() / "widened_feature_importance_v3.json"
    if feature_importance_path.exists():
        importance = json.loads(feature_importance_path.read_text())
    else:
        development = train_frame["origin_index"].to_numpy() < 20
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=320,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=1000,
            colsample_bytree=0.8,
            reg_lambda=8.0,
            n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
            random_state=1337,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
        x_development = renewal.matrix(train_frame.loc[development], selected)
        model.fit(
            x_development,
            y_train[development],
            sample_weight=renewal.temporal_weights(train_frame.loc[development, "origin_index"].to_numpy()),
            callbacks=[lgb.log_evaluation(0)],
        )
        importance = {column: float(value) for column, value in zip(selected, model.booster_.feature_importance(importance_type="gain"))}
        write_json(feature_importance_path, importance)
        del x_development, model
        gc.collect()
    columns = sorted(selected, key=lambda column: importance.get(column, 0.0), reverse=True)[:96]
    for split, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        values = renewal.matrix(frame, columns)
        temporary = paths[split].with_name(f"widened_{split}_v3.{os.getpid()}.tmp.npy")
        np.save(temporary, values)
        os.replace(temporary, paths[split])
        register_artifact(
            f"{VERSION} widened matrix {split}",
            paths[split],
            "The forward-gated 379-feature lane-1 matrix in original task order.",
            f"{VERSION}:widened_v3:{split}",
        )
        del values
    diagnostics = {
        "columns": columns,
        "feature_count": len(columns),
        "candidate_feature_count": len(selected),
        "rows": {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)},
        "seconds": time.time() - phase_started,
        "blocks": ["state", "product", "product_text"],
    }
    write_json(metadata_path, diagnostics)
    announce("widened_matrices", started, f"features={len(columns)} seconds={diagnostics['seconds']:.1f}")
    del train_frame, val_frame, test_frame
    gc.collect()
    return np.load(paths["train"], mmap_mode="r"), np.load(paths["val"], mmap_mode="r"), np.load(paths["test"], mmap_mode="r"), diagnostics


def fused_matrix(widened: np.ndarray, graph: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(widened, dtype=np.float32), np.asarray(graph, dtype=np.float32)], axis=1)


def auc_rows(y: np.ndarray, origin: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> list[dict]:
    rows = []
    for fold in FOLDS:
        selected = (origin == fold) & np.isfinite(baseline) & np.isfinite(candidate)
        if selected.any():
            rows.append(
                {
                    "fold": fold,
                    "count": int(selected.sum()),
                    "baseline_auc": float(roc_auc_score(y[selected], baseline[selected])),
                    "candidate_auc": float(roc_auc_score(y[selected], candidate[selected])),
                    "delta": float(roc_auc_score(y[selected], candidate[selected]) - roc_auc_score(y[selected], baseline[selected])),
                }
            )
    return rows


def ensure_graph_oof(train: SeedData, widened: np.ndarray, graph: np.ndarray, baseline: np.ndarray, started: float) -> tuple[np.ndarray, dict]:
    path = cache_root() / "graph_fused_oof_widened_v3.npy"
    diagnostic_path = cache_root() / "graph_fused_oof_widened_v3.json"
    if path.exists() and diagnostic_path.exists():
        return np.load(path, mmap_mode="r"), json.loads(diagnostic_path.read_text())
    x = fused_matrix(widened, graph)
    output = np.full(len(train.customer), np.nan, dtype=np.float32)
    fold_rows = []
    gate = None
    for fold in FOLDS:
        fit = train.origin < fold
        hold = train.origin == fold
        phase_started = time.time()
        model = train_graph_model(x[fit], train.label[fit], train.origin[fit])
        output[hold] = model.predict_proba(x[hold])[:, 1].astype(np.float32)
        baseline_auc = float(roc_auc_score(train.label[hold], baseline[hold]))
        fused_auc = float(roc_auc_score(train.label[hold], output[hold]))
        blends = []
        for weight in (0.05, 0.10):
            blended = (1.0 - weight) * rank_values(baseline[hold]) + weight * rank_values(output[hold])
            blends.append({"weight": weight, "auc": float(roc_auc_score(train.label[hold], blended)), "delta": float(roc_auc_score(train.label[hold], blended) - baseline_auc)})
        sparse = hold & (train.activity == 1)
        sparse_base = float(roc_auc_score(train.label[sparse], baseline[sparse]))
        sparse_fused = float(roc_auc_score(train.label[sparse], output[sparse]))
        row = {
            "fold": fold,
            "train_n": int(fit.sum()),
            "hold_n": int(hold.sum()),
            "baseline_auc": baseline_auc,
            "fused_auc": fused_auc,
            "fused_delta": fused_auc - baseline_auc,
            "fixed_blends": blends,
            "sparse_count": int(sparse.sum()),
            "sparse_baseline_auc": sparse_base,
            "sparse_fused_auc": sparse_fused,
            "seconds": time.time() - phase_started,
        }
        fold_rows.append(row)
        announce("graph_fold", started, f"fold={fold} fused_delta={row['fused_delta']:.6f} blend10_delta={blends[1]['delta']:.6f}")
        if fold == 20:
            positive_pooled = fused_auc > baseline_auc or any(item["delta"] > 0 for item in blends)
            no_pooled_regression = max([fused_auc, *[item["auc"] for item in blends]]) >= baseline_auc - 0.00005
            positive_sparse = sparse_fused > sparse_base
            gate = {
                "positive_pooled": positive_pooled,
                "positive_sparse": positive_sparse,
                "no_pooled_regression": no_pooled_regression,
                "continued": bool(positive_pooled or (positive_sparse and no_pooled_regression)),
            }
    np.save(path, output)
    diagnostics = {"folds": fold_rows, "fold20_gate": gate, "hyperparameters": {"trees": 500, "learning_rate": 0.035, "leaves": 63, "minimum_child": 1000, "feature_fraction": 0.8, "l2": 8.0}}
    write_json(diagnostic_path, diagnostics)
    diagnostics["input_feature_count"] = int(x.shape[1])
    register_artifact(f"{VERSION} graph-fused widened OOF", path, "Origins 20/24/30 residual graph plus widened LightGBM predictions.", f"{VERSION}:graph_oof:widened_v3")
    del x
    gc.collect()
    return np.load(path, mmap_mode="r"), diagnostics


@njit(cache=True)
def latest_rows(customers, days, eligible_day):
    latest = np.full(CUSTOMERS, -1, dtype=np.int64)
    latest_day = np.full(CUSTOMERS, -1, dtype=np.int32)
    for row in range(len(customers)):
        if days[row] + 91 <= eligible_day:
            customer = customers[row]
            if days[row] >= latest_day[customer]:
                latest_day[customer] = days[row]
                latest[customer] = row
    count = 0
    for customer in range(CUSTOMERS):
        count += latest[customer] >= 0
    output = np.empty(count, dtype=np.int64)
    position = 0
    for customer in range(CUSTOMERS):
        if latest[customer] >= 0:
            output[position] = latest[customer]
            position += 1
    return output


def sampled_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
    return indices[positions]


def vector_block(data: SeedData, indices: np.ndarray, pca: PCA) -> np.ndarray:
    base = np.nan_to_num(data.base[indices], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    product = np.asarray(data.product_text[indices, :24], dtype=np.float32)
    semantic = np.nan_to_num(np.asarray(data.semantic[indices], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    semantic_pca = pca.transform(semantic).astype(np.float32)
    return np.concatenate([base, product, semantic_pca], axis=1)


def fit_vectorizer(data: SeedData, source_indices: np.ndarray) -> tuple[PCA, np.ndarray, np.ndarray]:
    fit_indices = sampled_indices(source_indices, 160000)
    semantic = np.nan_to_num(np.asarray(data.semantic[fit_indices], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pca = PCA(n_components=16, svd_solver="randomized", random_state=1337)
    pca.fit(semantic)
    sample = vector_block(data, fit_indices, pca)
    median = np.median(sample, axis=0).astype(np.float32)
    lower = np.quantile(sample, 0.25, axis=0).astype(np.float32)
    upper = np.quantile(sample, 0.75, axis=0).astype(np.float32)
    scale = np.maximum(upper - lower, 1e-3).astype(np.float32)
    return pca, median, scale


def scaled_vectors(data: SeedData, indices: np.ndarray, pca: PCA, median: np.ndarray, scale: np.ndarray) -> np.ndarray:
    values = vector_block(data, indices, pca)
    values = np.nan_to_num((values - median) / scale, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(values, -8.0, 8.0).astype(np.float32)


def build_faiss(source: np.ndarray, nprobe: int) -> tuple[faiss.Index, dict]:
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    dimensions = source.shape[1]
    nlist = min(8192, max(64, len(source) // 40))
    quantizer = faiss.IndexFlatL2(dimensions)
    index = faiss.IndexIVFPQ(quantizer, dimensions, nlist, 8, 8)
    train_indices = sampled_indices(np.arange(len(source), dtype=np.int64), min(500000, len(source)))
    started = time.time()
    index.train(np.ascontiguousarray(source[train_indices]))
    index.add(np.ascontiguousarray(source))
    index.nprobe = nprobe
    return index, {"nlist": nlist, "m": 8, "bits": 8, "nprobe": nprobe, "train_rows": int(len(train_indices)), "source_rows": int(len(source)), "build_seconds": time.time() - started}


def benchmark_retrieval(index: faiss.Index, query: np.ndarray, candidates: int, seconds: float = 60.0) -> dict:
    started = time.time()
    rows = 0
    cursor = 0
    while cursor < len(query) and time.time() - started < seconds:
        end = min(cursor + 2048, len(query))
        index.search(np.ascontiguousarray(query[cursor:end]), candidates)
        rows += end - cursor
        cursor = end
    elapsed = time.time() - started
    return {"rows": rows, "seconds": elapsed, "queries_per_second": rows / max(elapsed, 1e-6), "candidates": candidates, "nprobe": int(index.nprobe)}


@njit(cache=True)
def summarize_neighbors(distances, neighbors, source_customer, source_label, source_day, query_customer, query_day, fallback):
    output = np.empty((len(query_customer), len(KNN_NAMES)), dtype=np.float32)
    log_two = math.log(2.0)
    for row in range(len(query_customer)):
        kept = 0
        labels = np.empty(100, dtype=np.float32)
        dists = np.empty(100, dtype=np.float32)
        ages = np.empty(100, dtype=np.float32)
        for position in range(neighbors.shape[1]):
            index = neighbors[row, position]
            if index < 0:
                continue
            if source_customer[index] == query_customer[row]:
                continue
            labels[kept] = source_label[index]
            dists[kept] = max(0.0, distances[row, position])
            ages[kept] = max(0.0, query_day - source_day[index])
            kept += 1
            if kept == 100:
                break
        base = fallback[row]
        if kept < 20:
            output[row, 0] = base
            output[row, 1] = base
            output[row, 2] = base
            output[row, 3] = base
            output[row, 4] = math.sqrt(dists[0]) if kept else 0.0
            output[row, 5] = output[row, 4]
            output[row, 6] = output[row, 4]
            output[row, 7] = -(base * math.log(max(base, 1e-6)) + (1.0 - base) * math.log(max(1.0 - base, 1e-6)))
            output[row, 8] = 0.0
            output[row, 9] = 0.0
            output[row, 10] = kept / max(1, neighbors.shape[1])
            output[row, 11] = kept
            output[row, 12] = 1.0
            continue
        top20 = 0.0
        top100 = 0.0
        inverse_total = 0.0
        inverse_label = 0.0
        age_total = 0.0
        age_label = 0.0
        distance20 = 0.0
        distance100 = 0.0
        age20 = 0.0
        age100 = 0.0
        for position in range(kept):
            label = labels[position]
            distance = math.sqrt(dists[position] + 1e-8)
            age = ages[position]
            inverse = 1.0 / (distance + 1e-3)
            age_weight = math.exp(-log_two * age / HALF_LIFE)
            top100 += label
            distance100 += distance
            age100 += age
            inverse_total += inverse
            inverse_label += inverse * label
            age_total += age_weight
            age_label += age_weight * label
            if position < 20:
                top20 += label
                distance20 += distance
                age20 += age
        rate = inverse_label / inverse_total
        entropy = -(rate * math.log(max(rate, 1e-6)) + (1.0 - rate) * math.log(max(1.0 - rate, 1e-6)))
        output[row, 0] = rate
        output[row, 1] = age_label / age_total
        output[row, 2] = top20 / 20.0
        output[row, 3] = top100 / kept
        output[row, 4] = math.sqrt(dists[0] + 1e-8)
        output[row, 5] = distance20 / 20.0
        output[row, 6] = distance100 / kept
        output[row, 7] = entropy
        output[row, 8] = age20 / 20.0
        output[row, 9] = age100 / kept
        output[row, 10] = kept / max(1, neighbors.shape[1])
        output[row, 11] = kept
        output[row, 12] = 0.0
    return output


def retrieve_target(source: SeedData, query: SeedData, query_indices: np.ndarray, target_day: int, fallback: np.ndarray, benchmark: bool, total_projected_queries: int, started: float) -> tuple[np.ndarray, dict]:
    source_indices = latest_rows(source.customer, source.day, target_day)
    pca, median, scale = fit_vectorizer(source, source_indices)
    source_vectors = scaled_vectors(source, source_indices, pca, median, scale)
    query_vectors = scaled_vectors(query, query_indices, pca, median, scale)
    index, index_diagnostics = build_faiss(source_vectors, 16)
    settings = {"nprobe": 16, "candidates": 512}
    benchmark_rows = None
    if benchmark:
        benchmark_rows = benchmark_retrieval(index, query_vectors, 512, 60.0)
        projected = total_projected_queries / max(benchmark_rows["queries_per_second"], 1e-6)
        benchmark_rows["projected_seconds"] = projected
        if projected > 5400.0:
            index.nprobe = 8
            reduced = benchmark_retrieval(index, query_vectors, 256, min(60.0, benchmark_rows["seconds"]))
            reduced_projected = total_projected_queries / max(reduced["queries_per_second"], 1e-6)
            reduced["projected_seconds"] = reduced_projected
            benchmark_rows["reduced"] = reduced
            settings = {"nprobe": 8, "candidates": 256}
            if reduced_projected > 5400.0:
                return np.column_stack([fallback, fallback, fallback, fallback, np.zeros((len(query_indices), 9), dtype=np.float32)]).astype(np.float32), {"dropped": True, "reason": "retrieval projection exceeded 90 minutes after one reduction", "benchmark": benchmark_rows, "index": index_diagnostics}
    phase_started = time.time()
    chunks = []
    for begin in range(0, len(query_indices), 4096):
        end = min(begin + 4096, len(query_indices))
        distances, neighbors = index.search(np.ascontiguousarray(query_vectors[begin:end]), settings["candidates"])
        chunks.append(
            summarize_neighbors(
                distances,
                neighbors,
                source.customer[source_indices],
                source.label[source_indices].astype(np.float32),
                source.day[source_indices],
                query.customer[query_indices[begin:end]],
                int(target_day),
                fallback[begin:end].astype(np.float32),
            )
        )
    output = np.concatenate(chunks, axis=0)
    diagnostics = {
        "dropped": False,
        "index": index_diagnostics,
        "settings": settings,
        "benchmark": benchmark_rows,
        "query_rows": int(len(query_indices)),
        "retrieval_seconds": time.time() - phase_started,
        "survival_rate": float(1.0 - output[:, 12].mean()),
        "mean_neighbors": float(output[:, 11].mean()),
        "semantic_pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    announce("knn_target", started, f"split={query.split} rows={len(query_indices)} qps={len(query_indices) / max(diagnostics['retrieval_seconds'], 1e-6):.1f}")
    return output, diagnostics


def combined_seed_data(train: SeedData, val: SeedData) -> SeedData:
    return SeedData(
        split="train_plus_val",
        row_id=np.arange(len(train.customer) + len(val.customer), dtype=np.int64),
        customer=np.concatenate([train.customer, val.customer]),
        origin=np.concatenate([train.origin, np.full(len(val.customer), 31, dtype=np.int16)]),
        day=np.concatenate([train.day, val.day]),
        label=np.concatenate([train.label, val.label]),
        base=np.concatenate([train.base, val.base]),
        recency=np.concatenate([train.recency, val.recency]),
        activity=np.concatenate([train.activity, val.activity]),
        own_rate=np.concatenate([train.own_rate, val.own_rate]),
        product_text=np.concatenate([np.asarray(train.product_text), np.asarray(val.product_text)]),
        semantic=np.concatenate([np.asarray(train.semantic), np.asarray(val.semantic)]),
    )


def ensure_knn_predictions(train: SeedData, val: SeedData, test: SeedData, graph_train: np.ndarray, graph_val: np.ndarray, graph_test: np.ndarray, origins: np.ndarray, started: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    paths = {
        "oof": cache_root() / "knn_oof.npy",
        "val": cache_root() / "knn_val.npy",
        "test": cache_root() / "knn_test.npy",
    }
    diagnostic_path = cache_root() / "knn_diagnostics.json"
    if all(path.exists() for path in paths.values()) and diagnostic_path.exists():
        return np.load(paths["oof"], mmap_mode="r"), np.load(paths["val"], mmap_mode="r"), np.load(paths["test"], mmap_mode="r"), json.loads(diagnostic_path.read_text())
    total_queries = sum(int((train.origin == fold).sum()) for fold in FOLDS) + len(val.customer) + len(test.customer)
    oof = np.full((len(train.customer), len(KNN_NAMES)), np.nan, dtype=np.float32)
    diagnostics = {"targets": [], "total_projected_queries": total_queries}
    dropped = False
    for fold in FOLDS:
        indices = np.flatnonzero(train.origin == fold)
        fallback = np.asarray(graph_train[indices, 4], dtype=np.float32)
        values, rows = retrieve_target(train, train, indices, int(origins[fold]), fallback, fold == 20, total_queries, started)
        oof[indices] = values
        rows["fold"] = fold
        diagnostics["targets"].append(rows)
        if rows["dropped"]:
            dropped = True
            break
    if dropped:
        for fold in FOLDS:
            indices = np.flatnonzero(train.origin == fold)
            fallback = np.asarray(graph_train[indices, 4], dtype=np.float32)
            oof[indices] = np.column_stack([fallback, fallback, fallback, fallback, np.zeros((len(indices), 9), dtype=np.float32)])
        val_values = np.column_stack([graph_val[:, 4], graph_val[:, 4], graph_val[:, 4], graph_val[:, 4], np.zeros((len(val.customer), 9), dtype=np.float32)]).astype(np.float32)
        test_values = np.column_stack([graph_test[:, 4], graph_test[:, 4], graph_test[:, 4], graph_test[:, 4], np.zeros((len(test.customer), 9), dtype=np.float32)]).astype(np.float32)
        diagnostics["dropped"] = True
    else:
        val_indices = np.arange(len(val.customer), dtype=np.int64)
        val_values, val_rows = retrieve_target(train, val, val_indices, int(val.day[0]), np.asarray(graph_val[:, 4], dtype=np.float32), False, total_queries, started)
        val_rows["split"] = "val"
        diagnostics["targets"].append(val_rows)
        combined = combined_seed_data(train, val)
        test_indices = np.arange(len(test.customer), dtype=np.int64)
        test_values, test_rows = retrieve_target(combined, test, test_indices, int(test.day[0]), np.asarray(graph_test[:, 4], dtype=np.float32), False, total_queries, started)
        test_rows["split"] = "test"
        diagnostics["targets"].append(test_rows)
        diagnostics["dropped"] = False
        del combined
    np.save(paths["oof"], oof)
    np.save(paths["val"], val_values)
    np.save(paths["test"], test_values)
    write_json(diagnostic_path, diagnostics)
    for name, path in paths.items():
        register_artifact(f"{VERSION} causal kNN {name}", path, "CPU FAISS external-customer retrieval predictions and diagnostics.", f"{VERSION}:knn:{name}")
    return np.load(paths["oof"], mmap_mode="r"), np.load(paths["val"], mmap_mode="r"), np.load(paths["test"], mmap_mode="r"), diagnostics


def prepared_weighted_auc(y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(prediction, kind="stable")
    sorted_prediction = prediction[order]
    starts = np.r_[0, np.flatnonzero(sorted_prediction[1:] != sorted_prediction[:-1]) + 1]
    return order, starts


def weighted_auc(y: np.ndarray, weights: np.ndarray, prepared: tuple[np.ndarray, np.ndarray]) -> float:
    order, starts = prepared
    ordered_y = y[order]
    ordered_weight = weights[order]
    positive = np.add.reduceat(ordered_weight * ordered_y, starts)
    negative = np.add.reduceat(ordered_weight * (1 - ordered_y), starts)
    total_positive = positive.sum()
    total_negative = negative.sum()
    cumulative_negative = np.cumsum(negative) - negative
    return float(np.sum(positive * (cumulative_negative + 0.5 * negative)) / (total_positive * total_negative))


def clustered_bootstrap(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, customers: np.ndarray, draws: int = 1000) -> dict:
    codes, uniques = pd.factorize(customers, sort=False)
    baseline_prepared = prepared_weighted_auc(y, baseline)
    candidate_prepared = prepared_weighted_auc(y, candidate)
    rng = np.random.default_rng(1337)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        cluster_weights = rng.poisson(1.0, len(uniques)).astype(np.float32)
        row_weights = cluster_weights[codes]
        differences[draw] = weighted_auc(y, row_weights, candidate_prepared) - weighted_auc(y, row_weights, baseline_prepared)
    return {
        "draws": draws,
        "observed_delta": float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
        "mean_delta": float(differences.mean()),
        "standard_error": float(differences.std(ddof=1)),
        "probability_improvement": float(np.mean(differences > 0.0)),
        "lower_10": float(np.quantile(differences, 0.1)),
        "upper_90": float(np.quantile(differences, 0.9)),
    }


def per_origin_rank_features(origins: np.ndarray, run14: np.ndarray, graph: np.ndarray, knn: np.ndarray, activity: np.ndarray, selected_folds: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    indices = []
    for fold in selected_folds:
        selected = np.flatnonzero((origins == fold) & np.isfinite(run14) & np.isfinite(graph) & np.isfinite(knn))
        first = rank_values(run14[selected])
        second = rank_values(graph[selected])
        third = rank_values(knn[selected])
        sparse = (activity[selected] == 1).astype(np.float32)
        rows.append(np.column_stack([first, second, third, third * sparse]).astype(np.float64))
        indices.append(selected)
    return np.concatenate(rows, axis=0), np.concatenate(indices)


def fit_nonnegative_stack(x: np.ndarray, y: np.ndarray, admitted: tuple[bool, bool]) -> dict:
    count = len(y)
    strength = 0.05

    def objective(parameters):
        intercept = parameters[0]
        weights = parameters[1:]
        score = intercept + x @ weights
        loss = np.mean(np.logaddexp(0.0, score) - y * score)
        penalty = np.sum(weights * weights) / (2.0 * strength * count)
        probability = expit(score)
        gradient_score = probability - y
        gradient = np.empty_like(parameters)
        gradient[0] = gradient_score.mean()
        gradient[1:] = x.T @ gradient_score / count + weights / (strength * count)
        return loss + penalty, gradient

    initial = np.array([math.log(y.mean() / (1.0 - y.mean())), 1.0, 0.05, 0.05, 0.0], dtype=np.float64)
    graph_cap = 0.10 if admitted[0] else 0.0
    knn_cap = 0.10 if admitted[1] else 0.0
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[(None, None), (0.0, None), (0.0, graph_cap), (0.0, knn_cap), (0.0, knn_cap)],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    return {
        "intercept": float(result.x[0]),
        "weights": result.x[1:].tolist(),
        "C": strength,
        "success": bool(result.success),
        "iterations": int(result.nit),
        "message": str(result.message),
        "development_admitted": {"graph": admitted[0], "knn": admitted[1]},
        "expert_coefficient_cap": 0.10,
    }


def stack_score(x: np.ndarray, stack: dict) -> np.ndarray:
    if stack.get("prediction_rule") == "fixed_rank_blend":
        graph_weight = float(stack.get("fixed_graph_weight", 0.0))
        knn_weight = float(stack.get("fixed_knn_weight", 0.0))
        return ((1.0 - graph_weight - knn_weight) * x[:, 0] + graph_weight * x[:, 1] + knn_weight * x[:, 2]).astype(np.float32)
    return expit(stack["intercept"] + x @ np.asarray(stack["weights"], dtype=np.float64)).astype(np.float32)


def slice_metrics(data: SeedData, indices: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> list[dict]:
    recency = data.recency[indices]
    activity = data.activity[indices]
    labels = data.label[indices]
    groups = {
        "activity_1": activity == 1,
        "activity_2_3": (activity >= 2) & (activity <= 3),
        "activity_4_plus": activity >= 4,
        "recency_0_14": recency <= 14,
        "recency_15_30": (recency > 14) & (recency <= 30),
        "recency_31_60": (recency > 30) & (recency <= 60),
        "recency_61_91": recency > 60,
    }
    rows = []
    for name, mask in groups.items():
        if mask.sum() and np.unique(labels[mask]).size == 2:
            base_auc = float(roc_auc_score(labels[mask], baseline[mask]))
            candidate_auc = float(roc_auc_score(labels[mask], candidate[mask]))
            rows.append({"stratum": name, "count": int(mask.sum()), "label_rate": float(labels[mask].mean()), "baseline_auc": base_auc, "candidate_auc": candidate_auc, "delta": candidate_auc - base_auc})
    return rows


def select_stack(train: SeedData, run14: np.ndarray, graph: np.ndarray, knn: np.ndarray, started: float) -> tuple[dict, dict]:
    admission = []
    admission_rows = []
    for name, expert in (("graph", graph), ("knn", knn)):
        deltas = []
        for fold in (20, 24):
            selected = train.origin == fold
            baseline_rank = rank_values(run14[selected])
            expert_rank = rank_values(expert[selected])
            blended = 0.95 * baseline_rank + 0.05 * expert_rank
            delta = float(roc_auc_score(train.label[selected], blended) - roc_auc_score(train.label[selected], baseline_rank))
            deltas.append(delta)
        accepted = min(deltas) > 0.0
        admission.append(accepted)
        admission_rows.append({"expert": name, "weight": 0.05, "fold_deltas": deltas, "accepted": accepted})
    development_x, development_indices = per_origin_rank_features(train.origin, run14, graph, knn, train.activity, (20, 24))
    stack = fit_nonnegative_stack(development_x, train.label[development_indices], (admission[0], admission[1]))
    stack["prediction_rule"] = "fixed_rank_blend"
    stack["fixed_graph_weight"] = 0.10 if admission[0] else 0.0
    stack["fixed_knn_weight"] = 0.05 if admission[1] else 0.0
    development_prediction = stack_score(development_x, stack)
    development_baseline = np.concatenate([rank_values(run14[train.origin == fold]) for fold in (20, 24)])
    confirmation_x, confirmation_indices = per_origin_rank_features(train.origin, run14, graph, knn, train.activity, (30,))
    confirmation_prediction = stack_score(confirmation_x, stack)
    confirmation_baseline = rank_values(run14[confirmation_indices])
    bootstrap = clustered_bootstrap(
        train.label[confirmation_indices],
        confirmation_baseline,
        confirmation_prediction,
        train.customer[confirmation_indices],
        draws=1000,
    )
    accepted = bootstrap["observed_delta"] > 0.0 and bootstrap["probability_improvement"] >= 0.8
    diagnostics = {
        "stack": stack,
        "development_expert_admission": admission_rows,
        "development_origins": [20, 24],
        "development_count": int(len(development_indices)),
        "development_baseline_auc": float(roc_auc_score(train.label[development_indices], development_baseline)),
        "development_candidate_auc": float(roc_auc_score(train.label[development_indices], development_prediction)),
        "confirmation_origin": 30,
        "confirmation_count": int(len(confirmation_indices)),
        "confirmation_baseline_auc": float(roc_auc_score(train.label[confirmation_indices], confirmation_baseline)),
        "confirmation_candidate_auc": float(roc_auc_score(train.label[confirmation_indices], confirmation_prediction)),
        "bootstrap": bootstrap,
        "accepted": bool(accepted),
        "slices": slice_metrics(train, confirmation_indices, confirmation_baseline, confirmation_prediction),
        "rank_correlations": {
            "run0014_graph": float(pd.Series(run14[confirmation_indices]).rank().corr(pd.Series(graph[confirmation_indices]).rank())),
            "run0014_knn": float(pd.Series(run14[confirmation_indices]).rank().corr(pd.Series(knn[confirmation_indices]).rank())),
            "graph_knn": float(pd.Series(graph[confirmation_indices]).rank().corr(pd.Series(knn[confirmation_indices]).rank())),
        },
    }
    announce("stack_confirmation", started, f"accepted={accepted} delta={bootstrap['observed_delta']:.6f} p={bootstrap['probability_improvement']:.3f}")
    return stack, diagnostics


def baseline_final() -> tuple[np.ndarray, np.ndarray]:
    root = cache_root()
    val = np.load(root / "run0014_val_predictions.npy", allow_pickle=False)
    test = np.load(root / "run0014_test_predictions.npy", allow_pickle=False)
    return val.astype(np.float32), test.astype(np.float32)


def final_stack_features(run14: np.ndarray, graph: np.ndarray, knn: np.ndarray, activity: np.ndarray) -> np.ndarray:
    first = rank_values(run14)
    second = rank_values(graph)
    third = rank_values(knn)
    sparse = (activity == 1).astype(np.float32)
    return np.column_stack([first, second, third, third * sparse]).astype(np.float64)


def fit_final_graph(train: SeedData, val: SeedData, test: SeedData, widened_train: np.ndarray, widened_val: np.ndarray, widened_test: np.ndarray, graph_train: np.ndarray, graph_val: np.ndarray, graph_test: np.ndarray, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    val_path = cache_root() / "graph_fused_val_widened_v3.npy"
    test_path = cache_root() / "graph_fused_test_widened_v3.npy"
    if val_path.exists() and test_path.exists():
        return np.load(val_path), np.load(test_path), {"cache": "hit"}
    x_train = fused_matrix(widened_train, graph_train)
    x_val = fused_matrix(widened_val, graph_val)
    model_a = train_graph_model(x_train, train.label, train.origin)
    val_prediction = model_a.predict_proba(x_val)[:, 1].astype(np.float32)
    announce("graph_model_a", started, f"train={len(train.label)} val={len(val_prediction)}")
    x_test = fused_matrix(widened_test, graph_test)
    x_b = np.concatenate([x_train, x_val], axis=0)
    y_b = np.concatenate([train.label, val.label])
    origin_b = np.concatenate([train.origin, np.full(len(val.label), 31, dtype=np.int16)])
    model_b = train_graph_model(x_b, y_b, origin_b)
    test_prediction = model_b.predict_proba(x_test)[:, 1].astype(np.float32)
    np.save(val_path, val_prediction)
    np.save(test_path, test_prediction)
    register_artifact(f"{VERSION} final widened graph predictions", val_path, "Model A widened graph-fused validation predictions.", f"{VERSION}:graph_final_val:widened_v3")
    register_artifact(f"{VERSION} final widened graph predictions", test_path, "Model B widened graph-fused test predictions.", f"{VERSION}:graph_final_test:widened_v3")
    diagnostics = {"cache": "miss", "model_a_train_rows": len(train.label), "model_b_train_rows": len(y_b), "validation_preserved_before_model_b": True}
    announce("graph_model_b", started, f"train_plus_val={len(y_b)} test={len(test_prediction)}")
    return val_prediction, test_prediction, diagnostics


def validation_resolution(labels: np.ndarray, candidate: np.ndarray, baseline: np.ndarray) -> dict:
    rng = np.random.default_rng(2026)
    scores = []
    differences = []
    for _ in range(100):
        indices = rng.integers(0, len(labels), len(labels))
        scores.append(float(roc_auc_score(labels[indices], candidate[indices])))
        differences.append(float(roc_auc_score(labels[indices], candidate[indices]) - roc_auc_score(labels[indices], baseline[indices])))
    return {
        "draws": 100,
        "candidate_auc": float(roc_auc_score(labels, candidate)),
        "baseline_auc": float(roc_auc_score(labels, baseline)),
        "candidate_standard_error": float(np.std(scores, ddof=1)),
        "paired_difference_mean": float(np.mean(differences)),
        "rank_correlation": float(pd.Series(candidate).rank().corr(pd.Series(baseline).rank())),
        "selection_use": "post-freeze diagnostic only",
    }


def debug_smoke(started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    origins = origin_days()
    train = load_seed_data("train", origins, debug=True)
    graph_train, _, _, graph_diagnostics = ensure_graph_features(train, None, None, origins, True, started)
    x = graph_matrix(train, graph_train)
    fold = int(np.max(train.origin))
    fit = train.origin < fold
    hold = train.origin == fold
    model = lgb.LGBMClassifier(objective="binary", n_estimators=20, learning_rate=0.05, num_leaves=31, min_child_samples=50, verbosity=-1, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")), random_state=1337)
    model.fit(x[fit], train.label[fit], callbacks=[lgb.log_evaluation(0)])
    prediction = model.predict_proba(x[hold])[:, 1]
    source_indices = np.flatnonzero(fit)[: min(4000, fit.sum())]
    query_indices = np.flatnonzero(hold)[: min(512, hold.sum())]
    semantic = np.nan_to_num(np.asarray(train.semantic[source_indices], dtype=np.float32))
    pca = PCA(n_components=16, random_state=1337).fit(semantic)
    source = vector_block(train, source_indices, pca)
    query = vector_block(train, query_indices, pca)
    index = faiss.IndexFlatL2(64)
    index.add(np.ascontiguousarray(source.astype(np.float32)))
    index.search(np.ascontiguousarray(query.astype(np.float32)), min(64, len(source)))
    val, test = baseline_final()
    diagnostics = {
        "debug": True,
        "graph": graph_diagnostics,
        "graph_fold": fold,
        "graph_hold_rows": int(hold.sum()),
        "graph_auc": float(roc_auc_score(train.label[hold], prediction)),
        "knn_source_rows": int(len(source)),
        "knn_query_rows": int(len(query)),
        "output_source": "banked legal run_0014 fallback",
    }
    announce("debug_complete", started, f"graph_rows={len(train.customer)} knn_queries={len(query)}")
    return val, test, diagnostics


def full_run(started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    origins = origin_days()
    train = load_seed_data("train", origins)
    val = load_seed_data("val", origins)
    test = load_seed_data("test", origins)
    announce("seed_caches", started, f"train={len(train.customer)} val={len(val.customer)} test={len(test.customer)}")
    graph_train, graph_val, graph_test, graph_diagnostics = ensure_graph_features(train, val, test, origins, False, started)
    widened_train, widened_val, widened_test, widened_diagnostics = ensure_widened_matrices(started)
    baseline_oof = run0014_oof(train)
    graph_oof, graph_gate = ensure_graph_oof(train, widened_train, graph_train, baseline_oof, started)
    knn_oof, knn_val, knn_test, knn_diagnostics = ensure_knn_predictions(train, val, test, graph_train, graph_val, graph_test, origins, started)
    stack, stack_diagnostics = select_stack(train, baseline_oof, graph_oof, knn_oof[:, 0], started)
    baseline_val, baseline_test = baseline_final()
    final_diagnostics = {"used_fallback": True}
    if stack_diagnostics["accepted"]:
        graph_val_prediction, graph_test_prediction, final_graph_diagnostics = fit_final_graph(train, val, test, widened_train, widened_val, widened_test, graph_train, graph_val, graph_test, started)
        val_features = final_stack_features(baseline_val, graph_val_prediction, knn_val[:, 0], val.activity)
        test_features = final_stack_features(baseline_test, graph_test_prediction, knn_test[:, 0], test.activity)
        val_prediction = stack_score(val_features, stack)
        test_prediction = stack_score(test_features, stack)
        final_diagnostics = {"used_fallback": False, "graph": final_graph_diagnostics}
    else:
        val_prediction = baseline_val.copy()
        test_prediction = baseline_test.copy()
    resolution = validation_resolution(val.label, val_prediction, baseline_val)
    diagnostics = {
        "debug": False,
        "version": VERSION,
        "banked_run0014_score": 0.7123846046555528,
        "graph_features": graph_diagnostics,
        "widened_features": widened_diagnostics,
        "graph_gate": graph_gate,
        "knn": knn_diagnostics,
        "stack_confirmation": stack_diagnostics,
        "final": final_diagnostics,
        "validation_resolution": resolution,
        "model_a_validation_source": "training labels only; all graph sources matured by query time and stack frozen on origins 20/24",
        "model_b_test_source": "historical train features unchanged; validation feature rows captured at V and labels admitted only after maturity at T",
        "validation_prediction_preserved_before_model_b": True,
        "source_customer_exclusion": "stable eight-bucket exclusion plus exact kNN customer-id removal",
    }
    return val_prediction, test_prediction, diagnostics


def run_residual_knn(debug: bool, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    if debug:
        return debug_smoke(started)
    return full_run(started)
