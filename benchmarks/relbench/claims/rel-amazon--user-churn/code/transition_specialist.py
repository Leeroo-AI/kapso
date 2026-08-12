from __future__ import annotations

import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


VERSION = "lane1_transition_specialist_v1"
CUSTOMERS = 1850193
PRODUCTS = 506012
GROUPS = 65521
WIDTH = 1 << 20
DIMENSIONS = 16
CHANNELS = 7
DEVELOPMENT = (25, 27, 28)
CONFIRMATION = 29
VAL_HASH = "945ded645fa8659fa118e2dd545da908119903d8032a3cb7f2969d50ae8a261f"
TEST_HASH = "020f4d7a58757b06aff8083c439c5a21a2c0f77722bb7af46fd40956166acc9f"
STRENGTHS = np.array([64.0, 64.0, 128.0, 128.0, 256.0, 256.0, 64.0], dtype=np.float32)
FEATURES_PER_CHANNEL = 10
STATE_FEATURES = 9
RAW_FEATURES = 7
SKETCH_FEATURES = 80
TRANSITION_FEATURES = CHANNELS * FEATURES_PER_CHANNEL + STATE_FEATURES + RAW_FEATURES + SKETCH_FEATURES


set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def shared_root() -> Path:
    return Path(os.environ["KAPSO_SHARED_CACHE_DIR"])


def task_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "tasks" / os.environ["RELBENCH_TASK"]


def announce(phase: str, started: float, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[transition-specialist] phase={phase} elapsed={time.time() - started:.1f}s{suffix}", flush=True)


def json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_value))
    os.replace(temporary, path)


def register_artifact(name: str, path: Path, description: str, content_key: str) -> None:
    registry = shared_root() / "artifacts.json"
    lock_path = shared_root() / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = json.loads(registry.read_text()) if registry.exists() else []
        relative = str(path.relative_to(shared_root()))
        if not any(row.get("path") == relative for row in rows):
            rows.append(
                {
                    "name": name,
                    "path": relative,
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py; the chronological transition pipeline extends missing versioned artifacts.",
                }
            )
            temporary = registry.with_name(f"artifacts.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(rows, indent=2, sort_keys=True))
            os.replace(temporary, registry)
        fcntl.flock(lock, fcntl.LOCK_UN)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def floor_predictions() -> tuple[np.ndarray, np.ndarray]:
    roots = [shared_root() / "full_candidate2_floor_v1", shared_root() / "lane3_tabm_v1" / "baseline"]
    for root in roots:
        val_path = root / "val_predictions.npy"
        test_path = root / "test_predictions.npy"
        if val_path.exists() and test_path.exists() and digest(val_path) == VAL_HASH and digest(test_path) == TEST_HASH:
            return np.load(val_path, allow_pickle=False).astype(np.float32), np.load(test_path, allow_pickle=False).astype(np.float32)
    raise RuntimeError("exact full_candidate2 floor is unavailable or hash-invalid")


def origin_days() -> np.ndarray:
    rows = duckdb.sql(
        f"SELECT DISTINCT timestamp FROM read_parquet('{task_root() / 'train.parquet'}') ORDER BY timestamp"
    ).fetchnumpy()["timestamp"]
    return (rows.astype("datetime64[s]").astype(np.int64) // 86400).astype(np.int32)


def split_labels(split: str) -> np.ndarray:
    return duckdb.sql(
        f"SELECT churn FROM read_parquet('{task_root() / f'{split}.parquet'}', file_row_number=true) ORDER BY file_row_number"
    ).fetchnumpy()["churn"].astype(np.int8)


def champion_feature_root() -> Path:
    return shared_root() / "lane3_churn_renewal_graph_v2" / "features"


def feature_files(split: str, debug: bool = False) -> list[Path]:
    root = champion_feature_root()
    if split == "train":
        paths = sorted(root.glob("train_[0-9][0-9].parquet"))
        if debug:
            return paths[:2]
        return paths
    return [root / f"{split}_00.parquet"]


def relation(paths: list[Path]) -> str:
    return "[" + ",".join(f"'{path}'" for path in paths) + "]"


def load_seed_metadata(split: str, origins: np.ndarray, debug: bool = False) -> dict[str, np.ndarray]:
    columns = "row_id, customer_id, origin_index, n_91, recency_days, tenure_days, n_all"
    frame = duckdb.sql(
        f"SELECT {columns} FROM read_parquet({relation(feature_files(split, debug))}) ORDER BY row_id"
    ).fetchdf()
    row_id = frame["row_id"].to_numpy(np.int64)
    if not debug and not np.array_equal(row_id, np.arange(len(frame), dtype=np.int64)):
        raise RuntimeError(f"{split} seed order is not original task order")
    origin = frame["origin_index"].to_numpy(np.int16)
    if split == "train":
        day = origins[origin]
    elif split == "val":
        origin[:] = len(origins)
        day = np.full(len(frame), int(pd.Timestamp("2015-10-01").timestamp() // 86400), dtype=np.int32)
    else:
        origin[:] = len(origins) + 1
        day = np.full(len(frame), int(pd.Timestamp("2016-01-01").timestamp() // 86400), dtype=np.int32)
    return {
        "row_id": row_id,
        "customer": frame["customer_id"].to_numpy(np.int32),
        "origin": origin,
        "day": day,
        "activity": frame["n_91"].to_numpy(np.int16),
        "recency": frame["recency_days"].to_numpy(np.float32),
        "tenure": frame["tenure_days"].to_numpy(np.float32),
        "depth": frame["n_all"].to_numpy(np.float32),
    }


def ensure_history_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inherited = shared_root() / "lane3_residual_knn_v1" / "customer_history_full"
    inherited_paths = [inherited / "pointer.npy", inherited / "day.npy", inherited / "product.npy"]
    if all(path.exists() for path in inherited_paths):
        return tuple(np.load(path, mmap_mode="r") for path in inherited_paths)
    root = cache_root() / "customer_history"
    root.mkdir(parents=True, exist_ok=True)
    paths = [root / "pointer.npy", root / "day.npy", root / "product.npy"]
    if all(path.exists() for path in paths):
        return tuple(np.load(path, mmap_mode="r") for path in paths)
    events = shared_root() / "lane3_churn_renewal_graph_v2" / "events.parquet"
    frame = duckdb.sql(
        f"SELECT customer_id, CAST(epoch(review_time) / 86400 AS INTEGER) event_day, product_id FROM read_parquet('{events}') ORDER BY customer_id, review_time, product_id"
    ).fetchdf()
    customers = frame["customer_id"].to_numpy(np.int32)
    counts = np.bincount(customers, minlength=CUSTOMERS)
    pointer = np.empty(CUSTOMERS + 1, dtype=np.int64)
    pointer[0] = 0
    np.cumsum(counts, out=pointer[1:])
    values = [pointer, frame["event_day"].to_numpy(np.int32), frame["product_id"].to_numpy(np.int32)]
    for path, value in zip(paths, values):
        np.save(path, value)
    register_artifact(f"{VERSION} customer history", root, "Customer-ordered review day and product arrays.", f"{VERSION}:history:v1")
    return tuple(np.load(path, mmap_mode="r") for path in paths)


def ensure_product_metadata() -> tuple[np.ndarray, np.ndarray]:
    path = cache_root() / "product_metadata.npz"
    if path.exists():
        values = np.load(path, allow_pickle=False)
        return values["category"], values["brand"]
    events = shared_root() / "lane3_churn_renewal_graph_v2" / "events.parquet"
    frame = duckdb.sql(
        f"SELECT product_id, first(category_level2_hash ORDER BY review_time) category, first(brand_hash ORDER BY review_time) brand FROM read_parquet('{events}') GROUP BY product_id ORDER BY product_id"
    ).fetchdf()
    category = np.zeros(PRODUCTS, dtype=np.int32)
    brand = np.zeros(PRODUCTS, dtype=np.int32)
    products = frame["product_id"].to_numpy(np.int32)
    category[products] = frame["category"].to_numpy(np.int32)
    brand[products] = frame["brand"].to_numpy(np.int32)
    temporary = path.with_name(f"product_metadata.{os.getpid()}.npz")
    np.savez_compressed(temporary, category=category, brand=brand)
    os.replace(temporary, path)
    register_artifact(f"{VERSION} product metadata", path, "Product category-level-2 and brand hashes used by transition states.", f"{VERSION}:product_metadata:v1")
    return category, brand


@njit(cache=True, parallel=True)
def extract_states(pointer, event_days, event_products, customers, seed_days, product_category, product_brand):
    output = np.full((len(customers), 16), -1, dtype=np.int32)
    for row in prange(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        low = left
        high = right
        seed_day = seed_days[row]
        while low < high:
            middle = (low + high) // 2
            if event_days[middle] <= seed_day:
                low = middle + 1
            else:
                high = middle
        last_index = low - 1
        if last_index < left:
            continue
        last_product = event_products[last_index]
        previous_index = last_index - 1
        raw_repeat = 0
        if previous_index >= left and event_products[previous_index] == last_product:
            raw_repeat = 1
        while previous_index >= left and event_products[previous_index] == last_product:
            previous_index -= 1
        last_category = product_category[last_product]
        last_brand = product_brand[last_product]
        output[row, 0] = last_product
        output[row, 2] = last_category
        output[row, 4] = last_brand
        output[row, 9] = raw_repeat
        output[row, 14] = event_days[last_index]
        if previous_index < left:
            output[row, 7] = 0
            output[row, 8] = 0
            output[row, 10] = 0
            output[row, 11] = 0
            output[row, 12] = 0
            output[row, 13] = 0
            continue
        previous_product = event_products[previous_index]
        previous_category = product_category[previous_product]
        previous_brand = product_brand[previous_product]
        gap = event_days[last_index] - event_days[previous_index]
        active_days = 0
        prior_day = -2147483647
        for index in range(previous_index + 1, last_index):
            day = event_days[index]
            if day != prior_day:
                active_days += 1
                prior_day = day
        output[row, 1] = previous_product
        output[row, 3] = previous_category
        output[row, 5] = previous_brand
        output[row, 6] = gap
        output[row, 7] = last_index - previous_index - 1
        output[row, 8] = active_days
        output[row, 10] = previous_product != last_product
        output[row, 11] = previous_category != last_category
        output[row, 12] = previous_brand != last_brand
        output[row, 13] = seed_day - event_days[previous_index] > 91
        output[row, 15] = event_days[previous_index]
    return output


def ensure_states(split: str, metadata: dict[str, np.ndarray], started: float) -> np.ndarray:
    path = cache_root() / f"state_{split}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    pointer, event_days, event_products = ensure_history_arrays()
    category, brand = ensure_product_metadata()
    phase = time.time()
    values = extract_states(pointer, event_days, event_products, metadata["customer"], metadata["day"], category, brand)
    temporary = path.with_name(f"state_{split}.{os.getpid()}.tmp.npy")
    np.save(temporary, values)
    os.replace(temporary, path)
    seconds = time.time() - phase
    rate = 60.0 * len(values) / max(seconds, 1e-6)
    register_artifact(f"{VERSION} {split} transition states", path, "Last and previous-distinct product/category/brand states in original seed order.", f"{VERSION}:state:{split}:v1")
    announce("transition_states", started, f"split={split} rows={len(values)} rate={rate:.0f}/min")
    del values
    gc.collect()
    return np.load(path, mmap_mode="r")


@njit(cache=True)
def create_event_transitions(pointer, event_days, event_products):
    out_day = np.empty(len(event_days), dtype=np.int32)
    out_source = np.empty(len(event_days), dtype=np.int32)
    out_destination = np.empty(len(event_days), dtype=np.int32)
    count = 0
    for customer in range(len(pointer) - 1):
        left = pointer[customer]
        right = pointer[customer + 1]
        if left >= right:
            continue
        previous = event_products[left]
        for index in range(left + 1, right):
            current = event_products[index]
            if current != previous:
                out_day[count] = event_days[index]
                out_source[count] = previous
                out_destination[count] = current
                count += 1
                previous = current
    return out_day[:count], out_source[:count], out_destination[:count]


def ensure_event_transitions(started: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = cache_root() / "event_transitions.npz"
    if path.exists():
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        return values["day"], values["source"], values["destination"]
    pointer, event_days, event_products = ensure_history_arrays()
    phase = time.time()
    day, source, destination = create_event_transitions(pointer, event_days, event_products)
    order = np.argsort(day, kind="stable")
    temporary = path.with_name(f"event_transitions.{os.getpid()}.npz")
    np.savez(temporary, day=day[order], source=source[order], destination=destination[order])
    os.replace(temporary, path)
    register_artifact(f"{VERSION} event transitions", path, "Chronological distinct-product review transitions for raw frequency and surprise.", f"{VERSION}:event_transitions:v1")
    announce("event_transitions", started, f"rows={len(day)} seconds={time.time() - phase:.1f}")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    return values["day"], values["source"], values["destination"]


@njit(cache=True)
def mix_index(first, second, salt):
    value = np.uint64(first + 2) * np.uint64(0x9E3779B185EBCA87)
    value ^= np.uint64(second + 2) * np.uint64(0xC2B2AE3D27D4EB4F)
    value ^= np.uint64(salt + 1) * np.uint64(0x165667B19E3779F9)
    value ^= value >> np.uint64(33)
    value *= np.uint64(0xFF51AFD7ED558CCD)
    value ^= value >> np.uint64(33)
    return int(value & np.uint64(WIDTH - 1))


@njit(cache=True)
def hash64(value, salt):
    result = np.uint64(value + 2) * np.uint64(0x9E3779B185EBCA87)
    result ^= np.uint64(salt + 1) * np.uint64(0xC2B2AE3D27D4EB4F)
    result ^= result >> np.uint64(30)
    result *= np.uint64(0xBF58476D1CE4E5B9)
    result ^= result >> np.uint64(27)
    result *= np.uint64(0x94D049BB133111EB)
    result ^= result >> np.uint64(31)
    return result


@njit(cache=True)
def popcount(value):
    count = 0
    while value:
        value &= value - np.uint64(1)
        count += 1
    return count


@njit(cache=True)
def distinct_estimate(bits, support):
    occupied = popcount(bits)
    if occupied <= 0:
        return 0.0
    if occupied >= 63:
        return float(max(support, 64))
    return -64.0 * math.log((64.0 - occupied) / 64.0)


@njit(cache=True)
def state_cell(recency, tenure, activity):
    recency_bin = 0
    if recency > 14:
        recency_bin = 1
    if recency > 30:
        recency_bin = 2
    if recency > 60:
        recency_bin = 3
    tenure_bin = 0
    if tenure > 91:
        tenure_bin = 1
    if tenure > 365:
        tenure_bin = 2
    activity_bin = 0
    if activity > 1:
        activity_bin = 1
    if activity > 3:
        activity_bin = 2
    return recency_bin * 9 + tenure_bin * 3 + activity_bin


@njit(cache=True)
def channel_keys(state, recency, tenure, activity):
    keys = np.empty(CHANNELS, dtype=np.int32)
    cell = state_cell(recency, tenure, activity)
    keys[0] = mix_index(state[3], state[2], 0)
    keys[1] = mix_index(state[5], state[4], 1)
    keys[2] = mix_index(state[3], state[0], 2)
    keys[3] = mix_index(state[1], state[2], 3)
    keys[4] = mix_index(state[1], state[0], 4)
    keys[5] = mix_index(state[0], cell, 5)
    keys[6] = mix_index(state[2], cell, 6)
    return keys


@njit(cache=True)
def update_raw_until(days, sources, destinations, position, query_day, pair_count, source_count, destination_count, source_bits):
    total = position
    while position < len(days) and days[position] <= query_day:
        source = sources[position]
        destination = destinations[position]
        key = mix_index(source, destination, 17)
        pair_count[key] += 1
        source_count[source] += 1
        destination_count[destination] += 1
        bit = int(hash64(destination, 19) & np.uint64(63))
        source_bits[source] |= np.uint64(1) << np.uint64(bit)
        position += 1
    return position, position


@njit(cache=True)
def update_accumulators(states, residuals, customers, source_days, recency, tenure, activity, query_day, counts, sums, decay180_sum, decay180_weight, decay730_sum, decay730_weight, last_day, customer_bits, forward, forward_weight, reverse, reverse_weight, category_forward, category_forward_weight, category_reverse, category_reverse_weight):
    log_two = math.log(2.0)
    for row in range(len(states)):
        state = states[row]
        keys = channel_keys(state, recency[row], tenure[row], activity[row])
        residual = residuals[row]
        age = max(0, query_day - source_days[row])
        source_weight180 = math.exp(-log_two * age / 180.0)
        source_weight730 = math.exp(-log_two * age / 730.0)
        for channel in range(CHANNELS):
            if channel < 5 and state[1] < 0:
                continue
            key = keys[channel]
            previous_day = last_day[channel, key]
            if previous_day >= 0 and previous_day < query_day:
                gap = query_day - previous_day
                decay180 = math.exp(-log_two * gap / 180.0)
                decay730 = math.exp(-log_two * gap / 730.0)
                decay180_sum[channel, key] *= decay180
                decay180_weight[channel, key] *= decay180
                decay730_sum[channel, key] *= decay730
                decay730_weight[channel, key] *= decay730
            counts[channel, key] += 1
            sums[channel, key] += residual
            decay180_sum[channel, key] += residual * source_weight180
            decay180_weight[channel, key] += source_weight180
            decay730_sum[channel, key] += residual * source_weight730
            decay730_weight[channel, key] += source_weight730
            last_day[channel, key] = query_day
            bit = int(hash64(customers[row], channel + 31) & np.uint64(63))
            customer_bits[channel, key] |= np.uint64(1) << np.uint64(bit)
        previous_product = state[1]
        last_product = state[0]
        previous_category = state[3]
        last_category = state[2]
        if previous_product >= 0:
            for dimension in range(DIMENSIONS):
                sign_destination = 1.0 if hash64(last_product, dimension + 101) & np.uint64(1) else -1.0
                sign_source = 1.0 if hash64(previous_product, dimension + 137) & np.uint64(1) else -1.0
                sign_category_destination = 1.0 if hash64(last_category, dimension + 173) & np.uint64(1) else -1.0
                sign_category_source = 1.0 if hash64(previous_category, dimension + 211) & np.uint64(1) else -1.0
                forward[previous_product, dimension] += sign_destination * residual
                reverse[last_product, dimension] += sign_source * residual
                category_forward[previous_category, dimension] += sign_category_destination * residual
                category_reverse[last_category, dimension] += sign_category_source * residual
            forward_weight[previous_product] += 1.0
            reverse_weight[last_product] += 1.0
            category_forward_weight[previous_category] += 1.0
            category_reverse_weight[last_category] += 1.0


@njit(cache=True)
def prior_values(recency, tenure, activity, state_count, state_sum, global_count, global_sum):
    output = np.empty(len(recency), dtype=np.float32)
    global_mean = global_sum / global_count if global_count > 0 else 0.5
    for row in range(len(recency)):
        cell = state_cell(recency[row], tenure[row], activity[row])
        output[row] = (state_sum[cell] + 32.0 * global_mean) / (state_count[cell] + 32.0)
    return output


@njit(cache=True)
def update_state_prior(labels, recency, tenure, activity, state_count, state_sum):
    for row in range(len(labels)):
        cell = state_cell(recency[row], tenure[row], activity[row])
        state_count[cell] += 1
        state_sum[cell] += labels[row]


@njit(cache=True, parallel=True)
def query_features(states, customers, recency, tenure, activity, pair_count, source_count, destination_count, source_bits, raw_total, counts, sums, decay180_sum, decay180_weight, decay730_sum, decay730_weight, customer_bits, forward, forward_weight, reverse, reverse_weight, category_forward, category_forward_weight, category_reverse, category_reverse_weight):
    output = np.zeros((len(states), TRANSITION_FEATURES), dtype=np.float32)
    for row in prange(len(states)):
        state = states[row]
        keys = channel_keys(state, recency[row], tenure[row], activity[row])
        direct = np.zeros(CHANNELS, dtype=np.float32)
        usable = np.zeros(CHANNELS, dtype=np.int8)
        position = 0
        for channel in range(CHANNELS):
            missing = channel < 5 and state[1] < 0
            key = keys[channel]
            support = counts[channel, key]
            distinct = distinct_estimate(customer_bits[channel, key], support)
            strength = STRENGTHS[channel]
            eb = sums[channel, key] / (support + strength)
            decayed180 = decay180_sum[channel, key] / (decay180_weight[channel, key] + strength)
            decayed730 = decay730_sum[channel, key] / (decay730_weight[channel, key] + strength)
            direct[channel] = eb
            usable[channel] = distinct >= 8.0 and not missing
            selected = eb
            backoff = 0.0
            if not usable[channel]:
                backoff = 1.0
                if channel in (2, 3):
                    selected = direct[0]
                elif channel == 4:
                    selected = 0.5 * (direct[2] + direct[3])
                    if not usable[2] and not usable[3]:
                        selected = direct[0]
                        backoff = 2.0
                elif channel == 5:
                    category_key = keys[6]
                    selected = sums[6, category_key] / (counts[6, category_key] + STRENGTHS[6])
                elif channel in (0, 1, 6):
                    selected = 0.0
                    backoff = 2.0
            output[row, position] = math.log1p(support)
            output[row, position + 1] = math.log1p(distinct)
            output[row, position + 2] = 1.0 if usable[channel] else 0.0
            output[row, position + 3] = eb
            output[row, position + 4] = decayed180
            output[row, position + 5] = decayed730
            output[row, position + 6] = decayed180 - decayed730
            output[row, position + 7] = 1.0 if missing else 0.0
            output[row, position + 8] = backoff
            output[row, position + 9] = selected
            position += FEATURES_PER_CHANNEL
        gap = state[6]
        output[row, position] = math.log1p(max(gap, 0))
        output[row, position + 1] = math.log1p(max(state[7], 0))
        output[row, position + 2] = math.log1p(max(state[8], 0))
        output[row, position + 3] = state[9]
        output[row, position + 4] = state[10]
        output[row, position + 5] = state[11]
        output[row, position + 6] = state[12]
        output[row, position + 7] = state[13]
        output[row, position + 8] = state[1] < 0
        position += STATE_FEATURES
        previous_product = state[1]
        last_product = state[0]
        if previous_product >= 0:
            pair_key = mix_index(previous_product, last_product, 17)
            pair = pair_count[pair_key]
            source = source_count[previous_product]
            destination = destination_count[last_product]
            fanout = distinct_estimate(source_bits[previous_product], source)
            conditional = (pair + 0.5) / (source + max(fanout, 1.0))
            lift = (pair + 0.5) * max(raw_total, 1) / ((source + 1.0) * (destination + 1.0))
            output[row, position] = math.log1p(pair)
            output[row, position + 1] = math.log1p(source)
            output[row, position + 2] = math.log1p(destination)
            output[row, position + 3] = math.log1p(fanout)
            output[row, position + 4] = -math.log(max(conditional, 1e-9))
            output[row, position + 5] = math.log(max(lift, 1e-9))
            output[row, position + 6] = pair / max(source, 1)
        position += RAW_FEATURES
        product_forward_norm = 0.0
        product_reverse_norm = 0.0
        product_dot = 0.0
        category_forward_norm = 0.0
        category_reverse_norm = 0.0
        category_dot = 0.0
        if previous_product >= 0:
            prior_category = state[3]
            last_category = state[2]
            product_forward_denominator = forward_weight[previous_product] + 256.0
            product_reverse_denominator = reverse_weight[last_product] + 256.0
            category_forward_denominator = category_forward_weight[prior_category] + 64.0
            category_reverse_denominator = category_reverse_weight[last_category] + 64.0
            for dimension in range(DIMENSIONS):
                product_forward_value = forward[previous_product, dimension] / product_forward_denominator
                product_reverse_value = reverse[last_product, dimension] / product_reverse_denominator
                category_forward_value = category_forward[prior_category, dimension] / category_forward_denominator
                category_reverse_value = category_reverse[last_category, dimension] / category_reverse_denominator
                output[row, position + dimension] = product_forward_value
                output[row, position + DIMENSIONS + dimension] = product_reverse_value
                output[row, position + 2 * DIMENSIONS + dimension] = category_forward_value
                output[row, position + 3 * DIMENSIONS + dimension] = category_reverse_value
                product_forward_norm += product_forward_value * product_forward_value
                product_reverse_norm += product_reverse_value * product_reverse_value
                product_dot += product_forward_value * product_reverse_value
                category_forward_norm += category_forward_value * category_forward_value
                category_reverse_norm += category_reverse_value * category_reverse_value
                category_dot += category_forward_value * category_reverse_value
            product_forward_norm = math.sqrt(product_forward_norm)
            product_reverse_norm = math.sqrt(product_reverse_norm)
            category_forward_norm = math.sqrt(category_forward_norm)
            category_reverse_norm = math.sqrt(category_reverse_norm)
            metric = position + 4 * DIMENSIONS
            output[row, metric] = product_forward_norm
            output[row, metric + 1] = product_reverse_norm
            output[row, metric + 2] = product_dot
            output[row, metric + 3] = product_dot / max(product_forward_norm * product_reverse_norm, 1e-9)
            output[row, metric + 4] = category_forward_norm
            output[row, metric + 5] = category_reverse_norm
            output[row, metric + 6] = category_dot
            output[row, metric + 7] = category_dot / max(category_forward_norm * category_reverse_norm, 1e-9)
            output[row, metric + 8] = math.log1p(forward_weight[previous_product])
            output[row, metric + 9] = math.log1p(reverse_weight[last_product])
            output[row, metric + 10] = math.log1p(category_forward_weight[prior_category])
            output[row, metric + 11] = math.log1p(category_reverse_weight[last_category])
            destination_sign_sum = 0.0
            source_sign_sum = 0.0
            category_destination_sign_sum = 0.0
            category_source_sign_sum = 0.0
            for dimension in range(DIMENSIONS):
                destination_sign_sum += output[row, position + dimension] * (1.0 if hash64(last_product, dimension + 101) & np.uint64(1) else -1.0)
                source_sign_sum += output[row, position + DIMENSIONS + dimension] * (1.0 if hash64(previous_product, dimension + 137) & np.uint64(1) else -1.0)
                category_destination_sign_sum += output[row, position + 2 * DIMENSIONS + dimension] * (1.0 if hash64(last_category, dimension + 173) & np.uint64(1) else -1.0)
                category_source_sign_sum += output[row, position + 3 * DIMENSIONS + dimension] * (1.0 if hash64(prior_category, dimension + 211) & np.uint64(1) else -1.0)
            output[row, metric + 12] = destination_sign_sum / DIMENSIONS
            output[row, metric + 13] = source_sign_sum / DIMENSIONS
            output[row, metric + 14] = category_destination_sign_sum / DIMENSIONS
            output[row, metric + 15] = category_source_sign_sum / DIMENSIONS
    return output


def empty_accumulators() -> dict[str, np.ndarray]:
    return {
        "counts": np.zeros((CHANNELS, WIDTH), dtype=np.int32),
        "sums": np.zeros((CHANNELS, WIDTH), dtype=np.float32),
        "decay180_sum": np.zeros((CHANNELS, WIDTH), dtype=np.float32),
        "decay180_weight": np.zeros((CHANNELS, WIDTH), dtype=np.float32),
        "decay730_sum": np.zeros((CHANNELS, WIDTH), dtype=np.float32),
        "decay730_weight": np.zeros((CHANNELS, WIDTH), dtype=np.float32),
        "last_day": np.full((CHANNELS, WIDTH), -1, dtype=np.int32),
        "customer_bits": np.zeros((CHANNELS, WIDTH), dtype=np.uint64),
        "forward": np.zeros((PRODUCTS, DIMENSIONS), dtype=np.float32),
        "forward_weight": np.zeros(PRODUCTS, dtype=np.float32),
        "reverse": np.zeros((PRODUCTS, DIMENSIONS), dtype=np.float32),
        "reverse_weight": np.zeros(PRODUCTS, dtype=np.float32),
        "category_forward": np.zeros((GROUPS, DIMENSIONS), dtype=np.float32),
        "category_forward_weight": np.zeros(GROUPS, dtype=np.float32),
        "category_reverse": np.zeros((GROUPS, DIMENSIONS), dtype=np.float32),
        "category_reverse_weight": np.zeros(GROUPS, dtype=np.float32),
    }


def build_transition_features(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], train_state: np.ndarray, val_state: np.ndarray, test_state: np.ndarray, labels: np.ndarray, val_labels: np.ndarray, origins: np.ndarray, started: float) -> tuple[dict[str, np.ndarray], dict]:
    paths = {split: cache_root() / f"transition_{split}_activity1.npy" for split in ("train", "val", "test")}
    index_paths = {split: cache_root() / f"activity1_rows_{split}.npy" for split in ("train", "val", "test")}
    diagnostic_path = cache_root() / "transition_features.json"
    if all(path.exists() for path in [*paths.values(), *index_paths.values(), diagnostic_path]):
        return {
            **{f"x_{split}": np.load(path, mmap_mode="r") for split, path in paths.items()},
            **{f"rows_{split}": np.load(path, mmap_mode="r") for split, path in index_paths.items()},
        }, json.loads(diagnostic_path.read_text())
    event_day, event_source, event_destination = ensure_event_transitions(started)
    pair_count = np.zeros(WIDTH, dtype=np.int32)
    source_count = np.zeros(PRODUCTS, dtype=np.int32)
    destination_count = np.zeros(PRODUCTS, dtype=np.int32)
    source_bits = np.zeros(PRODUCTS, dtype=np.uint64)
    accumulators = empty_accumulators()
    state_count = np.zeros(36, dtype=np.int64)
    state_sum = np.zeros(36, dtype=np.float64)
    global_count = 0
    global_sum = 0.0
    train_residual = np.zeros(len(labels), dtype=np.float32)
    val_residual = np.zeros(len(val_labels), dtype=np.float32)
    event_position = 0
    pending = []
    matrices = {}
    rows = {}
    phase = time.time()

    def apply(indices: np.ndarray, source_metadata: dict[str, np.ndarray], source_state: np.ndarray, source_labels: np.ndarray, residual: np.ndarray, query_day: int) -> None:
        nonlocal global_count, global_sum
        if len(indices) == 0:
            return
        update_accumulators(
            np.asarray(source_state[indices]), residual[indices], source_metadata["customer"][indices], source_metadata["day"][indices], source_metadata["recency"][indices], source_metadata["tenure"][indices], source_metadata["activity"][indices], query_day,
            accumulators["counts"], accumulators["sums"], accumulators["decay180_sum"], accumulators["decay180_weight"], accumulators["decay730_sum"], accumulators["decay730_weight"], accumulators["last_day"], accumulators["customer_bits"], accumulators["forward"], accumulators["forward_weight"], accumulators["reverse"], accumulators["reverse_weight"], accumulators["category_forward"], accumulators["category_forward_weight"], accumulators["category_reverse"], accumulators["category_reverse_weight"],
        )
        update_state_prior(source_labels[indices], source_metadata["recency"][indices], source_metadata["tenure"][indices], source_metadata["activity"][indices], state_count, state_sum)
        global_count += len(indices)
        global_sum += float(source_labels[indices].sum())

    def query(split: str, metadata: dict[str, np.ndarray], state: np.ndarray, selected: np.ndarray, raw_total: int) -> np.ndarray:
        return query_features(
            np.asarray(state[selected]), metadata["customer"][selected], metadata["recency"][selected], metadata["tenure"][selected], metadata["activity"][selected], pair_count, source_count, destination_count, source_bits, raw_total,
            accumulators["counts"], accumulators["sums"], accumulators["decay180_sum"], accumulators["decay180_weight"], accumulators["decay730_sum"], accumulators["decay730_weight"], accumulators["customer_bits"], accumulators["forward"], accumulators["forward_weight"], accumulators["reverse"], accumulators["reverse_weight"], accumulators["category_forward"], accumulators["category_forward_weight"], accumulators["category_reverse"], accumulators["category_reverse_weight"],
        )

    train_rows = np.flatnonzero(train["activity"] == 1).astype(np.int64)
    train_matrix = np.lib.format.open_memmap(paths["train"], mode="w+", dtype=np.float32, shape=(len(train_rows), TRANSITION_FEATURES))
    activity_position = np.full(len(labels), -1, dtype=np.int64)
    activity_position[train_rows] = np.arange(len(train_rows), dtype=np.int64)
    for current in range(len(origins)):
        query_day = int(origins[current])
        remaining = []
        for available, indices in pending:
            if available <= query_day:
                apply(indices, train, train_state, labels, train_residual, query_day)
            else:
                remaining.append((available, indices))
        pending = remaining
        event_position, raw_total = update_raw_until(event_day, event_source, event_destination, event_position, query_day, pair_count, source_count, destination_count, source_bits)
        indices = np.flatnonzero(train["origin"] == current)
        selected = indices[train["activity"][indices] == 1]
        if len(selected):
            train_matrix[activity_position[selected]] = query("train", train, train_state, selected, raw_total)
        prior = prior_values(train["recency"][indices], train["tenure"][indices], train["activity"][indices], state_count, state_sum, global_count, global_sum)
        train_residual[indices] = labels[indices].astype(np.float32) - prior
        pending.append((query_day + 91, indices))
        announce("transition_origin", started, f"origin={current} rows={len(indices)} activity1={len(selected)} mature={global_count}")
    train_matrix.flush()
    del train_matrix, activity_position
    gc.collect()

    val_day = int(val["day"][0])
    for available, indices in pending:
        if available <= val_day:
            apply(indices, train, train_state, labels, train_residual, val_day)
    event_position, raw_total = update_raw_until(event_day, event_source, event_destination, event_position, val_day, pair_count, source_count, destination_count, source_bits)
    val_rows = np.flatnonzero(val["activity"] == 1).astype(np.int64)
    val_matrix = query("val", val, val_state, val_rows, raw_total)
    np.save(paths["val"], val_matrix)
    val_prior = prior_values(val["recency"], val["tenure"], val["activity"], state_count, state_sum, global_count, global_sum)
    val_residual[:] = val_labels.astype(np.float32) - val_prior
    announce("transition_model_a", started, f"val_activity1={len(val_rows)} mature={global_count}")

    test_day = int(test["day"][0])
    val_indices = np.arange(len(val_labels), dtype=np.int64)
    if val_day + 91 <= test_day:
        apply(val_indices, val, val_state, val_labels, val_residual, test_day)
    for available, indices in pending:
        if available > val_day and available <= test_day:
            apply(indices, train, train_state, labels, train_residual, test_day)
    event_position, raw_total = update_raw_until(event_day, event_source, event_destination, event_position, test_day, pair_count, source_count, destination_count, source_bits)
    test_rows = np.flatnonzero(test["activity"] == 1).astype(np.int64)
    test_matrix = query("test", test, test_state, test_rows, raw_total)
    np.save(paths["test"], test_matrix)
    for split, selected in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        np.save(index_paths[split], selected)
        register_artifact(f"{VERSION} {split} activity-one transition matrix", paths[split], "Causal mature-residual transition features for activity-one seeds.", f"{VERSION}:transition:{split}:v1")
    train_previous = np.asarray(train_state[train_rows, 1]) >= 0
    val_previous = np.asarray(val_state[val_rows, 1]) >= 0
    test_previous = np.asarray(test_state[test_rows, 1]) >= 0
    diagnostics = {
        "rows": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "activity_one_share": {"train": len(train_rows) / len(labels), "val": len(val_rows) / len(val_labels), "test": len(test_rows) / len(test["activity"])},
        "previous_distinct_coverage": {"train": float(train_previous.mean()), "val": float(val_previous.mean()), "test": float(test_previous.mean())},
        "feature_count": TRANSITION_FEATURES,
        "channels": CHANNELS,
        "hash_width": WIDTH,
        "countsketch_dimensions": DIMENSIONS,
        "half_lives": [180, 730],
        "minimum_distinct_customers": 8,
        "maturity_days": 91,
        "seconds": time.time() - phase,
        "rows_per_minute": 60.0 * (len(labels) + len(val_labels) + len(test["activity"])) / max(time.time() - phase, 1e-6),
    }
    write_json(diagnostic_path, diagnostics)
    announce("transition_features", started, f"rows={sum(diagnostics['rows'].values())} features={TRANSITION_FEATURES} rate={diagnostics['rows_per_minute']:.0f}/min")
    return {
        **{f"x_{split}": np.load(path, mmap_mode="r") for split, path in paths.items()},
        **{f"rows_{split}": np.load(path, mmap_mode="r") for split, path in index_paths.items()},
    }, diagnostics


def load_champion_module():
    path = shared_root() / "champion" / "code" / "renewal.py"
    specification = importlib.util.spec_from_file_location("renewal", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def ensure_compact_matrices(train: dict[str, np.ndarray], val: dict[str, np.ndarray], test: dict[str, np.ndarray], labels: np.ndarray, val_labels: np.ndarray, started: float) -> tuple[dict[str, np.ndarray], dict]:
    paths = {split: cache_root() / f"compact_{split}.npy" for split in ("train", "val", "test")}
    diagnostic_path = cache_root() / "compact_matrices.json"
    if all(path.exists() for path in paths.values()) and diagnostic_path.exists():
        return {split: np.load(path, mmap_mode="r") for split, path in paths.items()}, json.loads(diagnostic_path.read_text())
    renewal = load_champion_module()
    phase = time.time()
    train_frame = renewal.load_feature_frame(feature_files("train"))
    val_frame = renewal.load_feature_frame(feature_files("val"))
    test_frame = renewal.load_feature_frame(feature_files("test"))
    train_frame, val_frame, test_frame = renewal.add_causal_relational_targets(train_frame, labels, val_frame, val_labels, test_frame)
    train_frame = renewal.add_renewal_transforms(train_frame)
    val_frame = renewal.add_renewal_transforms(val_frame)
    test_frame = renewal.add_renewal_transforms(test_frame)
    train_frame, val_frame, test_frame = renewal.add_customer_name_features(train_frame, val_frame, test_frame)
    columns = renewal.feature_columns(train_frame)
    for split, frame in (("train", train_frame), ("val", val_frame), ("test", test_frame)):
        matrix = renewal.matrix(frame, columns)
        temporary = paths[split].with_name(f"compact_{split}.{os.getpid()}.tmp.npy")
        np.save(temporary, matrix)
        os.replace(temporary, paths[split])
        register_artifact(f"{VERSION} compact champion matrix {split}", paths[split], "Exact compact champion feature lineage in original row order.", f"{VERSION}:compact:{split}:v1")
        del matrix
    diagnostics = {"feature_count": len(columns), "columns": columns, "seconds": time.time() - phase, "rows": {"train": len(train_frame), "val": len(val_frame), "test": len(test_frame)}}
    write_json(diagnostic_path, diagnostics)
    announce("compact_matrices", started, f"features={len(columns)} seconds={diagnostics['seconds']:.1f}")
    del train_frame, val_frame, test_frame
    gc.collect()
    return {split: np.load(path, mmap_mode="r") for split, path in paths.items()}, diagnostics


def rank_values(values: np.ndarray) -> np.ndarray:
    return (rankdata(np.asarray(values), method="average") / len(values)).astype(np.float32)


def temporal_weights(origins: np.ndarray) -> np.ndarray:
    latest = float(np.max(origins))
    return np.clip(np.exp(-0.025 * (latest - origins)), 0.55, 1.0).astype(np.float32)


def train_base(x: np.ndarray, y: np.ndarray, origins: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=520, learning_rate=0.04, num_leaves=64, max_depth=-1, min_child_samples=500, subsample=0.85, subsample_freq=1, colsample_bytree=0.84, reg_alpha=0.2, reg_lambda=3.0, max_bin=127, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")), random_state=1337, verbosity=-1, deterministic=True, force_col_wise=True,
    )
    model.fit(x, y, sample_weight=temporal_weights(origins), callbacks=[lgb.log_evaluation(0)])
    return model


def train_expert(x: np.ndarray, y: np.ndarray, origins: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary", n_estimators=500, learning_rate=0.03, num_leaves=31, max_depth=6, min_child_samples=4000, colsample_bytree=0.8, reg_alpha=2.0, reg_lambda=32.0, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")), random_state=1337, verbosity=-1, deterministic=True, force_col_wise=True,
    )
    model.fit(x, y, sample_weight=temporal_weights(origins), callbacks=[lgb.log_evaluation(0)])
    return model


def route(base: np.ndarray, expert: np.ndarray, activity_mask: np.ndarray) -> np.ndarray:
    output = rank_values(base)
    base_activity = rank_values(base[activity_mask])
    expert_activity = rank_values(expert)
    output[activity_mask] += 0.25 * (expert_activity - base_activity)
    return output.astype(np.float32)


def prepared_auc(y: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    cumulative_negative = np.cumsum(negative) - negative
    return float(np.sum(positive * (cumulative_negative + 0.5 * negative)) / (positive.sum() * negative.sum()))


def clustered_bootstrap(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, customers: np.ndarray, draws: int = 1000) -> dict:
    codes, uniques = pd.factorize(customers, sort=False)
    baseline_prepared = prepared_auc(y, baseline)
    candidate_prepared = prepared_auc(y, candidate)
    rng = np.random.default_rng(1337)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        weights = rng.poisson(1.0, len(uniques)).astype(np.float32)[codes]
        differences[draw] = weighted_auc(y, weights, candidate_prepared) - weighted_auc(y, weights, baseline_prepared)
    return {
        "draws": draws,
        "observed_delta": float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
        "mean_delta": float(differences.mean()),
        "standard_error": float(differences.std(ddof=1)),
        "probability_improvement": float(np.mean(differences > 0.0)),
        "lower_10": float(np.quantile(differences, 0.1)),
        "upper_90": float(np.quantile(differences, 0.9)),
    }


def slice_rows(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, recency: np.ndarray, activity: np.ndarray) -> list[dict]:
    groups = {
        "activity_1": activity == 1,
        "activity_2_3": (activity >= 2) & (activity <= 3),
        "activity_4_plus": activity >= 4,
        "activity1_recency_0_14": (activity == 1) & (recency <= 14),
        "activity1_recency_15_30": (activity == 1) & (recency > 14) & (recency <= 30),
        "activity1_recency_31_60": (activity == 1) & (recency > 30) & (recency <= 60),
        "activity1_recency_61_91": (activity == 1) & (recency > 60),
    }
    output = []
    for name, mask in groups.items():
        if mask.sum() and np.unique(labels[mask]).size == 2:
            base_auc = float(roc_auc_score(labels[mask], baseline[mask]))
            candidate_auc = float(roc_auc_score(labels[mask], candidate[mask]))
            output.append({"stratum": name, "count": int(mask.sum()), "label_rate": float(labels[mask].mean()), "baseline_auc": base_auc, "candidate_auc": candidate_auc, "delta": candidate_auc - base_auc})
    return output


def ensure_gate_predictions(compact: dict[str, np.ndarray], transition: dict[str, np.ndarray], train: dict[str, np.ndarray], labels: np.ndarray, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    base_path = cache_root() / "gate_base_oof.npy"
    expert_path = cache_root() / "gate_expert_oof.npy"
    diagnostic_path = cache_root() / "gate.json"
    if base_path.exists() and expert_path.exists() and diagnostic_path.exists():
        return np.load(base_path, mmap_mode="r"), np.load(expert_path, mmap_mode="r"), json.loads(diagnostic_path.read_text())
    base_oof = np.full(len(labels), np.nan, dtype=np.float32)
    expert_oof = np.full(len(labels), np.nan, dtype=np.float32)
    activity_rows = np.asarray(transition["rows_train"])
    activity_origin = train["origin"][activity_rows]
    activity_labels = labels[activity_rows]
    activity_compact = np.asarray(compact["train"][activity_rows], dtype=np.float32)
    specialist_matrix = np.concatenate([activity_compact, np.asarray(transition["x_train"], dtype=np.float32)], axis=1)
    gate_rows = []
    dev_pass = True
    for fold in (*DEVELOPMENT, CONFIRMATION):
        base_fit = train["origin"] < fold
        base_hold = train["origin"] == fold
        base_model = train_base(np.asarray(compact["train"][base_fit], dtype=np.float32), labels[base_fit], train["origin"][base_fit])
        base_oof[base_hold] = base_model.predict_proba(np.asarray(compact["train"][base_hold], dtype=np.float32))[:, 1].astype(np.float32)
        del base_model
        expert_fit = activity_origin < fold
        expert_hold = activity_origin == fold
        expert_model = train_expert(specialist_matrix[expert_fit], activity_labels[expert_fit], activity_origin[expert_fit])
        expert_prediction = expert_model.predict_proba(specialist_matrix[expert_hold])[:, 1].astype(np.float32)
        expert_oof[activity_rows[expert_hold]] = expert_prediction
        del expert_model
        hold_indices = np.flatnonzero(base_hold)
        hold_activity = train["activity"][hold_indices] == 1
        candidate = route(base_oof[hold_indices], expert_prediction, hold_activity)
        baseline = rank_values(base_oof[hold_indices])
        overall_delta = float(roc_auc_score(labels[hold_indices], candidate) - roc_auc_score(labels[hold_indices], baseline))
        activity_delta = float(roc_auc_score(labels[hold_indices][hold_activity], candidate[hold_activity]) - roc_auc_score(labels[hold_indices][hold_activity], baseline[hold_activity]))
        row = {"fold": fold, "count": len(hold_indices), "activity1_count": int(hold_activity.sum()), "baseline_auc": float(roc_auc_score(labels[hold_indices], baseline)), "candidate_auc": float(roc_auc_score(labels[hold_indices], candidate)), "overall_delta": overall_delta, "activity1_delta": activity_delta}
        gate_rows.append(row)
        announce("specialist_fold", started, f"fold={fold} overall_delta={overall_delta:.6f} activity1_delta={activity_delta:.6f}")
        if fold in DEVELOPMENT and not (overall_delta > 0.0 and activity_delta > 0.0):
            dev_pass = False
        if fold == max(DEVELOPMENT) and not dev_pass:
            break
    confirmation_row = next((row for row in gate_rows if row["fold"] == CONFIRMATION), None)
    confirmation = None
    slices = []
    accepted = False
    if dev_pass and confirmation_row is not None:
        hold_indices = np.flatnonzero(train["origin"] == CONFIRMATION)
        hold_activity = train["activity"][hold_indices] == 1
        baseline = rank_values(base_oof[hold_indices])
        candidate = route(base_oof[hold_indices], expert_oof[hold_indices[hold_activity]], hold_activity)
        confirmation = clustered_bootstrap(labels[hold_indices], baseline, candidate, train["customer"][hold_indices], 1000)
        slices = slice_rows(labels[hold_indices], baseline, candidate, train["recency"][hold_indices], train["activity"][hold_indices])
        critical_reversal = any(row["stratum"].startswith("activity1_recency") and row["count"] >= 10000 and row["delta"] < 0.0 for row in slices)
        accepted = confirmation["observed_delta"] > 0.0 and confirmation["probability_improvement"] >= 0.8 and not critical_reversal and confirmation_row["activity1_delta"] > 0.0
    np.save(base_path, base_oof)
    np.save(expert_path, expert_oof)
    diagnostics = {
        "development_origins": list(DEVELOPMENT),
        "confirmation_origin": CONFIRMATION,
        "folds": gate_rows,
        "development_passed": dev_pass,
        "confirmation": confirmation,
        "slices": slices,
        "accepted": accepted,
        "router_coefficient": 0.25,
        "specialist_features": specialist_matrix.shape[1],
        "hyperparameters": {"trees": 500, "learning_rate": 0.03, "leaves": 31, "depth": 6, "minimum_child": 4000, "feature_fraction": 0.8, "l1": 2.0, "l2": 32.0},
    }
    write_json(diagnostic_path, diagnostics)
    register_artifact(f"{VERSION} forward gate", diagnostic_path, "Origins 25/27/28 development and untouched origin-29 specialist confirmation.", f"{VERSION}:gate:v1")
    announce("specialist_gate", started, f"development={dev_pass} accepted={accepted}")
    del activity_compact, specialist_matrix
    gc.collect()
    return np.load(base_path, mmap_mode="r"), np.load(expert_path, mmap_mode="r"), diagnostics


def ensure_final_predictions(compact: dict[str, np.ndarray], transition: dict[str, np.ndarray], train: dict[str, np.ndarray], val: dict[str, np.ndarray], labels: np.ndarray, val_labels: np.ndarray, gate: dict, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    path = cache_root() / "final_predictions.npz"
    diagnostic_path = cache_root() / "final.json"
    if path.exists() and diagnostic_path.exists():
        values = np.load(path, allow_pickle=False)
        return values["val"], values["test"], json.loads(diagnostic_path.read_text())
    baseline_val, baseline_test = floor_predictions()
    if not gate["accepted"]:
        val_prediction = baseline_val.copy()
        test_prediction = baseline_test.copy()
        diagnostics = {"used_floor": True, "validation_preserved_before_model_b": True}
    else:
        train_rows = np.asarray(transition["rows_train"])
        val_rows = np.asarray(transition["rows_val"])
        test_rows = np.asarray(transition["rows_test"])
        x_train = np.concatenate([np.asarray(compact["train"][train_rows], dtype=np.float32), np.asarray(transition["x_train"], dtype=np.float32)], axis=1)
        x_val = np.concatenate([np.asarray(compact["val"][val_rows], dtype=np.float32), np.asarray(transition["x_val"], dtype=np.float32)], axis=1)
        x_test = np.concatenate([np.asarray(compact["test"][test_rows], dtype=np.float32), np.asarray(transition["x_test"], dtype=np.float32)], axis=1)
        model_a = train_expert(x_train, labels[train_rows], train["origin"][train_rows])
        val_expert = model_a.predict_proba(x_val)[:, 1].astype(np.float32)
        val_prediction = route(baseline_val, val_expert, val["activity"] == 1).copy()
        del model_a
        x_b = np.concatenate([x_train, x_val], axis=0)
        y_b = np.concatenate([labels[train_rows], val_labels[val_rows]])
        origin_b = np.concatenate([train["origin"][train_rows], np.full(len(val_rows), 31, dtype=np.int16)])
        model_b = train_expert(x_b, y_b, origin_b)
        test_expert = model_b.predict_proba(x_test)[:, 1].astype(np.float32)
        test_mask = np.zeros(len(baseline_test), dtype=bool)
        test_mask[test_rows] = True
        test_prediction = route(baseline_test, test_expert, test_mask)
        diagnostics = {"used_floor": False, "model_a_rows": len(train_rows), "model_b_rows": len(y_b), "validation_preserved_before_model_b": True}
    temporary = path.with_name(f"final_predictions.{os.getpid()}.npz")
    np.savez(temporary, val=val_prediction.astype(np.float32), test=test_prediction.astype(np.float32))
    os.replace(temporary, path)
    write_json(diagnostic_path, diagnostics)
    register_artifact(f"{VERSION} final predictions", path, "Frozen routed specialist or exact full_candidate2 floor under the legal two-model lineage.", f"{VERSION}:final:v1")
    announce("final_predictions", started, f"used_floor={diagnostics['used_floor']}")
    return val_prediction, test_prediction, diagnostics


def debug_smoke(started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    origins = origin_days()
    train = load_seed_metadata("train", origins, debug=True)
    pointer, event_days, event_products = ensure_history_arrays()
    category, brand = ensure_product_metadata()
    count = min(20000, len(train["customer"]))
    state = extract_states(pointer, event_days, event_products, train["customer"][:count], train["day"][:count], category, brand)
    labels = split_labels("train")[train["row_id"][:count]]
    matrix = np.column_stack([state[:, 6:14], train["recency"][:count], train["tenure"][:count]]).astype(np.float32)
    fit = train["origin"][:count] < train["origin"][:count].max()
    hold = ~fit
    model = lgb.LGBMClassifier(objective="binary", n_estimators=20, learning_rate=0.05, num_leaves=15, min_child_samples=100, n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")), verbosity=-1, random_state=1337)
    model.fit(matrix[fit], labels[fit], callbacks=[lgb.log_evaluation(0)])
    prediction = model.predict_proba(matrix[hold])[:, 1]
    val, test = floor_predictions()
    diagnostics = {"debug": True, "state_rows": count, "previous_distinct_coverage": float(np.mean(state[:, 1] >= 0)), "hold_rows": int(hold.sum()), "hold_auc": float(roc_auc_score(labels[hold], prediction)), "output_source": "exact full_candidate2 floor"}
    announce("debug_complete", started, f"rows={count} hold={hold.sum()}")
    return val, test, diagnostics


def full_run(started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    cached = cache_root() / "final_predictions.npz"
    cached_diagnostics = cache_root() / "run_diagnostics.json"
    if cached.exists() and cached_diagnostics.exists():
        values = np.load(cached, allow_pickle=False)
        return values["val"], values["test"], json.loads(cached_diagnostics.read_text())
    origins = origin_days()
    train = load_seed_metadata("train", origins)
    val = load_seed_metadata("val", origins)
    test = load_seed_metadata("test", origins)
    labels = split_labels("train")
    val_labels = split_labels("val")
    announce("seed_metadata", started, f"train={len(labels)} val={len(val_labels)} test={len(test['customer'])}")
    train_state = ensure_states("train", train, started)
    val_state = ensure_states("val", val, started)
    test_state = ensure_states("test", test, started)
    transition, transition_diagnostics = build_transition_features(train, val, test, train_state, val_state, test_state, labels, val_labels, origins, started)
    compact, compact_diagnostics = ensure_compact_matrices(train, val, test, labels, val_labels, started)
    _, _, gate = ensure_gate_predictions(compact, transition, train, labels, started)
    val_prediction, test_prediction, final = ensure_final_predictions(compact, transition, train, val, labels, val_labels, gate, started)
    diagnostics = {
        "debug": False,
        "version": VERSION,
        "banked_floor_score": 0.7124372823047946,
        "transition": transition_diagnostics,
        "compact": compact_diagnostics,
        "gate": gate,
        "final": final,
        "model_a_validation_source": "official training labels only; every transition label is delayed by 91 days",
        "model_b_test_source": "official training plus fully mature validation labels",
        "validation_prediction_preserved_before_model_b": True,
        "router": "base_rank + 0.25 * I(n_91 == 1) * (expert_rank_within_activity1 - base_rank_within_activity1)",
    }
    write_json(cached_diagnostics, diagnostics)
    return val_prediction, test_prediction, diagnostics


def run_transition_specialist(debug: bool, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    if debug:
        return debug_smoke(started)
    return full_run(started)
