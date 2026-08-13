from __future__ import annotations

import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np


EVENT_VERSION = "lane2_intensityfree_v1"
FULL_ORIGINS = (
    "2013-10-03",
    "2014-01-02",
    "2014-04-03",
    "2014-07-03",
    "2014-10-02",
    "2015-01-01",
    "2015-04-02",
    "2015-07-02",
    "2015-10-01",
    "2016-01-01",
)


def shared_root() -> Path:
    return Path(os.environ["KAPSO_SHARED_CACHE_DIR"])


def cache_root() -> Path:
    path = shared_root() / EVENT_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def register_artifact(name: str, path: Path, description: str, key: str, hint: str) -> None:
    registry = shared_root() / "artifacts.json"
    lock_path = shared_root() / "artifacts.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            records = []
        record = {
            "name": name,
            "path": str(path.relative_to(shared_root())),
            "description": description,
            "content_key": key,
            "rebuild_hint": hint,
        }
        records = [item for item in records if item.get("content_key") != key]
        records.append(record)
        temporary = registry.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def text_paths() -> dict[str, Path]:
    root = shared_root() / "lane2_sequence_index_v3" / "modernbert_8949b9_random_projection_v3_tokens96"
    paths = {
        "review": root / "review_embeddings.npy",
        "product": root / "product_embeddings.npy",
        "event_doc": root / "event_doc_index.npy",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing cached ModernBERT marks: {missing}")
    return paths


def preserve_fallback() -> Path:
    destination = cache_root() / "archived_campaign_blend.npz"
    if destination.exists():
        return destination
    source = shared_root() / "lane0_gbdt_widened_v1" / "gbdt_final_predictions.npz"
    if not source.exists():
        raise RuntimeError("Archived campaign blend predictions are unavailable")
    with np.load(source, allow_pickle=False) as values:
        validation = np.asarray(values["validation"], dtype=np.float64)
        test = np.asarray(values["test"], dtype=np.float64)
    temporary = destination.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, validation=validation, test=test)
    os.replace(temporary, destination)
    register_artifact(
        "lane2 archived campaign blend fallback",
        destination,
        "Immutable copies of the internally selected widened GBDT campaign finalist for fallback and decorrelated rank averaging",
        f"{EVENT_VERSION}:campaign_blend_fallback",
        "Copy validation and test from lane0_gbdt_widened_v1/gbdt_final_predictions.npz",
    )
    return destination


def prepare_coalesced_events(index: dict[str, np.ndarray]) -> Path:
    root = cache_root() / "coalesced_events_v1"
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "metadata.json"
    names = (
        "customer",
        "day",
        "product",
        "rating_mean",
        "rating_std",
        "verified_share",
        "text_missing_share",
        "product_missing_share",
        "multiplicity",
        "distinct_products",
        "gap",
        "category",
        "brand_frequency_bucket",
        "offsets",
        "review_mark",
        "product_mark",
    )
    if metadata.exists() and all((root / f"{name}.npy").exists() for name in names):
        values = json.loads(metadata.read_text())
        print(f"[coalesce] reused events={values['events']:,} rows={values['review_rows']:,}", flush=True)
        return root
    started = time.time()
    customer_rows = np.asarray(index["customer"])
    day_rows = np.asarray(index["day"])
    product_rows = np.asarray(index["product"])
    first = np.empty(len(customer_rows), dtype=np.bool_)
    first[0] = True
    first[1:] = (customer_rows[1:] != customer_rows[:-1]) | (day_rows[1:] != day_rows[:-1])
    starts = np.flatnonzero(first).astype(np.int64)
    event_count = len(starts)
    ends = np.empty(event_count, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = len(customer_rows)
    multiplicity = ends - starts
    event_customer = customer_rows[starts].astype(np.int32)
    event_day = day_rows[starts].astype(np.int32)
    representative_product = product_rows[starts].astype(np.int32)
    counts = np.bincount(event_customer, minlength=len(index["offsets"]) - 1)
    offsets = np.empty(len(counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    gap = np.zeros(event_count, dtype=np.int16)
    same_customer = event_customer[1:] == event_customer[:-1]
    gap[1:][same_customer] = np.minimum(
        event_day[1:][same_customer] - event_day[:-1][same_customer],
        np.iinfo(np.int16).max,
    ).astype(np.int16)
    product_brand = np.asarray(index["product_brand"], dtype=np.int64)
    brand_counts = np.bincount(product_brand, minlength=65536)
    brand_bucket = np.minimum(
        np.floor(np.log2(1.0 + brand_counts[product_brand[representative_product]])).astype(np.int16),
        31,
    ).astype(np.uint8)
    scalar_values = {
        "customer": event_customer,
        "day": event_day,
        "product": representative_product,
        "multiplicity": np.minimum(multiplicity, np.iinfo(np.int16).max).astype(np.int16),
        "gap": gap,
        "category": np.asarray(index["product_category"])[representative_product].astype(np.int16),
        "brand_frequency_bucket": brand_bucket,
        "offsets": offsets,
    }
    for name, value in scalar_values.items():
        atomic_save(root / f"{name}.npy", value)
    review_path = root / f"review_mark.{os.getpid()}.tmp.npy"
    product_path = root / f"product_mark.{os.getpid()}.tmp.npy"
    scalar_paths = {
        name: root / f"{name}.{os.getpid()}.tmp.npy"
        for name in (
            "rating_mean",
            "rating_std",
            "verified_share",
            "text_missing_share",
            "product_missing_share",
            "distinct_products",
        )
    }
    review_output = np.lib.format.open_memmap(review_path, mode="w+", dtype=np.float16, shape=(event_count, 64))
    product_output = np.lib.format.open_memmap(product_path, mode="w+", dtype=np.float16, shape=(event_count, 32))
    scalar_output = {
        name: np.lib.format.open_memmap(path, mode="w+", dtype=np.float32 if name != "distinct_products" else np.int16, shape=(event_count,))
        for name, path in scalar_paths.items()
    }
    paths = text_paths()
    review_embeddings = np.load(paths["review"], mmap_mode="r")
    product_embeddings = np.load(paths["product"], mmap_mode="r")
    event_doc = np.load(paths["event_doc"], mmap_mode="r")
    ratings = np.asarray(index["rating"])
    verified = np.asarray(index["verified"])
    missing = np.asarray(index["text_missing"])
    product_missing = np.asarray(index["product_text_missing"])
    chunk = 200_000
    for event_start in range(0, event_count, chunk):
        event_stop = min(event_start + chunk, event_count)
        row_start = int(starts[event_start])
        row_stop = int(starts[event_stop]) if event_stop < event_count else len(customer_rows)
        local = starts[event_start:event_stop] - row_start
        local_count = multiplicity[event_start:event_stop].astype(np.float32)
        row_slice = slice(row_start, row_stop)
        review_values = review_embeddings[event_doc[row_slice]].astype(np.float32)
        product_values = product_embeddings[product_rows[row_slice]].astype(np.float32)
        review_output[event_start:event_stop] = (
            np.add.reduceat(review_values, local, axis=0) / local_count[:, None]
        ).astype(np.float16)
        product_output[event_start:event_stop] = (
            np.add.reduceat(product_values, local, axis=0) / local_count[:, None]
        ).astype(np.float16)
        rating_sum = np.add.reduceat(ratings[row_slice].astype(np.float32), local)
        rating_square = np.add.reduceat(np.square(ratings[row_slice].astype(np.float32)), local)
        rating_mean = rating_sum / local_count
        scalar_output["rating_mean"][event_start:event_stop] = rating_mean
        scalar_output["rating_std"][event_start:event_stop] = np.sqrt(
            np.maximum(rating_square / local_count - np.square(rating_mean), 0.0)
        )
        scalar_output["verified_share"][event_start:event_stop] = (
            np.add.reduceat(verified[row_slice].astype(np.float32), local) / local_count
        )
        scalar_output["text_missing_share"][event_start:event_stop] = (
            np.add.reduceat(missing[row_slice].astype(np.float32), local) / local_count
        )
        scalar_output["product_missing_share"][event_start:event_stop] = (
            np.add.reduceat(product_missing[product_rows[row_slice]].astype(np.float32), local) / local_count
        )
        product_local = product_rows[row_slice]
        distinct_flag = np.empty(len(product_local), dtype=np.int16)
        distinct_flag[0] = 1
        distinct_flag[1:] = (product_local[1:] != product_local[:-1]).astype(np.int16)
        distinct_flag[local] = 1
        scalar_output["distinct_products"][event_start:event_stop] = np.add.reduceat(distinct_flag, local)
        if event_stop == event_count or event_stop // 1_000_000 != event_start // 1_000_000:
            review_output.flush()
            product_output.flush()
            for output in scalar_output.values():
                output.flush()
            print(f"[coalesce] events={event_stop:,}/{event_count:,} elapsed={time.time() - started:.1f}s", flush=True)
    del review_output, product_output
    for output in scalar_output.values():
        output.flush()
    del scalar_output
    os.replace(review_path, root / "review_mark.npy")
    os.replace(product_path, root / "product_mark.npy")
    for name, path in scalar_paths.items():
        os.replace(path, root / f"{name}.npy")
    elapsed = time.time() - started
    atomic_json(
        metadata,
        {
            "version": EVENT_VERSION,
            "review_rows": int(len(customer_rows)),
            "events": int(event_count),
            "same_time_followups_removed": int(len(customer_rows) - event_count),
            "customers": int((counts > 0).sum()),
            "seconds": elapsed,
        },
    )
    register_artifact(
        "lane2 coalesced marked review events",
        root,
        "Customer-time events with pooled review and product ModernBERT marks plus multiplicity, product diversity, rating, verification, and static buckets",
        f"{EVENT_VERSION}:coalesced_events_v1",
        "Run main.py using lane2_sequence_index_v3 and its completed ModernBERT caches",
    )
    print(f"[coalesce] completed events={event_count:,} rate={event_count / elapsed:.1f}/s", flush=True)
    return root


def load_events(index: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    root = prepare_coalesced_events(index)
    names = (
        "customer",
        "day",
        "product",
        "rating_mean",
        "rating_std",
        "verified_share",
        "text_missing_share",
        "product_missing_share",
        "multiplicity",
        "distinct_products",
        "gap",
        "category",
        "brand_frequency_bucket",
        "offsets",
        "review_mark",
        "product_mark",
    )
    return {name: np.load(root / f"{name}.npy", mmap_mode="r") for name in names}


def feature_root(debug: bool = False) -> Path:
    name = "lane3_latent_community_v3_debug" if debug else "lane3_latent_community_v3_full"
    path = shared_root() / name
    if not path.exists():
        raise RuntimeError(f"Missing cutoff-aligned feature cache {path}")
    return path


def prepare_customer_features() -> Path:
    root = cache_root() / "customer_static_v1"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "features.npy"
    metadata = root / "metadata.json"
    if path.exists() and metadata.exists():
        return path
    source = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"] / "db" / "customer.parquet"
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    query = f"""
        SELECT
            customer_id::INTEGER AS customer,
            ln(1 + count(*) OVER (PARTITION BY customer_name))::FLOAT AS name_frequency,
            length(coalesce(customer_name, ''))::FLOAT AS name_length,
            len(regexp_split_to_array(trim(coalesce(customer_name, '')), '\\s+'))::FLOAT AS token_count,
            regexp_matches(coalesce(customer_name, ''), '[0-9]')::UTINYINT AS has_digit,
            regexp_matches(coalesce(customer_name, ''), '[^A-Za-z0-9 ]')::UTINYINT AS has_punctuation,
            (coalesce(customer_name, '') = lower(coalesce(customer_name, '')))::UTINYINT AS all_lower,
            (coalesce(customer_name, '') = upper(coalesce(customer_name, '')))::UTINYINT AS all_upper,
            (coalesce(customer_name, '') = 'Amazon Customer')::UTINYINT AS amazon_customer,
            (coalesce(customer_name, '') = 'Kindle Customer')::UTINYINT AS kindle_customer,
            contains(lower(coalesce(customer_name, '')), 'reader')::UTINYINT AS reader_name,
            contains(lower(coalesce(customer_name, '')), 'book')::UTINYINT AS book_name,
            (customer_name IS NULL OR trim(coalesce(customer_name, '')) = '')::UTINYINT AS missing,
            unicode(substr(upper(trim(coalesce(customer_name, ''))), 1, 1))::INTEGER AS initial_code
        FROM read_parquet('{source}')
        ORDER BY customer_id
    """
    started = time.time()
    values = connection.execute(query).fetchnumpy()
    count = len(values["customer"])
    output = np.zeros((count, 39), dtype=np.float16)
    scalar_names = (
        "name_frequency",
        "name_length",
        "token_count",
        "has_digit",
        "has_punctuation",
        "all_lower",
        "all_upper",
        "amazon_customer",
        "kindle_customer",
        "reader_name",
        "book_name",
        "missing",
    )
    for column, name in enumerate(scalar_names):
        output[:, column] = np.asarray(values[name], dtype=np.float16)
    initial = np.asarray(values["initial_code"], dtype=np.int32)
    initial = np.where((initial >= 65) & (initial <= 90), initial - 65, 26)
    output[np.arange(count), 12 + initial] = 1.0
    atomic_save(path, output)
    elapsed = time.time() - started
    atomic_json(metadata, {"customers": count, "dimensions": 39, "seconds": elapsed})
    register_artifact(
        "lane2 customer-name static features",
        root,
        "History-free customer-name frequency, structure, default-profile flags, and first-initial buckets",
        f"{EVENT_VERSION}:customer_static_v1",
        "Run main.py against customer.parquet",
    )
    print(f"[customer_static] customers={count:,} dimensions=39 elapsed={elapsed:.1f}s", flush=True)
    return path


def load_snapshot(origin: str, debug: bool = False) -> dict[str, np.ndarray]:
    path = feature_root(debug) / f"features_{origin}.npz"
    if not path.exists():
        raise RuntimeError(f"Missing feature snapshot {path}")
    with np.load(path, allow_pickle=False) as values:
        result = {name: np.asarray(values[name]) for name in values.files}
    customer_features = np.load(prepare_customer_features(), mmap_mode="r")
    result["customer_static"] = np.asarray(customer_features[result["customer_id"].astype(np.int64)])
    cutoff_age = float((np.datetime64(origin, "D") - np.datetime64("2008-01-01", "D")).astype(np.int64))
    ego = np.asarray(result["ego"], dtype=np.float32)
    last_day = cutoff_age - ego[:, 0]
    first_day = last_day - ego[:, 9] * np.maximum(ego[:, 5] - 1.0, 0.0)
    result["calendar_context"] = np.column_stack((last_day / 2922.0, first_day / 2922.0)).astype(np.float32)
    return result


def archive_candidate(name: str, validation: np.ndarray, test: np.ndarray, metadata: dict) -> Path:
    root = cache_root() / "candidates"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.npz"
    temporary = path.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, validation=validation, test=test)
    os.replace(temporary, path)
    atomic_json(root / f"{name}.json", metadata)
    return path
