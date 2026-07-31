from __future__ import annotations

# Imports

import fcntl
import hashlib
import json
import math
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# Configuration

WINDOWS = [7, 28, 91, 182, 365]
HALF_LIVES = [14, 45, 120, 365]
HASH_DIM = 16
EMBEDDING_DIM = 64
SHRINKAGE_STRENGTH = 20.0
SEED = 3407
CACHE_VERSION = "lane0_count_l1_all_table_v2"
CORE_CACHE_VERSION = "lane0_count_l1_all_table_core_v3_exact_windows"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


# Logging

def phase_log(name: str, started: float, detail: str = "") -> None:
    elapsed = time.time() - started
    suffix = f" {detail}" if detail else ""
    print(f"[phase] {name} elapsed={elapsed:.2f}s{suffix}", flush=True)


def append_locked(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not path.exists():
            path.write_text("")
        with path.open("a") as handle:
            handle.write(text)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def initialize_living_documents(cache_dir: Path, mode: str) -> None:
    table_path = cache_dir / "table_information.md"
    feature_path = cache_dir / "features_history.md"
    if not table_path.exists():
        table_path.write_text("# Table information\n\n## Notes\n")
    if not feature_path.exists():
        feature_path.write_text("# Feature history\n\n## Entries\n")
    groups = [
        ("count-L1 target", "price-weighted future review-count regression with zero-price handling"),
        ("causal review history", "7/28/91/182/365-day and lifetime product histories, non-overlapping quarters, recency, gaps, decay, ratings, verification, and text completeness"),
        ("level-invariant dynamics", "fast/slow ratios, preceding-window ratios, category/brand shares and percentiles, and observation-count blending"),
        ("hierarchy priors", "completed-window product, brand, category, price-band, and global priors with strength-20 shrinkage"),
        ("completed-history cohort widening", "brand, category, and price-band medians plus shrinkage for robust history, same-season, and fast/slow estimates with log-count encodings"),
        ("customer audience", "two-hop reviewer activity, diversity, rating, verification, heavy/new-reviewer, and customer-name aggregates"),
        ("historical token hash", "fixed signed 16-dimensional unfitted review-text and summary token sketch"),
        ("MiniLM product text", "pinned all-MiniLM-L6-v2 product embeddings reduced to 64 PCA dimensions"),
    ]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    parts = []
    for name, what in groups:
        parts.append(
            f"\n### {name}\n- run/experiment: lane0-{stamp}-{mode} | status: PROPOSED\n"
            f"- what: {what}\n- outcome: pending internal expanding-origin folds.\n"
            f"- takeaway: preserve causal row-time censoring and immutable row alignment.\n"
        )
    append_locked(feature_path, "".join(parts))
    table_path.read_text()
    feature_path.read_text()


def record_living_outcome(cache_dir: Path, title: str, outcome: str, takeaway: str) -> None:
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    text = (
        f"\n### {title}\n- run/experiment: lane0-{stamp} | status: TESTED-KEPT\n"
        f"- what: frozen causal count-L1 all-table design.\n- outcome: {outcome}\n"
        f"- takeaway: {takeaway}\n"
    )
    append_locked(cache_dir / "features_history.md", text)


def register_artifact(cache_dir: Path, name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = []
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        relative = str(path.relative_to(cache_dir))
        record = {
            "name": name,
            "path": relative,
            "description": description,
            "content_key": content_key,
            "rebuild_hint": rebuild_hint,
        }
        records = [item for item in records if item.get("name") != name]
        records.append(record)
        temporary = registry.with_name(registry.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(records, indent=2))
        os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# Data loading

def data_paths() -> dict[str, Path]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    task_root = root / "tasks" / os.environ["RELBENCH_TASK"]
    return {
        "root": root,
        "product": root / "db" / "product.parquet",
        "review": root / "db" / "review.parquet",
        "customer": root / "db" / "customer.parquet",
        "train": task_root / "train.parquet",
        "val": task_root / "val.parquet",
        "test": task_root / "test.parquet",
    }


def load_task_frames(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(paths["train"], columns=["timestamp", "product_id", "ltv"])
    val = pd.read_parquet(paths["val"], columns=["timestamp", "product_id", "ltv"])
    test = pd.read_parquet(paths["test"], columns=["timestamp", "product_id"])
    frames = []
    for split, frame in (("train", train), ("val", val), ("test", test)):
        frame = frame.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["product_id"] = frame["product_id"].astype(np.int64)
        frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
        frame.insert(0, "split", split)
        frames.append(frame)
    return frames[0], frames[1], frames[2]


def load_static_products(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    query = f"""
        SELECT
            product_id,
            price,
            coalesce(array_to_string(category, ' > '), '') AS category_text,
            coalesce(brand, '') AS brand_text,
            length(coalesce(title, '')) AS title_length,
            length(coalesce(description, '')) AS description_length,
            (title IS NULL OR length(trim(title)) = 0)::INTEGER AS title_missing,
            (description IS NULL OR length(trim(description)) = 0)::INTEGER AS description_missing,
            (category IS NULL)::INTEGER AS category_missing,
            (brand IS NULL OR length(trim(brand)) = 0)::INTEGER AS brand_missing,
            concat_ws(' | ', coalesce(brand, ''), coalesce(title, ''),
                      coalesce(array_to_string(category, ' > '), ''),
                      coalesce(description, '')) AS product_text
        FROM read_parquet('{paths["product"]}')
        ORDER BY product_id
    """
    products = con.execute(query).fetchdf()
    con.close()
    products["product_id"] = products["product_id"].astype(np.int64)
    products["category_code"] = pd.factorize(products["category_text"], sort=True)[0].astype(np.int32)
    products["brand_code"] = pd.factorize(products["brand_text"], sort=True)[0].astype(np.int32)
    products["category_frequency"] = products.groupby("category_code")["product_id"].transform("size").astype(np.float32)
    products["brand_frequency"] = products.groupby("brand_code")["product_id"].transform("size").astype(np.float32)
    products["log_price"] = np.log1p(np.maximum(products["price"].fillna(0).to_numpy(), 0)).astype(np.float32)
    products["price_band"] = np.floor(products["log_price"].to_numpy() * 4).clip(0, 40).astype(np.int32)
    static_columns = [
        "product_id", "price", "log_price", "category_code", "brand_code",
        "category_frequency", "brand_frequency", "price_band", "title_length",
        "description_length", "title_missing", "description_missing",
        "category_missing", "brand_missing",
    ]
    return products[static_columns].copy(), products[["product_id", "product_text"]].copy()


# Measurement

def verify_target_support(train: pd.DataFrame, static: pd.DataFrame) -> dict[str, float | int]:
    joined = train[["product_id", "ltv"]].merge(static[["product_id", "price"]], on="product_id", how="left", validate="many_to_one")
    price = joined["price"].to_numpy(dtype=np.float64)
    target = joined["ltv"].to_numpy(dtype=np.float64)
    positive = np.isfinite(price) & (price > 0)
    inferred = target[positive] / price[positive]
    tolerance = 1e-5 * np.maximum(1.0, np.abs(inferred))
    noninteger = int(np.sum(np.abs(inferred - np.rint(inferred)) > tolerance))
    summary = {
        "rows": int(len(joined)),
        "positive_price": int(positive.sum()),
        "zero_price": int(np.sum(price == 0)),
        "missing_price": int(np.sum(~np.isfinite(price))),
        "negative_price": int(np.sum(np.isfinite(price) & (price < 0))),
        "noninteger_count": noninteger,
        "min_count": float(np.min(inferred)),
        "max_count": float(np.max(inferred)),
    }
    if noninteger or summary["min_count"] < 1 - 1e-5:
        raise RuntimeError(f"target support verification failed: {summary}")
    return summary


def reliability_diagnostics(paths: dict[str, Path]) -> dict[str, float | str | bool]:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    origin = con.execute(
        f"SELECT timestamp, count(*) n, avg(ltv) mean, var_samp(ltv) var FROM read_parquet('{paths['train']}') GROUP BY timestamp ORDER BY timestamp"
    ).fetchdf()
    sampling_variance = np.mean(origin["var"].to_numpy() / origin["n"].to_numpy())
    label_ratio = float(np.var(origin["mean"].to_numpy(), ddof=1) / sampling_variance)
    mean_changes = np.diff(origin["mean"].to_numpy())
    break_index = int(np.argmax(np.abs(mean_changes)))
    volume_query = f"""
        WITH origins AS (
            SELECT DISTINCT timestamp AS origin FROM read_parquet('{paths['train']}')
            UNION SELECT DISTINCT timestamp FROM read_parquet('{paths['val']}')
            UNION SELECT DISTINCT timestamp FROM read_parquet('{paths['test']}')
        )
        SELECT origin, count(*) AS volume
        FROM origins
        JOIN read_parquet('{paths['review']}') r
          ON r.review_time > origin - INTERVAL '91 days' AND r.review_time <= origin
        GROUP BY origin ORDER BY origin
    """
    volumes = con.execute(volume_query).fetchdf()
    con.close()
    changes = np.diff(volumes["volume"].to_numpy(dtype=np.float64))
    historical = changes[:-1]
    median_change = float(np.median(historical))
    robust_scale = float(1.4826 * np.median(np.abs(historical - median_change)))
    boundary_change = float(changes[-1])
    gap_units = float(abs(boundary_change - median_change) / max(robust_scale, 1.0))
    return {
        "label_variance_ratio": label_ratio,
        "break_from": str(origin.iloc[break_index]["timestamp"].date()),
        "break_to": str(origin.iloc[break_index + 1]["timestamp"].date()),
        "break_delta": float(mean_changes[break_index]),
        "validation_window_volume": int(volumes.iloc[-2]["volume"]),
        "test_window_volume": int(volumes.iloc[-1]["volume"]),
        "boundary_change": boundary_change,
        "boundary_gap_mad_units": gap_units,
        "both_extreme": bool(label_ratio > 10 and gap_units > 5),
    }


# Causal warehouses

def configure_duckdb(con: duckdb.DuckDBPyConnection, temporary_dir: Path) -> None:
    temporary_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '11'))}")
    con.execute("SET memory_limit='80GB'")
    con.execute(f"SET temp_directory='{temporary_dir}'")
    con.execute("SET preserve_insertion_order=false")


def seed_keys(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    keys = pd.concat(
        [train[["timestamp", "product_id"]], val[["timestamp", "product_id"]], test[["timestamp", "product_id"]]],
        ignore_index=True,
    ).drop_duplicates()
    return keys.rename(columns={"timestamp": "origin"})


def build_core_warehouse(paths: dict[str, Path], keys: pd.DataFrame, cache_dir: Path, work_dir: Path) -> Path:
    output = cache_dir / f"{CORE_CACHE_VERSION}.parquet"
    if output.exists():
        return output
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    con = duckdb.connect()
    configure_duckdb(con, work_dir / "duckdb_core")
    con.register("seed_keys", keys)
    count_columns = ",\n".join(
        [f"count(*) FILTER (WHERE review_time > origin - INTERVAL '{window} days') AS c{window}" for window in WINDOWS]
    )
    customer_columns = ",\n".join(
        [f"count(DISTINCT customer_id) FILTER (WHERE review_time > origin - INTERVAL '{window} days') AS u{window}" for window in WINDOWS]
    )
    decay_columns = ",\n".join(
        [f"sum(exp(-ln(2) * age_days / {half_life}.0)) AS decay{half_life}" for half_life in HALF_LIVES]
    )
    query = f"""
        COPY (
            WITH joined AS MATERIALIZED (
                SELECT
                    s.origin,
                    s.product_id,
                    r.review_time,
                    r.customer_id,
                    r.rating,
                    r.verified,
                    length(r.review_text) AS review_text_length,
                    length(r.summary) AS summary_length,
                    (r.review_text IS NULL OR length(trim(r.review_text)) = 0)::INTEGER AS review_text_missing,
                    (r.summary IS NULL OR length(trim(r.summary)) = 0)::INTEGER AS summary_missing,
                    date_diff('day', r.review_time, s.origin) AS age_days
                FROM seed_keys s
                JOIN read_parquet('{paths['review']}') r
                  ON s.product_id = r.product_id AND r.review_time <= s.origin
            ), sequenced AS (
                SELECT *, date_diff(
                    'day',
                    lag(review_time) OVER (PARTITION BY origin, product_id ORDER BY review_time),
                    review_time
                ) AS gap_days
                FROM joined
            )
            SELECT
                origin,
                product_id,
                count(*) AS c_lifetime,
                {count_columns},
                {customer_columns},
                count(*) FILTER (WHERE review_time > origin - INTERVAL '182 days' AND review_time <= origin - INTERVAL '91 days') AS c182_91,
                count(*) FILTER (WHERE review_time > origin - INTERVAL '273 days' AND review_time <= origin - INTERVAL '182 days') AS c273_182,
                count(*) FILTER (WHERE review_time > origin - INTERVAL '365 days' AND review_time <= origin - INTERVAL '274 days') AS c365_274,
                min(age_days) AS days_since_last_review,
                max(age_days) AS days_since_first_review,
                count(DISTINCT date_trunc('week', review_time)) FILTER (WHERE review_time > origin - INTERVAL '365 days') / 52.0 AS active_week_fraction,
                avg(gap_days) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS interarrival_mean_365,
                median(gap_days) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS interarrival_median_365,
                {decay_columns},
                avg(rating) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS rating_mean_91,
                stddev_pop(rating) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS rating_std_91,
                avg(rating) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS rating_mean_365,
                stddev_pop(rating) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS rating_std_365,
                avg(verified::INTEGER) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS verified_share_91,
                avg(verified::INTEGER) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS verified_share_365,
                avg(review_text_missing) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS review_text_missing_91,
                avg(review_text_missing) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS review_text_missing_365,
                avg(summary_missing) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS summary_missing_91,
                avg(summary_missing) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS summary_missing_365,
                avg(review_text_length) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS review_text_length_91,
                avg(review_text_length) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS review_text_length_365,
                avg(summary_length) FILTER (WHERE review_time > origin - INTERVAL '91 days') AS summary_length_91,
                avg(summary_length) FILTER (WHERE review_time > origin - INTERVAL '365 days') AS summary_length_365
            FROM sequenced
            GROUP BY origin, product_id
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()
    os.replace(temporary, output)
    register_artifact(
        cache_dir,
        CORE_CACHE_VERSION,
        output,
        "Causal all-origin product review histories through each seed timestamp",
        f"{CORE_CACHE_VERSION}:{paths['review'].stat().st_size}",
        "Rebuild with feature_pipeline.build_core_warehouse",
    )
    return output


def build_audience_warehouse(paths: dict[str, Path], keys: pd.DataFrame, cache_dir: Path, work_dir: Path) -> Path:
    output = cache_dir / f"{CACHE_VERSION}_audience.parquet"
    if output.exists():
        return output
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    con = duckdb.connect()
    configure_duckdb(con, work_dir / "duckdb_audience")
    con.register("seed_keys", keys)
    query = f"""
        COPY (
            WITH origins AS (SELECT DISTINCT origin FROM seed_keys),
            origin_review AS MATERIALIZED (
                SELECT o.origin, r.product_id, r.customer_id, r.rating, r.verified
                FROM origins o
                JOIN read_parquet('{paths['review']}') r
                  ON r.review_time > o.origin - INTERVAL '365 days' AND r.review_time <= o.origin
            ), customer_features AS (
                SELECT
                    r.origin,
                    r.customer_id,
                    count(*) AS customer_activity,
                    count(DISTINCT r.product_id) AS customer_diversity,
                    avg(r.rating) AS customer_rating_mean,
                    avg(r.verified::INTEGER) AS customer_verified_share,
                    length(coalesce(c.customer_name, '')) AS customer_name_length,
                    (c.customer_name IS NULL OR length(trim(c.customer_name)) = 0)::INTEGER AS customer_name_missing
                FROM origin_review r
                LEFT JOIN read_parquet('{paths['customer']}') c USING (customer_id)
                GROUP BY ALL
            ), audience_links AS (
                SELECT DISTINCT r.origin, r.product_id, r.customer_id
                FROM origin_review r
                JOIN seed_keys s ON r.origin = s.origin AND r.product_id = s.product_id
            )
            SELECT
                a.origin,
                a.product_id,
                count(*) AS audience_customer_count,
                median(c.customer_activity) AS audience_activity_median,
                quantile_cont(c.customer_activity, 0.9) AS audience_activity_q90,
                median(c.customer_diversity) AS audience_diversity_median,
                quantile_cont(c.customer_diversity, 0.9) AS audience_diversity_q90,
                median(c.customer_rating_mean) AS audience_rating_median,
                quantile_cont(c.customer_rating_mean, 0.9) AS audience_rating_q90,
                median(c.customer_verified_share) AS audience_verified_median,
                avg((c.customer_activity >= 10)::INTEGER) AS audience_heavy_reviewer_share,
                avg((c.customer_activity = 1)::INTEGER) AS audience_new_reviewer_share,
                median(c.customer_name_length) AS audience_name_length_median,
                avg(c.customer_name_missing) AS audience_name_missing_share
            FROM audience_links a
            JOIN customer_features c USING (origin, customer_id)
            GROUP BY a.origin, a.product_id
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()
    os.replace(temporary, output)
    register_artifact(
        cache_dir,
        f"{CACHE_VERSION}-audience",
        output,
        "Two-hop product reviewer audience features using the customer table",
        f"{CACHE_VERSION}:{paths['review'].stat().st_size}:audience",
        "Rebuild with feature_pipeline.build_audience_warehouse",
    )
    return output


# Historical review text

def build_review_hash_cache(paths: dict[str, Path], cache_dir: Path) -> Path:
    output = cache_dir / f"{CACHE_VERSION}_review_hash16.parquet"
    if output.exists():
        return output
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(
        n_features=HASH_DIM,
        alternate_sign=True,
        analyzer="word",
        ngram_range=(1, 1),
        lowercase=True,
        norm=None,
        dtype=np.float32,
    )
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    parquet = pq.ParquetFile(paths["review"])
    writer = None
    try:
        for batch in parquet.iter_batches(
            batch_size=40000,
            columns=["review_time", "product_id", "review_text", "summary"],
            use_threads=True,
        ):
            frame = batch.to_pandas()
            documents = (
                frame["summary"].fillna("").astype(str)
                + " | "
                + frame["review_text"].fillna("").astype(str)
            ).tolist()
            matrix = vectorizer.transform(documents).toarray().astype(np.float32, copy=False)
            scale = np.maximum(np.abs(matrix).sum(axis=1, keepdims=True), 1.0)
            matrix /= scale
            arrays = {
                "review_time": pa.array(frame["review_time"], type=pa.timestamp("ns")),
                "product_id": pa.array(frame["product_id"].to_numpy(dtype=np.int64)),
            }
            for index in range(HASH_DIM):
                arrays[f"token_hash_{index:02d}"] = pa.array(matrix[:, index])
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, output)
    register_artifact(
        cache_dir,
        f"{CACHE_VERSION}-review-hash16",
        output,
        "Fixed unfitted signed token-hash vectors for historical review text and summaries",
        f"{CACHE_VERSION}:{paths['review'].stat().st_size}:hashingvectorizer16",
        "Stream review parquet through HashingVectorizer with alternate_sign enabled",
    )
    return output


def build_historical_hash_features(hash_path: Path, keys: pd.DataFrame, cache_dir: Path, work_dir: Path) -> Path:
    output = cache_dir / f"{CACHE_VERSION}_historical_hash_features.parquet"
    if output.exists():
        return output
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    con = duckdb.connect()
    configure_duckdb(con, work_dir / "duckdb_hash")
    con.register("seed_keys", keys)
    columns = []
    for index in range(HASH_DIM):
        name = f"token_hash_{index:02d}"
        columns.append(f"avg(h.{name}) FILTER (WHERE h.review_time > s.origin - INTERVAL '91 days') AS review_hash_91_{index:02d}")
        columns.append(f"avg(h.{name}) AS review_hash_365_{index:02d}")
    query = f"""
        COPY (
            SELECT s.origin, s.product_id, {', '.join(columns)}
            FROM seed_keys s
            JOIN read_parquet('{hash_path}') h
              ON s.product_id = h.product_id
             AND h.review_time > s.origin - INTERVAL '365 days'
             AND h.review_time <= s.origin
            GROUP BY s.origin, s.product_id
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(query)
    con.close()
    os.replace(temporary, output)
    register_artifact(
        cache_dir,
        f"{CACHE_VERSION}-historical-hash-features",
        output,
        "Causal 91-day and 365-day product token-hash sketches",
        f"{CACHE_VERSION}:{hash_path.stat().st_size}:historical-hash",
        "Aggregate cached review token hashes by seed origin and product",
    )
    return output


# Product text

def build_product_embedding_cache(product_text: pd.DataFrame, cache_dir: Path, work_dir: Path) -> Path:
    output = cache_dir / f"{CACHE_VERSION}_minilm_{MINILM_REVISION[:12]}_pca64.parquet"
    if output.exists():
        return output
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import PCA
    from transformers.utils import logging as transformer_logging

    transformer_logging.disable_progress_bar()
    raw_path = cache_dir / f"{CACHE_VERSION}_minilm_{MINILM_REVISION[:12]}_raw384.npy"
    if raw_path.exists():
        raw = np.load(raw_path, mmap_mode="r")
        if raw.shape != (len(product_text), 384) or raw.dtype != np.float32:
            raise RuntimeError(f"invalid cached MiniLM matrix: {raw.shape} {raw.dtype}")
    else:
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        model = SentenceTransformer(
            MINILM_MODEL,
            revision=MINILM_REVISION,
            device=device,
            local_files_only=True,
        )
        if model.get_sentence_embedding_dimension() != 384:
            raise RuntimeError("all-MiniLM-L6-v2 embedding dimension is not 384")
        temporary_raw = raw_path.with_name(raw_path.name + f".{os.getpid()}.tmp")
        raw = np.lib.format.open_memmap(
            temporary_raw, mode="w+", dtype=np.float32, shape=(len(product_text), 384)
        )
        texts = product_text["product_text"].fillna("").astype(str).to_numpy()
        batch_rows = 12000
        for start in range(0, len(texts), batch_rows):
            stop = min(start + batch_rows, len(texts))
            raw[start:stop] = model.encode(
                texts[start:stop].tolist(),
                batch_size=512,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        raw.flush()
        del raw
        os.replace(temporary_raw, raw_path)
        raw = np.load(raw_path, mmap_mode="r")
    register_artifact(
        cache_dir,
        f"{CACHE_VERSION}-minilm-raw384",
        raw_path,
        "Pinned all-MiniLM-L6-v2 384-dimensional product embeddings before PCA",
        f"{CACHE_VERSION}:{MINILM_REVISION}:raw384:{len(product_text)}",
        "Encode brand, title, category, description with the pinned sentence transformer",
    )
    sample_size = min(100000, len(raw))
    sample_indices = np.linspace(0, len(raw) - 1, sample_size, dtype=np.int64)
    pca = PCA(n_components=EMBEDDING_DIM, svd_solver="randomized", random_state=SEED)
    pca.fit(np.asarray(raw[sample_indices]))
    temporary = output.with_name(output.name + f".{os.getpid()}.tmp")
    writer = None
    try:
        for start in range(0, len(raw), 30000):
            stop = min(start + 30000, len(raw))
            reduced = pca.transform(np.asarray(raw[start:stop])).astype(np.float32)
            arrays = {"product_id": pa.array(product_text["product_id"].to_numpy(dtype=np.int64)[start:stop])}
            for index in range(EMBEDDING_DIM):
                arrays[f"product_embedding_{index:02d}"] = pa.array(reduced[:, index])
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    del raw
    os.replace(temporary, output)
    register_artifact(
        cache_dir,
        f"{CACHE_VERSION}-minilm-pca64",
        output,
        "Pinned all-MiniLM-L6-v2 384-dimensional product text embeddings reduced to PCA-64",
        f"{CACHE_VERSION}:{MINILM_REVISION}:pca64:{len(product_text)}",
        "Encode brand, title, category, description and fit randomized PCA on a fixed 100k sample",
    )
    return output


# Feature assembly

def merge_keyed(frame: pd.DataFrame, feature_path: Path) -> pd.DataFrame:
    features = pd.read_parquet(feature_path)
    features = features.rename(columns={"origin": "timestamp"})
    return frame.merge(features, on=["timestamp", "product_id"], how="left", sort=False, validate="many_to_one")


def safe_ratio(numerator: pd.Series, denominator: pd.Series, smoothing: float = 1.0) -> np.ndarray:
    return ((numerator.to_numpy(dtype=np.float64) + smoothing) / (denominator.to_numpy(dtype=np.float64) + smoothing)).astype(np.float32)


def add_regime_resistant_features(frame: pd.DataFrame) -> pd.DataFrame:
    count_columns = ["c_lifetime"] + [f"c{window}" for window in WINDOWS] + ["c182_91", "c273_182", "c365_274"]
    customer_columns = [f"u{window}" for window in WINDOWS]
    for column in count_columns + customer_columns:
        frame[column] = frame[column].fillna(0).astype(np.float32)
    recency_columns = ["days_since_last_review", "days_since_first_review", "interarrival_mean_365", "interarrival_median_365"]
    for column in recency_columns:
        frame[column] = frame[column].fillna(9999).astype(np.float32)
    frame["ratio_7_to_28_rate"] = safe_ratio(frame["c7"], frame["c28"] / 4.0)
    frame["ratio_28_to_91_rate"] = safe_ratio(frame["c28"], frame["c91"] / 3.25)
    frame["ratio_91_to_365_rate"] = safe_ratio(frame["c91"], frame["c365"] / 4.0)
    frame["ratio_recent_to_previous_91"] = safe_ratio(frame["c91"], frame["c182_91"])
    frame["ratio_previous_quarters"] = safe_ratio(frame["c182_91"], frame["c273_182"])
    frame["same_season_to_annual_rate"] = safe_ratio(frame["c365_274"], frame["c365"] / 4.0)
    fast_weight = frame["c28"].to_numpy(dtype=np.float32) / (frame["c28"].to_numpy(dtype=np.float32) + SHRINKAGE_STRENGTH)
    fast_rate = frame["c28"].to_numpy(dtype=np.float32) * (91.0 / 28.0)
    slow_rate = frame["c365"].to_numpy(dtype=np.float32) * (91.0 / 365.0)
    frame["blended_fast_slow_count"] = (fast_weight * fast_rate + (1.0 - fast_weight) * slow_rate).astype(np.float32)
    frame["history_active_quarters"] = (
        (frame[["c91", "c182_91", "c273_182", "c365_274"]].to_numpy() > 0).sum(axis=1).astype(np.float32)
    )
    quarters = frame[["c91", "c182_91", "c273_182", "c365_274"]].to_numpy(dtype=np.float32)
    frame["product_history_median"] = np.median(quarters, axis=1).astype(np.float32)
    weights_100 = np.ones(4, dtype=np.float32)
    weights_098 = np.power(0.98, np.arange(4, dtype=np.float32))
    frame["product_history_ewma_100"] = (quarters @ weights_100 / weights_100.sum()).astype(np.float32)
    frame["product_history_ewma_098"] = (quarters @ weights_098 / weights_098.sum()).astype(np.float32)
    frame["same_season_count"] = frame["c365_274"].astype(np.float32)
    group_specs = [
        ("category", ["timestamp", "category_code"]),
        ("brand", ["timestamp", "brand_code"]),
        ("price_band", ["timestamp", "price_band"]),
    ]
    for name, group_columns in group_specs:
        group = frame.groupby(group_columns, observed=True, sort=False)["c91"]
        frame[f"{name}_c91_median"] = group.transform("median").astype(np.float32)
        frame[f"{name}_cohort_count"] = group.transform("size").astype(np.float32)
        total = group.transform("sum").to_numpy(dtype=np.float32)
        size = group.transform("size").to_numpy(dtype=np.float32)
        frame[f"activity_share_{name}"] = (frame["c91"].to_numpy(dtype=np.float32) / np.maximum(total, 1.0)).astype(np.float32)
        frame[f"activity_percentile_{name}"] = group.rank(method="average", pct=True).to_numpy(dtype=np.float32)
    global_median = frame.groupby("timestamp", sort=False)["c91"].transform("median").to_numpy(dtype=np.float32)
    category_count = frame["category_cohort_count"].to_numpy(dtype=np.float32)
    category_median = frame["category_c91_median"].to_numpy(dtype=np.float32)
    category_shrunk = (category_count * category_median + SHRINKAGE_STRENGTH * global_median) / (category_count + SHRINKAGE_STRENGTH)
    brand_count = frame["brand_cohort_count"].to_numpy(dtype=np.float32)
    brand_median = frame["brand_c91_median"].to_numpy(dtype=np.float32)
    brand_shrunk = (brand_count * brand_median + SHRINKAGE_STRENGTH * category_shrunk) / (brand_count + SHRINKAGE_STRENGTH)
    product_count = np.minimum(frame["c_lifetime"].to_numpy(dtype=np.float32), 10000.0)
    product_prior = frame["product_history_median"].to_numpy(dtype=np.float32)
    frame["global_c91_median"] = global_median
    frame["category_c91_shrunk"] = category_shrunk.astype(np.float32)
    frame["brand_c91_shrunk"] = brand_shrunk.astype(np.float32)
    frame["product_hierarchy_shrunk"] = ((product_count * product_prior + SHRINKAGE_STRENGTH * brand_shrunk) / (product_count + SHRINKAGE_STRENGTH)).astype(np.float32)
    for source in ["product_history_median", "same_season_count", "blended_fast_slow_count"]:
        for name, group_columns in group_specs:
            frame[f"{name}_{source}_median"] = (
                frame.groupby(group_columns, observed=True, sort=False)[source]
                .transform("median")
                .astype(np.float32)
            )
        source_global = frame.groupby("timestamp", sort=False)[source].transform("median").to_numpy(dtype=np.float32)
        source_category = frame[f"category_{source}_median"].to_numpy(dtype=np.float32)
        source_category_shrunk = (
            category_count * source_category + SHRINKAGE_STRENGTH * source_global
        ) / (category_count + SHRINKAGE_STRENGTH)
        source_brand = frame[f"brand_{source}_median"].to_numpy(dtype=np.float32)
        source_brand_shrunk = (
            brand_count * source_brand + SHRINKAGE_STRENGTH * source_category_shrunk
        ) / (brand_count + SHRINKAGE_STRENGTH)
        frame[f"{source}_hierarchy_shrunk"] = (
            (
                product_count * frame[source].to_numpy(dtype=np.float32)
                + SHRINKAGE_STRENGTH * source_brand_shrunk
            )
            / (product_count + SHRINKAGE_STRENGTH)
        ).astype(np.float32)
    log_sources = [
        "c_lifetime", "c7", "c28", "c91", "c182", "c365", "c182_91", "c273_182",
        "c365_274", "u91", "u365", "decay14", "decay45", "decay120", "decay365",
        "category_frequency", "brand_frequency", "audience_customer_count",
    ]
    for source in log_sources:
        frame[f"log1p_{source}"] = np.log1p(np.maximum(frame[source].fillna(0).to_numpy(dtype=np.float32), 0)).astype(np.float32)
    return frame


def select_time_decay(frame: pd.DataFrame) -> tuple[float, dict[str, float]]:
    train = frame[(frame["split"] == "train") & np.isfinite(frame["price"]) & (frame["price"] > 0)]
    origins = np.sort(train["timestamp"].unique())
    desired = pd.to_datetime(["2013-01-03", "2013-10-03", "2014-01-02", "2014-10-02", "2015-01-01"])
    fold_origins = [origin for origin in desired if np.datetime64(origin) in origins]
    scores: dict[str, float] = {}
    for decay, column in ((1.0, "product_history_ewma_100"), (0.98, "product_history_ewma_098")):
        fold_scores = []
        for origin in fold_origins:
            fold = train[train["timestamp"] == origin]
            prediction = np.maximum(1.0, fold[column].to_numpy(dtype=np.float64))
            truth = fold["ltv"].to_numpy(dtype=np.float64) / fold["price"].to_numpy(dtype=np.float64)
            fold_scores.append(float(np.mean(fold["price"].to_numpy(dtype=np.float64) * np.abs(truth - prediction))))
        scores[str(decay)] = float(np.median(fold_scores))
    selected = min((1.0, 0.98), key=lambda candidate: scores[str(candidate)])
    frame["product_history_ewma"] = frame["product_history_ewma_100" if selected == 1.0 else "product_history_ewma_098"].astype(np.float32)
    frame.drop(columns=["product_history_ewma_100", "product_history_ewma_098"], inplace=True)
    return selected, scores


def finalize_numeric_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[int]]:
    excluded = {"split", "row_id", "timestamp", "product_id", "ltv", "count_target", "brand_code"}
    feature_names = [
        column for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in feature_names:
        values = frame[column].to_numpy(dtype=np.float64, copy=False)
        values[~np.isfinite(values)] = np.nan
        frame[column] = values.astype(np.float32)
    categorical_names = [name for name in ["category_code", "price_band"] if name in feature_names]
    categorical_indices = [feature_names.index(name) for name in categorical_names]
    return frame, feature_names, categorical_indices


def build_feature_frame(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    static: pd.DataFrame,
    product_text: pd.DataFrame,
    paths: dict[str, Path],
    cache_dir: Path,
    work_dir: Path,
    debug: bool,
) -> tuple[pd.DataFrame, list[str], list[int], float, dict[str, float]]:
    started = time.time()
    keys = seed_keys(train, val, test)
    core_path = build_core_warehouse(paths, keys, cache_dir, work_dir)
    phase_log("core warehouse ready", started, str(core_path.name))
    audience_path = build_audience_warehouse(paths, keys, cache_dir, work_dir)
    phase_log("audience warehouse ready", started, str(audience_path.name))
    combined = pd.concat([train, val, test], ignore_index=True, sort=False)
    combined = merge_keyed(combined, core_path)
    combined = merge_keyed(combined, audience_path)
    combined = combined.merge(static, on="product_id", how="left", sort=False, validate="many_to_one")
    if debug:
        zero_features = {}
        for index in range(HASH_DIM):
            zero_features[f"review_hash_91_{index:02d}"] = np.zeros(len(combined), dtype=np.float32)
            zero_features[f"review_hash_365_{index:02d}"] = np.zeros(len(combined), dtype=np.float32)
        for index in range(EMBEDDING_DIM):
            zero_features[f"product_embedding_{index:02d}"] = np.zeros(len(combined), dtype=np.float32)
        combined = pd.concat([combined, pd.DataFrame(zero_features)], axis=1)
    else:
        hash_path = build_review_hash_cache(paths, cache_dir)
        phase_log("review token hashes ready", started, str(hash_path.name))
        hash_feature_path = build_historical_hash_features(hash_path, keys, cache_dir, work_dir)
        combined = merge_keyed(combined, hash_feature_path)
        phase_log("historical token hashes ready", started, str(hash_feature_path.name))
        embedding_path = build_product_embedding_cache(product_text, cache_dir, work_dir)
        embeddings = pd.read_parquet(embedding_path)
        combined = combined.merge(embeddings, on="product_id", how="left", sort=False, validate="many_to_one")
        phase_log("MiniLM PCA features ready", started, str(embedding_path.name))
    combined = add_regime_resistant_features(combined.copy())
    selected_decay, decay_scores = select_time_decay(combined)
    combined, feature_names, categorical_indices = finalize_numeric_features(combined)
    for split, original in (("train", train), ("val", val), ("test", test)):
        observed = combined.loc[combined["split"] == split, "row_id"].to_numpy(dtype=np.int64)
        expected = original["row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(observed, expected):
            raise RuntimeError(f"row_id alignment changed for {split}")
    phase_log("feature frame assembled", started, f"rows={len(combined)} features={len(feature_names)} decay={selected_decay}")
    return combined, feature_names, categorical_indices, selected_decay, decay_scores
