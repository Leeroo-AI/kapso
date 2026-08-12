from __future__ import annotations

import hashlib
import json
import math
import os
import time
import fcntl
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as functional
from numba import njit
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


VERSION = "lane2_cutoff_transformer_v1"
PRODUCT_VOCAB = 131072
CATEGORY_VOCAB = 8192
BRAND_VOCAB = 8192
GAP_BUCKETS = 32
HIDDEN = 160
HEADS = 4
LAYERS = 3
FFN = 448
DROPOUT = 0.10
FULL_LENGTH = 64
DEBUG_LENGTH = 16
BATCH_SIZE = 1024
PRETRAIN_LR = 5e-4
FINETUNE_LR = 1.5e-4
FOLD_ORIGINS = (20, 24, 30)
SEMANTIC_NAMES = ("reader_intent", "engagement", "sentiment", "specificity", "risk", "future_intent")


def cache_root() -> Path:
    root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / VERSION
    root.mkdir(parents=True, exist_ok=True)
    return root


def database_root() -> Path:
    return Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]


def task_path(split: str) -> Path:
    return database_root() / "tasks" / os.environ["RELBENCH_TASK"] / f"{split}.parquet"


def connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    return con


def announce(phase: str, started: float, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[lane2] phase={phase} elapsed={time.time() - started:.1f}s{suffix}", flush=True)


def register_artifact(name: str, path: Path, description: str, key: str) -> None:
    registry = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "artifacts.json"
    lock_path = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "artifacts.lock"
    relative = str(path.relative_to(Path(os.environ["KAPSO_SHARED_CACHE_DIR"])))
    record = {"name": name, "path": relative, "description": description, "content_key": key, "rebuild_hint": f"run main.py with {VERSION}"}
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        entries = []
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except Exception:
                entries = []
        if not any(row.get("path") == relative for row in entries):
            entries.append(record)
            temporary = registry.with_name(f"artifacts_{os.getpid()}.json")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        env_path = parent / ".env"
        if not env_path.exists():
            continue
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                key = line.split("=", 1)[1].strip().strip("'\"")
                if key:
                    os.environ["OPENAI_API_KEY"] = key
                    return key
    raise RuntimeError("OPENAI_API_KEY was not available for the mandatory hosted semantic measurement")


def teacher_schema() -> dict:
    properties = {name: {"type": "number", "minimum": 0, "maximum": 1} for name in SEMANTIC_NAMES}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "review_semantics",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "rows": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}, **properties},
                            "required": ["id", *SEMANTIC_NAMES],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["rows"],
                "additionalProperties": False,
            },
        },
    }


def select_teacher_rows(earliest_cutoff: pd.Timestamp, debug: bool) -> pd.DataFrame:
    limit = 64 if debug else 4096
    review = database_root() / "db" / "review.parquet"
    query = f"""
        SELECT customer_id, product_id, review_time,
               substr(coalesce(summary, ''), 1, 180) AS summary,
               substr(coalesce(review_text, ''), 1, 700) AS review_text
        FROM read_parquet('{review}')
        WHERE review_time <= TIMESTAMP '{earliest_cutoff}'
        ORDER BY hash(customer_id, product_id, review_time)
        LIMIT {limit}
    """
    frame = connection().execute(query).fetch_df()
    frame.insert(0, "id", np.arange(len(frame), dtype=np.int64))
    return frame


def request_teacher_batch(client, rows: pd.DataFrame) -> list[dict]:
    records = []
    for row in rows.itertuples(index=False):
        records.append({"id": int(row.id), "summary": str(row.summary), "review": str(row.review_text)})
    prompt = (
        "Read each Amazon Books review as evidence about the reader and review state. "
        "Return reader_intent as commitment to reading/buying, engagement as depth of use, "
        "sentiment from negative 0 to positive 1, specificity as concrete detail, risk as dissatisfaction/return risk, "
        "and future_intent as likelihood of future reviewing or purchasing. Use calibrated numbers in [0,1].\n"
        + json.dumps(records, ensure_ascii=False)
    )
    response = client.chat.completions.create(
        model="gpt-5.6-luna",
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort="low",
        response_format=teacher_schema(),
        max_completion_tokens=12000,
    )
    return json.loads(response.choices[0].message.content)["rows"]


def ensure_teacher(earliest_cutoff: pd.Timestamp, debug: bool, started: float) -> tuple[Path, int, int]:
    from openai import OpenAI

    root = cache_root()
    suffix = "debug" if debug else "full"
    input_path = root / f"teacher_input_{suffix}.parquet"
    label_path = root / f"teacher_labels_{suffix}.parquet"
    audit_path = root / "teacher_full_audit.json"
    if label_path.exists() and input_path.exists():
        rows = pd.read_parquet(input_path)
        if not debug:
            client = OpenAI(api_key=load_api_key(), timeout=120.0, max_retries=2)
            calls = 0
            audited = 0
            audit_rows = rows.iloc[:32] if audit_path.exists() else rows
            for beginning in range(0, len(audit_rows), 64):
                batch = audit_rows.iloc[beginning : beginning + 64]
                last_error = None
                for _ in range(3):
                    try:
                        audit = request_teacher_batch(client, batch)
                        calls += 1
                        audited += len(batch)
                        last_error = None
                        break
                    except Exception as error:
                        calls += 1
                        last_error = error
                if last_error is not None:
                    raise RuntimeError(f"hosted teacher audit failed after retries: {last_error}")
            audit_path.write_text(json.dumps({"rows": audited, "calls": calls, "model": "gpt-5.6-luna"}))
            announce("llm_teacher_audit", started, f"calls={calls} rows={audited} model=gpt-5.6-luna cached=1")
            return label_path, calls, audited
        return label_path, 0, 0
    rows = select_teacher_rows(earliest_cutoff, debug)
    rows.to_parquet(input_path, index=False)
    if debug:
        values = np.zeros((len(rows), len(SEMANTIC_NAMES)), dtype=np.float32)
        text = (rows["summary"].fillna("") + " " + rows["review_text"].fillna("")).str.lower()
        values[:, 0] = text.str.contains("read|book|author", regex=True).to_numpy(np.float32)
        values[:, 1] = np.clip(text.str.len().to_numpy(np.float32) / 700.0, 0, 1)
        values[:, 2] = 0.5
        values[:, 3] = values[:, 1]
        values[:, 4] = text.str.contains("bad|return|waste|disappoint", regex=True).to_numpy(np.float32)
        values[:, 5] = text.str.contains("again|next|more|recommend", regex=True).to_numpy(np.float32)
        labels = rows[["id"]].copy()
        for index, name in enumerate(SEMANTIC_NAMES):
            labels[name] = values[:, index]
        labels.to_parquet(label_path, index=False)
        return label_path, 0, 0
    client = OpenAI(api_key=load_api_key(), timeout=180.0, max_retries=2)
    output = []
    calls = 0
    for beginning in range(0, len(rows), 64):
        batch = rows.iloc[beginning : beginning + 64]
        pending = batch.copy()
        mapped = {}
        last_error = None
        for _ in range(4):
            try:
                calls += 1
                result = request_teacher_batch(client, pending)
                expected = set(int(value) for value in pending["id"])
                mapped.update({int(row["id"]): row for row in result if int(row["id"]) in expected})
                missing = [int(value) for value in batch["id"] if int(value) not in mapped]
                if missing:
                    pending = batch[batch["id"].isin(missing)].copy()
                    last_error = ValueError(f"teacher response omitted {len(missing)} ids")
                    continue
                last_error = None
                break
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise RuntimeError(f"hosted teacher failed after retries: {last_error}")
        output.extend(mapped[int(value)] for value in batch["id"])
        print(f"[lane2] llm_teacher_batch={calls} rows={len(output)}", flush=True)
    labels = pd.DataFrame(output)
    labels = labels[["id", *SEMANTIC_NAMES]].sort_values("id").reset_index(drop=True)
    labels.to_parquet(label_path, index=False)
    audit_path.write_text(json.dumps({"rows": len(labels), "calls": calls, "model": "gpt-5.6-luna"}))
    announce("llm_teacher_complete", started, f"calls={calls} rows={len(labels)} model=gpt-5.6-luna cutoff={earliest_cutoff.date()}")
    register_artifact("lane2 hosted review semantics", label_path, "gpt-5.6-luna structured review attributes before earliest forward cutoff", VERSION + "_teacher_4096")
    return label_path, calls, len(labels)


def ensure_distiller(earliest_cutoff: pd.Timestamp, debug: bool, started: float) -> tuple[HashingVectorizer, Ridge, dict]:
    root = cache_root()
    suffix = "debug" if debug else "full"
    model_path = root / f"semantic_distiller_{suffix}.joblib"
    label_path, calls, teacher_rows = ensure_teacher(earliest_cutoff, debug, started)
    if model_path.exists():
        vectorizer, model, metadata = joblib.load(model_path)
        metadata = {**metadata, "hosted_calls": calls, "hosted_rows": teacher_rows}
        return vectorizer, model, metadata
    inputs = pd.read_parquet(root / f"teacher_input_{suffix}.parquet")
    labels = pd.read_parquet(label_path)
    merged = inputs.merge(labels, on="id", how="inner", validate="one_to_one")
    text = (merged["summary"].fillna("") + " " + merged["review_text"].fillna("")).tolist()
    vectorizer = HashingVectorizer(n_features=8192, alternate_sign=False, norm="l2", ngram_range=(1, 2), dtype=np.float32)
    x = vectorizer.transform(text)
    y = merged[list(SEMANTIC_NAMES)].to_numpy(np.float32)
    model = Ridge(alpha=20.0, fit_intercept=True, solver="lsqr")
    model.fit(x, y)
    fitted = np.clip(model.predict(x), 0, 1)
    metadata = {
        "teacher_cutoff": str(earliest_cutoff),
        "teacher_rows": int(len(labels)),
        "hosted_calls": calls,
        "hosted_rows": teacher_rows,
        "train_mae": float(np.mean(np.abs(fitted - y))),
    }
    joblib.dump((vectorizer, model, metadata), model_path)
    register_artifact("lane2 hashed semantic distiller", model_path, "fixed 8192-feature hashed-text ridge distiller", VERSION + "_distiller")
    announce("semantic_distiller", started, f"teacher_rows={len(labels)} train_mae={metadata['train_mae']:.5f}")
    return vectorizer, model, metadata


def ensure_event_days(debug: bool, started: float) -> Path:
    root = cache_root()
    suffix = "debug" if debug else "full"
    path = root / f"event_days_{suffix}.parquet"
    if path.exists():
        return path
    review = database_root() / "db" / "review.parquet"
    product = database_root() / "db" / "product.parquet"
    customer_filter = ""
    if debug:
        train = task_path("train")
        customer_filter = f"JOIN (SELECT DISTINCT customer_id FROM read_parquet('{train}') ORDER BY hash(customer_id) LIMIT 30000) selected USING (customer_id)"
    temporary = root / f"event_days_{suffix}_{os.getpid()}.parquet"
    query = f"""
        COPY (
            WITH pooled AS (
                SELECT
                    r.customer_id,
                    CAST(epoch(CAST(r.review_time AS DATE)) / 86400 AS INTEGER) AS event_day,
                    count(*) AS multiplicity,
                    avg(r.rating) AS rating,
                    avg(CAST(r.verified AS FLOAT)) AS verified,
                    avg(length(coalesce(r.review_text, ''))) AS text_length,
                    avg(length(coalesce(r.summary, ''))) AS summary_length,
                    arg_max(substr(coalesce(r.review_text, ''), 1, 900), r.review_time) AS review_text,
                    arg_max(substr(coalesce(r.summary, ''), 1, 220), r.review_time) AS summary,
                    arg_max(r.product_id, r.review_time) AS product_id,
                    arg_max(CAST(hash(coalesce(CAST(p.category AS VARCHAR), '')) % {CATEGORY_VOCAB} AS INTEGER), r.review_time) AS category_id,
                    arg_max(CAST(hash(coalesce(p.brand, '')) % {BRAND_VOCAB} AS INTEGER), r.review_time) AS brand_id,
                    arg_max(coalesce(p.price, 0), r.review_time) AS price
                FROM read_parquet('{review}') r
                {customer_filter}
                LEFT JOIN read_parquet('{product}') p USING (product_id)
                GROUP BY r.customer_id, CAST(r.review_time AS DATE)
            ), product_daily AS (
                SELECT
                    product_id,
                    CAST(epoch(CAST(review_time AS DATE)) / 86400 AS INTEGER) AS event_day,
                    count(*) AS daily_reviews
                FROM read_parquet('{review}')
                GROUP BY product_id, CAST(review_time AS DATE)
            ), product_history AS (
                SELECT product_id, event_day, coalesce(sum(daily_reviews) OVER (
                    PARTITION BY product_id ORDER BY event_day
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS previous_product_reviews
                FROM product_daily
            ), marked AS (
                SELECT pooled.*, coalesce(product_history.previous_product_reviews, 0) AS previous_product_reviews
                FROM pooled LEFT JOIN product_history USING (product_id, event_day)
            )
            SELECT
                customer_id, event_day, CAST(multiplicity AS USMALLINT) AS multiplicity,
                CAST(rating AS FLOAT) AS rating, CAST(verified AS FLOAT) AS verified,
                CAST(text_length AS FLOAT) AS text_length, CAST(summary_length AS FLOAT) AS summary_length,
                review_text, summary, CAST(product_id % {PRODUCT_VOCAB} AS INTEGER) AS product_id,
                category_id, brand_id, CAST(price AS FLOAT) AS price,
                CAST(least(31, floor(log2(1 + previous_product_reviews))) AS UTINYINT) AS popularity_bucket
            FROM marked ORDER BY customer_id, event_day
        ) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
    """
    connection().execute(query)
    os.replace(temporary, path)
    announce("event_day_build", started, f"rows={pq.ParquetFile(path).metadata.num_rows}")
    register_artifact("lane2 event-day sequence table", path, "same-day pooled customer review events with causal product popularity", VERSION + f"_events_{suffix}")
    return path


def ensure_event_semantics(path: Path, vectorizer: HashingVectorizer, model: Ridge, debug: bool, started: float) -> Path:
    suffix = "debug" if debug else "full"
    output = cache_root() / f"event_semantics_{suffix}.npy"
    rows = pq.ParquetFile(path).metadata.num_rows
    if output.exists():
        array = np.load(output, mmap_mode="r")
        if array.shape == (rows, len(SEMANTIC_NAMES)):
            return output
    temporary = cache_root() / f"event_semantics_{suffix}_{os.getpid()}.npy"
    destination = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(rows, len(SEMANTIC_NAMES)))
    offset = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=65536, columns=["summary", "review_text"]):
        frame = batch.to_pandas()
        text = (frame["summary"].fillna("") + " " + frame["review_text"].fillna("")).tolist()
        values = np.clip(model.predict(vectorizer.transform(text)), 0, 1).astype(np.float16)
        destination[offset : offset + len(values)] = values
        offset += len(values)
        if offset % 1048576 < len(values):
            print(f"[lane2] semantic_distill_rows={offset}", flush=True)
    destination.flush()
    del destination
    os.replace(temporary, output)
    announce("event_semantics", started, f"rows={rows}")
    register_artifact("lane2 event semantic scores", output, "six fixed LLM-distilled semantic scores per event-day", VERSION + f"_event_semantics_{suffix}")
    return output


@njit(cache=True)
def upper_bound(values, left, right, target):
    while left < right:
        middle = (left + right) // 2
        if values[middle] <= target:
            left = middle + 1
        else:
            right = middle
    return left


@njit(cache=True)
def segment_sample_arrays(pointer, days, low_day, high_day, length):
    total = 0
    for customer in range(len(pointer) - 1):
        left = pointer[customer]
        right = pointer[customer + 1]
        beginning = upper_bound(days, left, right, low_day)
        ending = upper_bound(days, left, right, high_day)
        beginning = max(beginning, left + 1)
        if ending > beginning:
            total += (ending - beginning + length - 1) // length
    starts = np.empty(total, dtype=np.int64)
    sizes = np.empty(total, dtype=np.int16)
    customers = np.empty(total, dtype=np.int32)
    position = 0
    for customer in range(len(pointer) - 1):
        left = pointer[customer]
        right = pointer[customer + 1]
        beginning = upper_bound(days, left, right, low_day)
        ending = upper_bound(days, left, right, high_day)
        beginning = max(beginning, left + 1)
        while beginning < ending:
            size = min(length, ending - beginning)
            starts[position] = beginning - 1
            sizes[position] = size
            customers[position] = customer
            position += 1
            beginning += size
    return starts, sizes, customers


@njit(cache=True)
def lifetime_for_samples(pointer, days, categories, starts, customers):
    result = np.zeros((len(starts), 9), dtype=np.float32)
    edges = np.exp(np.linspace(np.log(1.0), np.log(2920.0), GAP_BUCKETS + 1))
    for row in range(len(starts)):
        left = pointer[customers[row]]
        end = starts[row]
        count = end - left
        if count <= 0:
            continue
        histogram = np.zeros(GAP_BUCKETS, dtype=np.int32)
        total = 0.0
        square = 0.0
        mask = np.uint64(0)
        for index in range(left, end):
            mask |= np.uint64(1) << np.uint64(categories[index] % 64)
            if index > left:
                gap = max(1, days[index] - days[index - 1])
                bucket = min(GAP_BUCKETS - 1, int(np.log(gap) / np.log(2920.0) * GAP_BUCKETS))
                histogram[bucket] += 1
                total += gap
                square += gap * gap
        gap_count = max(1, count - 1)
        mean = total / gap_count
        std = np.sqrt(max(0.0, square / gap_count - mean * mean))
        quantiles = np.zeros(3, dtype=np.float32)
        targets = np.array([0.25, 0.5, 0.75]) * gap_count
        cumulative = 0
        target_index = 0
        for bucket in range(GAP_BUCKETS):
            cumulative += histogram[bucket]
            while target_index < 3 and cumulative >= targets[target_index]:
                quantiles[target_index] = np.sqrt(edges[bucket] * edges[bucket + 1])
                target_index += 1
        bits = 0
        current = mask
        while current:
            bits += int(current & np.uint64(1))
            current >>= np.uint64(1)
        diversity = -64.0 * np.log(max(1.0 / 64.0, 1.0 - bits / 64.0))
        result[row, 0] = np.log1p(count)
        result[row, 1] = np.log1p(max(0, days[end - 1] - days[left]))
        result[row, 2] = np.log1p(mean)
        result[row, 3] = np.log1p(std)
        result[row, 4] = np.log1p(quantiles[0])
        result[row, 5] = np.log1p(quantiles[1])
        result[row, 6] = np.log1p(quantiles[2])
        result[row, 7] = np.log1p(diversity)
        result[row, 8] = bits / 64.0
    return result


@njit(cache=True)
def seed_index_arrays(pointer, days, multiplicity, customers, seed_days, length, system_activity):
    starts = np.empty(len(customers), dtype=np.int64)
    sizes = np.empty(len(customers), dtype=np.int16)
    query = np.zeros((len(customers), 11), dtype=np.float32)
    for row in range(len(customers)):
        customer = customers[row]
        left = pointer[customer]
        right = pointer[customer + 1]
        end = upper_bound(days, left, right, seed_days[row])
        start = max(left, end - length)
        starts[row] = start
        sizes[row] = end - start
        if end <= left:
            continue
        recency = seed_days[row] - days[end - 1]
        active7 = 0
        active30 = 0
        active91 = 0
        reviews91 = 0
        for index in range(end - 1, left - 1, -1):
            age = seed_days[row] - days[index]
            if age > 91:
                break
            active91 += 1
            reviews91 += multiplicity[index]
            active30 += age <= 30
            active7 += age <= 7
        angle = 2.0 * np.pi * ((seed_days[row] + 4) % 365.2425) / 365.2425
        query[row, 0] = np.log1p(recency)
        query[row, 1] = np.log1p(active7)
        query[row, 2] = np.log1p(active30)
        query[row, 3] = np.log1p(active91)
        query[row, 4] = np.log1p(reviews91)
        query[row, 5] = np.sin(angle)
        query[row, 6] = np.cos(angle)
        query[row, 7] = np.sin(2.0 * angle)
        query[row, 8] = np.cos(2.0 * angle)
        query[row, 9] = system_activity[row]
        query[row, 10] = active91 / max(1.0, recency + 1.0)
    return starts, sizes, query


class EventStore:
    def __init__(self, event_path: Path, semantic_path: Path, debug: bool, started: float):
        columns = [
            "customer_id", "event_day", "multiplicity", "rating", "verified", "text_length",
            "summary_length", "product_id", "category_id", "brand_id", "price", "popularity_bucket",
        ]
        frame = pq.read_table(event_path, columns=columns).to_pandas()
        self.customer = frame.pop("customer_id").to_numpy(np.int32)
        self.day = frame.pop("event_day").to_numpy(np.int32)
        self.multiplicity = frame.pop("multiplicity").to_numpy(np.int16)
        self.rating = frame.pop("rating").to_numpy(np.float32)
        self.verified = frame.pop("verified").to_numpy(np.float32)
        self.text_length = frame.pop("text_length").to_numpy(np.float32)
        self.summary_length = frame.pop("summary_length").to_numpy(np.float32)
        self.product = frame.pop("product_id").to_numpy(np.int32)
        self.category = frame.pop("category_id").to_numpy(np.int16)
        self.brand = frame.pop("brand_id").to_numpy(np.int16)
        self.price = frame.pop("price").to_numpy(np.float32)
        self.popularity = frame.pop("popularity_bucket").to_numpy(np.int8)
        self.semantic = np.asarray(np.load(semantic_path, mmap_mode="r"), dtype=np.float32)
        customer_count = 1_850_193
        counts = np.bincount(self.customer, minlength=customer_count)
        self.pointer = np.empty(customer_count + 1, dtype=np.int64)
        self.pointer[0] = 0
        np.cumsum(counts, out=self.pointer[1:])
        self.length = DEBUG_LENGTH if debug else FULL_LENGTH
        self.global_review_days = np.repeat(self.day, np.maximum(1, self.multiplicity.astype(np.int32)))
        self.global_review_days.sort()
        announce("event_store", started, f"event_days={len(self.day)} customers={(counts > 0).sum()} sequence_length={self.length}")

    def system_activity(self, seed_days: np.ndarray) -> np.ndarray:
        high = np.searchsorted(self.global_review_days, seed_days, side="right")
        low = np.searchsorted(self.global_review_days, seed_days - 91, side="left")
        return (np.log1p(high - low) / 16.0).astype(np.float32)

    def segment_samples(self, low_day: int, high_day: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        starts, sizes, customers = segment_sample_arrays(self.pointer, self.day, low_day, high_day, self.length)
        lifetime = lifetime_for_samples(self.pointer, self.day, self.category, starts, customers)
        order = np.argsort(self.day[starts + sizes], kind="stable")
        return starts[order], sizes[order], customers[order], lifetime[order]

    def seed_arrays(self, customers: np.ndarray, seed_days: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        system = self.system_activity(seed_days)
        starts, sizes, query = seed_index_arrays(self.pointer, self.day, self.multiplicity, customers, seed_days, self.length, system)
        lifetime = lifetime_for_samples(self.pointer, self.day, self.category, starts, customers)
        return starts, sizes, lifetime, query

    def event_batch(self, starts: np.ndarray, sizes: np.ndarray) -> dict[str, torch.Tensor]:
        positions = np.arange(self.length, dtype=np.int64)[None, :]
        indices = starts[:, None] + positions
        valid = positions < sizes[:, None]
        safe = np.where(valid, indices, 0)
        previous = np.maximum(safe - 1, 0)
        gaps = np.where(valid, np.maximum(0, self.day[safe] - self.day[previous]), 0).astype(np.float32)
        day = self.day[safe].astype(np.float32)
        annual = 2.0 * np.pi * ((day + 4.0) % 365.2425) / 365.2425
        numeric = np.stack(
            [
                np.log1p(gaps) / 8.0,
                np.sin(annual), np.cos(annual), np.sin(2.0 * annual), np.cos(2.0 * annual),
                (self.rating[safe] - 3.0) / 2.0,
                self.verified[safe],
                np.log1p(self.multiplicity[safe]) / 4.0,
                np.log1p(self.text_length[safe]) / 8.0,
                np.log1p(self.summary_length[safe]) / 5.0,
                np.log1p(np.maximum(0, self.price[safe])) / 6.0,
                self.popularity[safe].astype(np.float32) / 20.0,
                *[np.asarray(self.semantic[:, index], dtype=np.float32)[safe] for index in range(len(SEMANTIC_NAMES))],
            ],
            axis=-1,
        ).astype(np.float32)
        numeric[~valid] = 0
        return {
            "indices": torch.from_numpy(safe),
            "valid": torch.from_numpy(valid),
            "numeric": torch.from_numpy(numeric),
            "product": torch.from_numpy(np.where(valid, self.product[safe], 0).astype(np.int64)),
            "category": torch.from_numpy(np.where(valid, self.category[safe], 0).astype(np.int64)),
            "brand": torch.from_numpy(np.where(valid, self.brand[safe], 0).astype(np.int64)),
        }

    def target_batch(self, event_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        safe = event_batch["indices"].numpy()
        valid = event_batch["valid"].numpy()
        target = safe + 1
        gap = np.maximum(1, self.day[target] - self.day[safe])
        bucket = np.clip((np.log(gap) / np.log(2920.0) * GAP_BUCKETS).astype(np.int64), 0, GAP_BUCKETS - 1)
        rating = np.clip(np.rint(self.rating[target]).astype(np.int64) - 1, 0, 4)
        return {
            "gap": torch.from_numpy(np.where(valid, bucket, -100)),
            "category": torch.from_numpy(np.where(valid, self.category[target].astype(np.int64), -100)),
            "rating": torch.from_numpy(np.where(valid, rating, -100)),
            "verified": torch.from_numpy(np.where(valid, self.verified[target], 0).astype(np.float32)),
        }


class TemporalTransformer(nn.Module):
    def __init__(self, sequence_length: int):
        super().__init__()
        self.sequence_length = sequence_length
        self.product_embedding = nn.Embedding(PRODUCT_VOCAB, 32)
        self.category_embedding = nn.Embedding(CATEGORY_VOCAB, 16)
        self.brand_embedding = nn.Embedding(BRAND_VOCAB, 16)
        self.numeric_projection = nn.Sequential(nn.Linear(18, 96), nn.GELU(), nn.LayerNorm(96))
        self.event_projection = nn.Linear(160, HIDDEN)
        self.lifetime_projection = nn.Sequential(nn.Linear(9, HIDDEN), nn.GELU(), nn.LayerNorm(HIDDEN))
        self.query_projection = nn.Sequential(nn.Linear(11, HIDDEN), nn.GELU(), nn.LayerNorm(HIDDEN))
        self.position = nn.Embedding(sequence_length + 2, HIDDEN)
        layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN,
            nhead=HEADS,
            dim_feedforward=FFN,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=LAYERS, norm=nn.LayerNorm(HIDDEN))
        self.gap_head = nn.Linear(HIDDEN, GAP_BUCKETS)
        self.category_head = nn.Linear(HIDDEN, CATEGORY_VOCAB)
        self.rating_head = nn.Linear(HIDDEN, 5)
        self.verified_head = nn.Linear(HIDDEN, 1)
        self.binary_head = nn.Linear(HIDDEN, 1)

    def event_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        numeric = self.numeric_projection(batch["numeric"])
        marks = torch.cat(
            [
                self.product_embedding(batch["product"]),
                self.category_embedding(batch["category"]),
                self.brand_embedding(batch["brand"]),
                numeric,
            ],
            dim=-1,
        )
        return self.event_projection(marks)

    def causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def encode_events(self, batch: dict[str, torch.Tensor], lifetime: torch.Tensor) -> torch.Tensor:
        event = self.event_tokens(batch)
        life = self.lifetime_projection(lifetime).unsqueeze(1)
        tokens = torch.cat([life, event], dim=1)
        positions = self.position(torch.arange(tokens.shape[1], device=tokens.device)).unsqueeze(0)
        tokens = tokens + positions
        padding = torch.cat([torch.zeros((len(event), 1), dtype=torch.bool, device=tokens.device), ~batch["valid"]], dim=1)
        return self.transformer(tokens, mask=self.causal_mask(tokens.shape[1], tokens.device), src_key_padding_mask=padding)

    def pretraining_logits(self, batch: dict[str, torch.Tensor], lifetime: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.encode_events(batch, lifetime)[:, 1:]
        return self.gap_head(encoded), self.category_head(encoded), self.rating_head(encoded), self.verified_head(encoded).squeeze(-1)

    def last_event_state(self, batch: dict[str, torch.Tensor], lifetime: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_events(batch, lifetime)
        rows = torch.arange(len(encoded), device=encoded.device)
        positions = batch["valid"].sum(dim=1).long()
        return encoded[rows, positions]

    def classification_logits(self, batch: dict[str, torch.Tensor], lifetime: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        event = self.event_tokens(batch)
        life = self.lifetime_projection(lifetime).unsqueeze(1)
        query_token = self.query_projection(query).unsqueeze(1)
        tokens = torch.cat([life, event, query_token], dim=1)
        tokens = tokens + self.position(torch.arange(tokens.shape[1], device=tokens.device)).unsqueeze(0)
        padding = torch.cat(
            [
                torch.zeros((len(event), 1), dtype=torch.bool, device=tokens.device),
                ~batch["valid"],
                torch.zeros((len(event), 1), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        encoded = self.transformer(tokens, mask=self.causal_mask(tokens.shape[1], tokens.device), src_key_padding_mask=padding)
        return self.binary_head(encoded[:, -1]).squeeze(-1)


class FrozenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(HIDDEN + 9 + 11, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, state: torch.Tensor, lifetime: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([state, lifetime, query], dim=1)).squeeze(-1)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def state_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def pretraining_loss(logits: tuple[torch.Tensor, ...], targets: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
    gap_logits, category_logits, rating_logits, verified_logits = logits
    gap = functional.cross_entropy(gap_logits.flatten(0, 1), targets["gap"].flatten(), ignore_index=-100)
    category = functional.cross_entropy(category_logits.flatten(0, 1), targets["category"].flatten(), ignore_index=-100)
    rating = functional.cross_entropy(rating_logits.flatten(0, 1), targets["rating"].flatten(), ignore_index=-100)
    mask = targets["gap"] != -100
    verified = functional.binary_cross_entropy_with_logits(verified_logits[mask], targets["verified"][mask])
    loss = gap + 0.35 * category + 0.20 * (rating + verified)
    return loss, {"gap": float(gap.detach()), "category": float(category.detach()), "rating": float(rating.detach()), "verified": float(verified.detach())}


def load_origins() -> list[pd.Timestamp]:
    frame = connection().execute(f"SELECT DISTINCT timestamp FROM read_parquet('{task_path('train')}') ORDER BY timestamp").fetch_df()
    return list(pd.to_datetime(frame["timestamp"]))


def checkpoint_specs(origins: list[pd.Timestamp]) -> list[tuple[str, int]]:
    return [
        ("origin20", int(origins[20].timestamp() // 86400)),
        ("origin24", int(origins[24].timestamp() // 86400)),
        ("origin30", int(origins[30].timestamp() // 86400)),
        ("validation", int(pd.Timestamp("2015-10-01").timestamp() // 86400)),
        ("test", int(pd.Timestamp("2016-01-01").timestamp() // 86400)),
    ]


def ensure_pretrained(store: EventStore, origins: list[pd.Timestamp], debug: bool, started: float) -> tuple[dict[str, Path], dict]:
    device = torch.device("cuda")
    model = TemporalTransformer(store.length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=PRETRAIN_LR, weight_decay=0.02)
    checkpoints = {}
    diagnostics = {"segments": []}
    previous_day = int(store.day.min()) - 1
    specs = checkpoint_specs(origins)
    if debug:
        specs = [("origin20", specs[0][1])]
    for name, cutoff_day in specs:
        checkpoint = cache_root() / f"trunk_{name}_{'debug' if debug else 'full'}.pt"
        checkpoints[name] = checkpoint
        if checkpoint.exists():
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if not debug or int(payload.get("diagnostics", {}).get("steps", 0)) >= 300:
                model.load_state_dict(payload["model"])
                optimizer.load_state_dict(payload["optimizer"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)
                previous_day = cutoff_day
                diagnostics["segments"].append({"checkpoint": name, "cached": True, **payload.get("diagnostics", {})})
                continue
        starts, sizes, _, lifetime = store.segment_samples(previous_day, cutoff_day)
        if len(starts) == 0:
            raise RuntimeError(f"no self-supervised samples for checkpoint {name}")
        model.train()
        losses = []
        steps = 0
        segment_started = time.time()
        epochs = max(2, int(math.ceil(300 / max(1, math.ceil(len(starts) / BATCH_SIZE))))) if debug else 2
        maximum_steps = 300 if debug else None
        for epoch in range(epochs):
            for beginning in range(0, len(starts), BATCH_SIZE):
                selected = slice(beginning, min(len(starts), beginning + BATCH_SIZE))
                event = store.event_batch(starts[selected], sizes[selected])
                targets = store.target_batch(event)
                event = move_batch(event, device)
                targets = move_batch(targets, device)
                life = torch.from_numpy(lifetime[selected]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model.pretraining_logits(event, life)
                    loss, parts = pretraining_loss(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                steps += 1
                if steps % 100 == 0:
                    elapsed = time.time() - segment_started
                    rate = steps * BATCH_SIZE / max(elapsed, 1)
                    print(f"[lane2] pretrain_checkpoint={name} steps={steps} sequences_per_second={rate:.1f} loss={np.mean(losses[-100:]):.5f}", flush=True)
                if maximum_steps is not None and steps >= maximum_steps:
                    break
            if maximum_steps is not None and steps >= maximum_steps:
                break
        elapsed = time.time() - segment_started
        segment_diagnostics = {
            "samples": int(len(starts)),
            "steps": steps,
            "epochs": epochs,
            "mean_loss": float(np.mean(losses)),
            "sequences_per_second": float(steps * BATCH_SIZE / max(elapsed, 1)),
            "cutoff_day": cutoff_day,
        }
        payload = {"model": state_cpu(model), "optimizer": optimizer.state_dict(), "diagnostics": segment_diagnostics}
        temporary = checkpoint.with_name(f"{checkpoint.stem}_{os.getpid()}.pt")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        register_artifact(f"lane2 cutoff trunk {name}", checkpoint, f"causal temporal transformer through {name}", VERSION + f"_{name}")
        diagnostics["segments"].append({"checkpoint": name, "cached": False, **segment_diagnostics})
        previous_day = cutoff_day
        announce("pretrain_checkpoint", started, f"name={name} steps={steps} sequences_per_second={segment_diagnostics['sequences_per_second']:.1f}")
    return checkpoints, diagnostics


def load_split(split: str, origins: list[pd.Timestamp]) -> dict[str, np.ndarray]:
    columns = "file_row_number AS row_id, customer_id, CAST(epoch(timestamp) / 86400 AS INTEGER) AS seed_day"
    if split != "test":
        columns += ", churn"
    frame = connection().execute(f"SELECT {columns} FROM read_parquet('{task_path(split)}', file_row_number=true) ORDER BY row_id").fetch_df()
    result = {
        "customer": frame["customer_id"].to_numpy(np.int32),
        "seed_day": frame["seed_day"].to_numpy(np.int32),
    }
    if split != "test":
        result["label"] = frame["churn"].to_numpy(np.float32)
    origin_map = {int(value.timestamp() // 86400): index for index, value in enumerate(origins)}
    if split == "val":
        result["origin"] = np.full(len(frame), len(origins), dtype=np.int16)
    elif split == "test":
        result["origin"] = np.full(len(frame), len(origins) + 1, dtype=np.int16)
    else:
        result["origin"] = np.array([origin_map[int(value)] for value in result["seed_day"]], dtype=np.int16)
    return result


def ensure_seed_arrays(store: EventStore, data: dict[str, np.ndarray], split: str, debug: bool, started: float) -> None:
    suffix = "debug" if debug else "full"
    root = cache_root() / f"seed_arrays_{split}_{suffix}_ordered_v2"
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / f"{name}.npy" for name in ("starts", "sizes", "lifetime", "query")}
    if not all(path.exists() for path in paths.values()):
        starts, sizes, lifetime, query = store.seed_arrays(data["customer"], data["seed_day"])
        values = {"starts": starts, "sizes": sizes, "lifetime": lifetime, "query": query}
        for name, value in values.items():
            temporary = paths[name].with_name(f"{name}_{os.getpid()}.npy")
            np.save(temporary, value)
            os.replace(temporary, paths[name])
        announce("seed_arrays", started, f"split={split} rows={len(starts)}")
    for name, path in paths.items():
        data[name] = np.load(path, mmap_mode="r")


def batch_seed_tensors(store: EventStore, starts: np.ndarray, sizes: np.ndarray, lifetime: np.ndarray, query: np.ndarray, selected: slice, device: torch.device):
    event = move_batch(store.event_batch(starts[selected], sizes[selected]), device)
    life = torch.from_numpy(lifetime[selected]).to(device, non_blocking=True)
    query_tensor = torch.from_numpy(query[selected]).to(device, non_blocking=True)
    return event, life, query_tensor


def checkpoint_model(checkpoint: Path, length: int, device: torch.device) -> TemporalTransformer:
    model = TemporalTransformer(length)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return model.to(device)


def train_frozen_head(
    store: EventStore,
    checkpoint: Path,
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    epochs: int,
) -> tuple[TemporalTransformer, FrozenHead]:
    device = torch.device("cuda")
    trunk = checkpoint_model(checkpoint, store.length, device)
    trunk.eval()
    for parameter in trunk.parameters():
        parameter.requires_grad_(False)
    head = FrozenHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=FINETUNE_LR, weight_decay=0.02)
    starts = data["starts"][train_indices]
    sizes = data["sizes"][train_indices]
    lifetime = data["lifetime"][train_indices]
    query = data["query"][train_indices]
    labels = data["label"][train_indices]
    origins = data["origin"][train_indices]
    latest = int(origins.max())
    weights = np.clip(np.exp(-0.025 * (latest - origins)), 0.55, 1.0).astype(np.float32)
    head.train()
    step = 0
    for epoch in range(epochs):
        for beginning in range(0, len(train_indices), BATCH_SIZE):
            selected = slice(beginning, min(len(train_indices), beginning + BATCH_SIZE))
            event, life, query_tensor = batch_seed_tensors(store, starts, sizes, lifetime, query, selected, device)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                state = trunk.last_event_state(event, life)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = head(state, life, query_tensor)
                target = torch.from_numpy(labels[selected]).to(device)
                weight = torch.from_numpy(weights[selected]).to(device)
                loss = (functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            if step % 250 == 0:
                print(f"[lane2] frozen_head epoch={epoch + 1} step={step} loss={float(loss.detach()):.5f}", flush=True)
    return trunk, head


def train_full_model(
    store: EventStore,
    checkpoint: Path,
    data: dict[str, np.ndarray],
    train_indices: np.ndarray,
    epochs: int,
) -> TemporalTransformer:
    device = torch.device("cuda")
    model = checkpoint_model(checkpoint, store.length, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=0.02)
    starts = data["starts"][train_indices]
    sizes = data["sizes"][train_indices]
    lifetime = data["lifetime"][train_indices]
    query = data["query"][train_indices]
    labels = data["label"][train_indices]
    origins = data["origin"][train_indices]
    latest = int(origins.max())
    weights = np.clip(np.exp(-0.025 * (latest - origins)), 0.55, 1.0).astype(np.float32)
    step = 0
    for epoch in range(epochs):
        for beginning in range(0, len(train_indices), BATCH_SIZE):
            selected = slice(beginning, min(len(train_indices), beginning + BATCH_SIZE))
            event, life, query_tensor = batch_seed_tensors(store, starts, sizes, lifetime, query, selected, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model.classification_logits(event, life, query_tensor)
                target = torch.from_numpy(labels[selected]).to(device)
                weight = torch.from_numpy(weights[selected]).to(device)
                loss = (functional.binary_cross_entropy_with_logits(logits, target, reduction="none") * weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            if step % 250 == 0:
                print(f"[lane2] full_finetune epoch={epoch + 1} step={step} loss={float(loss.detach()):.5f}", flush=True)
    return model


@torch.no_grad()
def predict_frozen(
    store: EventStore,
    trunk: TemporalTransformer,
    head: FrozenHead,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda")
    starts = data["starts"][indices]
    sizes = data["sizes"][indices]
    lifetime = data["lifetime"][indices]
    query = data["query"][indices]
    result = np.empty(len(indices), dtype=np.float32)
    trunk.eval()
    head.eval()
    for beginning in range(0, len(indices), BATCH_SIZE):
        selected = slice(beginning, min(len(indices), beginning + BATCH_SIZE))
        event, life, query_tensor = batch_seed_tensors(store, starts, sizes, lifetime, query, selected, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = trunk.last_event_state(event, life)
            result[selected] = torch.sigmoid(head(state, life, query_tensor)).float().cpu().numpy()
    return result


@torch.no_grad()
def predict_full(
    store: EventStore,
    model: TemporalTransformer,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda")
    starts = data["starts"][indices]
    sizes = data["sizes"][indices]
    lifetime = data["lifetime"][indices]
    query = data["query"][indices]
    result = np.empty(len(indices), dtype=np.float32)
    model.eval()
    for beginning in range(0, len(indices), BATCH_SIZE):
        selected = slice(beginning, min(len(indices), beginning + BATCH_SIZE))
        event, life, query_tensor = batch_seed_tensors(store, starts, sizes, lifetime, query, selected, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result[selected] = torch.sigmoid(model.classification_logits(event, life, query_tensor)).float().cpu().numpy()
    return result


@torch.no_grad()
def predict_zero_shot(
    store: EventStore,
    checkpoint: Path,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
) -> np.ndarray:
    device = torch.device("cuda")
    model = checkpoint_model(checkpoint, store.length, device)
    model.eval()
    starts = data["starts"][indices]
    sizes = data["sizes"][indices]
    lifetime = data["lifetime"][indices]
    query = data["query"][indices]
    result = np.empty(len(indices), dtype=np.float32)
    upper = np.exp(np.linspace(np.log(1.0), np.log(2920.0), GAP_BUCKETS + 1))[1:]
    for beginning in range(0, len(indices), BATCH_SIZE):
        selected = slice(beginning, min(len(indices), beginning + BATCH_SIZE))
        event, life, _ = batch_seed_tensors(store, starts, sizes, lifetime, query, selected, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = model.last_event_state(event, life)
            probability = torch.softmax(model.gap_head(state).float(), dim=1).cpu().numpy()
        recency = np.expm1(query[selected, 0])
        for row in range(len(probability)):
            survival_now = probability[row, upper > recency[row]].sum()
            survival_future = probability[row, upper > recency[row] + 91.0].sum()
            result[beginning + row] = np.clip(survival_future / max(survival_now, 1e-6), 0, 1)
    del model
    torch.cuda.empty_cache()
    return result


def empirical_survival_validation(store: EventStore, data: dict[str, np.ndarray], indices: np.ndarray, zero_shot: np.ndarray) -> dict:
    customers = data["customer"][indices]
    seed_days = data["seed_day"][indices]
    recency = np.expm1(data["query"][indices, 0])
    groups = [(0, 7), (8, 30), (31, 60), (61, 91)]
    rows = []
    labels = data["label"][indices]
    for low, high in groups:
        mask = (recency >= low) & (recency <= high)
        if mask.sum() and np.unique(labels[mask]).size == 2:
            rows.append(
                {
                    "recency": f"{low}_{high}",
                    "count": int(mask.sum()),
                    "label_rate": float(labels[mask].mean()),
                    "score_mean": float(zero_shot[mask].mean()),
                    "auc": float(roc_auc_score(labels[mask], zero_shot[mask])),
                }
            )
    return {"overall_auc": float(roc_auc_score(labels, zero_shot)), "recency_bins": rows}


def semantic_probe(data: dict[str, np.ndarray], debug: bool) -> list[dict]:
    import lightgbm as lgb

    last = np.asarray(data["starts"] + np.maximum(data["sizes"], 1) - 1, dtype=np.int64)
    result = []
    folds = (20,) if debug else FOLD_ORIGINS
    for fold in folds:
        train_indices = np.flatnonzero(data["origin"] < fold)
        hold_indices = np.flatnonzero(data["origin"] == fold)
        if debug:
            train_indices = train_indices[:30000]
            hold_indices = hold_indices[:10000]
        if len(train_indices) > 1200000:
            stride = int(math.ceil(len(train_indices) / 1200000))
            train_indices = train_indices[::stride]
        semantic_train = np.asarray(data["event_semantic"][last[train_indices]], dtype=np.float32)
        semantic_hold = np.asarray(data["event_semantic"][last[hold_indices]], dtype=np.float32)
        base_train = np.concatenate([data["lifetime"][train_indices], data["query"][train_indices]], axis=1)
        base_hold = np.concatenate([data["lifetime"][hold_indices], data["query"][hold_indices]], axis=1)
        rows = {"base": (base_train, base_hold), "semantic": (np.concatenate([base_train, semantic_train], axis=1), np.concatenate([base_hold, semantic_hold], axis=1))}
        scores = {}
        for name, (x_train, x_hold) in rows.items():
            model = lgb.LGBMClassifier(
                objective="binary",
                n_estimators=100 if not debug else 20,
                learning_rate=0.06,
                num_leaves=31,
                min_child_samples=1000,
                reg_lambda=4.0,
                n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
                verbosity=-1,
                random_state=1337,
            )
            model.fit(x_train, data["label"][train_indices], callbacks=[lgb.log_evaluation(0)])
            scores[name] = float(roc_auc_score(data["label"][hold_indices], model.predict_proba(x_hold)[:, 1]))
        result.append({"fold": fold, "train_n": int(len(train_indices)), "hold_n": int(len(hold_indices)), "base_auc": scores["base"], "semantic_auc": scores["semantic"], "effect": scores["semantic"] - scores["base"]})
        print(f"[lane2] semantic_probe_fold={fold} base_auc={scores['base']:.6f} semantic_auc={scores['semantic']:.6f} effect={scores['semantic'] - scores['base']:+.6f}", flush=True)
    return result


def rank_values(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ((ranks + 0.5) / len(values)).astype(np.float32)


def bootstrap_support(labels: np.ndarray, challenger: np.ndarray, baseline: np.ndarray, draws: int = 100) -> tuple[float, float]:
    rng = np.random.default_rng(1337)
    differences = []
    for _ in range(draws):
        selected = rng.integers(0, len(labels), len(labels))
        sample_y = labels[selected]
        if np.unique(sample_y).size < 2:
            continue
        differences.append(roc_auc_score(sample_y, challenger[selected]) - roc_auc_score(sample_y, baseline[selected]))
    return float(np.mean(np.asarray(differences) > 0)), float(np.std(differences, ddof=1))


def transformer_folds(
    store: EventStore,
    checkpoints: dict[str, Path],
    train: dict[str, np.ndarray],
    debug: bool,
    started: float,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    predictions = {
        "zero": np.full(len(train["label"]), np.nan, dtype=np.float32),
        "frozen": np.full(len(train["label"]), np.nan, dtype=np.float32),
        "full": np.full(len(train["label"]), np.nan, dtype=np.float32),
    }
    rows = []
    folds = (20,) if debug else FOLD_ORIGINS
    for fold in folds:
        fold_path = cache_root() / f"fold_{fold}_{'debug_v3' if debug else 'full_ordered_v2'}.npz"
        hold_indices = np.flatnonzero(train["origin"] == fold)
        train_indices = np.flatnonzero(train["origin"] < fold)
        if debug:
            visible = store.pointer[train["customer"] + 1] > store.pointer[train["customer"]]
            train_indices = train_indices[visible[train_indices]][:30000]
            hold_indices = hold_indices[visible[hold_indices]][:10000]
        if fold_path.exists():
            cached = np.load(fold_path, allow_pickle=False)
            zero = cached["zero"]
            frozen = cached["frozen"]
            full = cached["full"]
        else:
            checkpoint = checkpoints[f"origin{fold}"]
            zero = predict_zero_shot(store, checkpoint, train, hold_indices)
            frozen_trunk, frozen_head = train_frozen_head(store, checkpoint, train, train_indices, 2)
            frozen = predict_frozen(store, frozen_trunk, frozen_head, train, hold_indices)
            del frozen_trunk, frozen_head
            torch.cuda.empty_cache()
            full_model = train_full_model(store, checkpoint, train, train_indices, 2)
            full = predict_full(store, full_model, train, hold_indices)
            del full_model
            torch.cuda.empty_cache()
            temporary = fold_path.with_name(f"fold_{fold}_{os.getpid()}.npz")
            np.savez_compressed(temporary, hold_indices=hold_indices, zero=zero, frozen=frozen, full=full)
            os.replace(temporary, fold_path)
        predictions["zero"][hold_indices] = zero
        predictions["frozen"][hold_indices] = frozen
        predictions["full"][hold_indices] = full
        scores = {name: float(roc_auc_score(train["label"][hold_indices], value)) for name, value in (("zero", zero), ("frozen", frozen), ("full", full))}
        support, standard_error = bootstrap_support(train["label"][hold_indices], full, frozen, draws=100)
        survival = empirical_survival_validation(store, train, hold_indices, zero)
        row = {"fold": fold, "train_n": int(len(train_indices)), "hold_n": int(len(hold_indices)), **scores, "full_over_frozen_support": support, "paired_bootstrap_se": standard_error, "zero_shot_survival": survival}
        rows.append(row)
        announce("transformer_fold", started, f"fold={fold} zero={scores['zero']:.6f} frozen={scores['frozen']:.6f} full={scores['full']:.6f} support={support:.3f}")
    return predictions, rows


def choose_head(fold_rows: list[dict]) -> str:
    frozen = np.array([row["frozen"] for row in fold_rows])
    full = np.array([row["full"] for row in fold_rows])
    support = np.array([row["full_over_frozen_support"] for row in fold_rows])
    if np.median(full - frozen) > 0 and np.median(support) >= 0.8:
        return "full"
    return "frozen"


def select_blend(
    labels: np.ndarray,
    origins: np.ndarray,
    transformer: np.ndarray,
    champion: np.ndarray,
    champion_mask: np.ndarray,
    debug: bool,
) -> tuple[float, list[dict], dict]:
    valid = np.isfinite(transformer) & np.isfinite(champion) & champion_mask
    results = []
    folds = (20,) if debug else FOLD_ORIGINS
    for weight in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
        fold_scores = []
        for fold in folds:
            selected = valid & (origins == fold)
            if not np.any(selected):
                continue
            prediction = (1.0 - weight) * rank_values(champion[selected]) + weight * rank_values(transformer[selected])
            fold_scores.append(float(roc_auc_score(labels[selected], prediction)))
        if fold_scores:
            results.append({"weight": weight, "fold_auc": fold_scores, "median_auc": float(np.median(fold_scores)), "worst_auc": float(np.min(fold_scores))})
    results.sort(key=lambda row: (row["median_auc"], row["worst_auc"]), reverse=True)
    baseline = next(row for row in results if row["weight"] == 0.0)
    proposed = results[0]
    admitted_row = None
    admitted_supports = []
    admitted_errors = []
    tested = []
    for candidate in results:
        if candidate["weight"] == 0 or candidate["median_auc"] <= baseline["median_auc"]:
            continue
        supports = []
        errors = []
        for fold in folds:
            selected = valid & (origins == fold)
            if not np.any(selected):
                continue
            base = rank_values(champion[selected])
            challenger = (1.0 - candidate["weight"]) * base + candidate["weight"] * rank_values(transformer[selected])
            support, error = bootstrap_support(labels[selected], challenger, base, draws=100)
            supports.append(support)
            errors.append(error)
        tested.append({"weight": candidate["weight"], "median_support": float(np.median(supports)), "median_paired_se": float(np.median(errors))})
        if np.median(supports) >= 0.8:
            admitted_row = candidate
            admitted_supports = supports
            admitted_errors = errors
            break
    admitted = admitted_row is not None
    weight = float(admitted_row["weight"] if admitted else 0.0)
    gate = {
        "proposed_weight": proposed["weight"],
        "admitted": admitted,
        "admitted_weight": weight,
        "median_support": float(np.median(admitted_supports)) if admitted_supports else 0.0,
        "median_paired_se": float(np.median(admitted_errors)) if admitted_errors else 0.0,
        "baseline_median": baseline["median_auc"],
        "proposed_median": proposed["median_auc"],
        "admitted_median": admitted_row["median_auc"] if admitted else baseline["median_auc"],
        "support_tests": tested,
    }
    return weight, results, gate


def slice_scores(store: EventStore, data: dict[str, np.ndarray], prediction: np.ndarray, indices: np.ndarray) -> list[dict]:
    query = data["query"][indices]
    sizes = data["sizes"][indices]
    labels = data["label"][indices]
    recency = np.expm1(query[:, 0])
    activity = np.expm1(query[:, 3])
    last = np.asarray(data["starts"][indices] + np.maximum(sizes, 1) - 1, dtype=np.int64)
    rarity = store.popularity[last]
    groups = {
        "activity_1": activity <= 1.5,
        "activity_2_3": (activity > 1.5) & (activity < 3.5),
        "activity_4_plus": activity >= 3.5,
        "sequence_truncated": sizes >= store.length,
        "sequence_not_truncated": sizes < store.length,
        "recency_0_7": recency <= 7,
        "recency_8_30": (recency > 7) & (recency <= 30),
        "recency_31_91": recency > 30,
        "product_rare": rarity <= 3,
        "product_common": rarity >= 8,
    }
    rows = []
    for name, mask in groups.items():
        if mask.sum() and np.unique(labels[mask]).size == 2:
            rows.append({"stratum": name, "count": int(mask.sum()), "label_rate": float(labels[mask].mean()), "auc": float(roc_auc_score(labels[mask], prediction[mask]))})
    return rows


def combine_labelled(train: dict[str, np.ndarray], val: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ("customer", "seed_day", "label", "origin", "starts", "sizes", "lifetime", "query")
    return {name: np.concatenate([np.asarray(train[name]), np.asarray(val[name])], axis=0) for name in names}


def final_transformer_predictions(
    store: EventStore,
    checkpoints: dict[str, Path],
    train: dict[str, np.ndarray],
    val: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    head_kind: str,
    started: float,
) -> tuple[np.ndarray, np.ndarray]:
    path = cache_root() / f"final_transformer_{head_kind}_ordered_v2.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        announce("final_transformer_cache", started, f"head={head_kind}")
        return cached["val"], cached["test"]
    train_indices = np.arange(len(train["label"]), dtype=np.int64)
    val_indices = np.arange(len(val["label"]), dtype=np.int64)
    if head_kind == "full":
        model_a = train_full_model(store, checkpoints["validation"], train, train_indices, 2)
        val_prediction = predict_full(store, model_a, val, val_indices).copy()
        del model_a
    else:
        trunk_a, head_a = train_frozen_head(store, checkpoints["validation"], train, train_indices, 2)
        val_prediction = predict_frozen(store, trunk_a, head_a, val, val_indices).copy()
        del trunk_a, head_a
    torch.cuda.empty_cache()
    announce("model_a_preserved", started, f"rows={len(val_prediction)} source=train_labels_only")
    combined = combine_labelled(train, val)
    combined_indices = np.arange(len(combined["label"]), dtype=np.int64)
    test_indices = np.arange(len(test["customer"]), dtype=np.int64)
    if head_kind == "full":
        model_b = train_full_model(store, checkpoints["test"], combined, combined_indices, 2)
        test_prediction = predict_full(store, model_b, test, test_indices)
        del model_b
    else:
        trunk_b, head_b = train_frozen_head(store, checkpoints["test"], combined, combined_indices, 2)
        test_prediction = predict_frozen(store, trunk_b, head_b, test, test_indices)
        del trunk_b, head_b
    torch.cuda.empty_cache()
    temporary = path.with_name(f"final_transformer_{head_kind}_{os.getpid()}.npz")
    np.savez_compressed(temporary, val=val_prediction, test=test_prediction)
    os.replace(temporary, path)
    register_artifact("lane2 final transformer predictions", path, "Model A validation and Model B test transformer predictions", VERSION + f"_final_{head_kind}")
    announce("model_b_complete", started, f"rows={len(test_prediction)} source=train_plus_validation_labels")
    return val_prediction, test_prediction


def validation_resolution(labels: np.ndarray, candidate: np.ndarray, champion: np.ndarray, transformer: np.ndarray) -> dict:
    rng = np.random.default_rng(7331)
    scores = []
    differences = []
    for _ in range(100):
        selected = rng.integers(0, len(labels), len(labels))
        y = labels[selected]
        candidate_auc = roc_auc_score(y, candidate[selected])
        scores.append(candidate_auc)
        differences.append(candidate_auc - roc_auc_score(y, champion[selected]))
    rank_candidate = rank_values(candidate)
    rank_champion = rank_values(champion)
    rank_transformer = rank_values(transformer)
    return {
        "candidate_auc": float(roc_auc_score(labels, candidate)),
        "champion_auc": float(roc_auc_score(labels, champion)),
        "transformer_auc_diagnostic_only": float(roc_auc_score(labels, transformer)),
        "bootstrap_standard_error": float(np.std(scores, ddof=1)),
        "paired_champion_difference_mean": float(np.mean(differences)),
        "candidate_champion_rank_correlation": float(np.corrcoef(rank_candidate, rank_champion)[0, 1]),
        "transformer_champion_rank_correlation": float(np.corrcoef(rank_transformer, rank_champion)[0, 1]),
        "bootstrap_draws": 100,
        "selection_use": "diagnostic_after_internal-fold selection only",
    }


def debug_subset(store: EventStore, train: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    visible = store.pointer[train["customer"] + 1] > store.pointer[train["customer"]]
    earlier = np.flatnonzero(visible & (train["origin"] < 20))[:30000]
    hold = np.flatnonzero(visible & (train["origin"] == 20))[:10000]
    selected = np.concatenate([earlier, hold])
    return {name: value[selected] for name, value in train.items() if name in {"customer", "seed_day", "label", "origin"}}


def run_transformer(debug: bool, champion: dict, started: float) -> tuple[np.ndarray, np.ndarray, dict]:
    origins = load_origins()
    earliest = origins[20]
    vectorizer, distiller, semantic_metadata = ensure_distiller(earliest, debug, started)
    event_path = ensure_event_days(debug, started)
    semantic_path = ensure_event_semantics(event_path, vectorizer, distiller, debug, started)
    store = EventStore(event_path, semantic_path, debug, started)
    train = load_split("train", origins)
    if debug:
        train = debug_subset(store, train)
        starts, sizes, lifetime, query = store.seed_arrays(train["customer"], train["seed_day"])
        train.update({"starts": starts, "sizes": sizes, "lifetime": lifetime, "query": query})
    else:
        ensure_seed_arrays(store, train, "train", debug, started)
    train["event_semantic"] = store.semantic
    checkpoints, pretraining = ensure_pretrained(store, origins, debug, started)
    semantic_effect = semantic_probe(train, debug)
    fold_predictions, fold_rows = transformer_folds(store, checkpoints, train, debug, started)
    head_kind = choose_head(fold_rows)
    transformer_oof = fold_predictions[head_kind]
    if debug:
        val_prediction = np.full(409792, 0.624, dtype=np.float32)
        test_prediction = np.full(351885, 0.624, dtype=np.float32)
        diagnostics = {
            "debug": True,
            "semantic": semantic_metadata,
            "semantic_effect": semantic_effect,
            "pretraining": pretraining,
            "transformer_folds": fold_rows,
            "selected_head": head_kind,
            "model_a_validation_source": "full-shape debug fallback",
            "model_b_test_source": "full-shape debug fallback",
        }
        return val_prediction, test_prediction, diagnostics
    champion_oof = champion["oof"]
    champion_mask = champion["oof_mask"].astype(bool)
    weight, blend_results, blend_gate = select_blend(
        train["label"],
        train["origin"],
        transformer_oof,
        champion_oof,
        champion_mask,
        debug,
    )
    announce("blend_gate", started, f"head={head_kind} transformer_weight={weight:.1f} support={blend_gate['median_support']:.3f}")
    val = load_split("val", origins)
    test = load_split("test", origins)
    ensure_seed_arrays(store, val, "val", debug, started)
    ensure_seed_arrays(store, test, "test", debug, started)
    transformer_val, transformer_test = final_transformer_predictions(store, checkpoints, train, val, test, head_kind, started)
    if weight > 0:
        val_prediction = ((1.0 - weight) * rank_values(champion["val"]) + weight * rank_values(transformer_val)).astype(np.float32)
        test_prediction = ((1.0 - weight) * rank_values(champion["test"]) + weight * rank_values(transformer_test)).astype(np.float32)
    else:
        val_prediction = np.asarray(champion["val"], dtype=np.float32).copy()
        test_prediction = np.asarray(champion["test"], dtype=np.float32).copy()
    mask = np.isfinite(transformer_oof) & champion_mask
    slice_indices = np.flatnonzero(mask)
    diagnostics = {
        "debug": False,
        "semantic": semantic_metadata,
        "semantic_effect": semantic_effect,
        "pretraining": pretraining,
        "transformer_folds": fold_rows,
        "selected_head": head_kind,
        "blend_weight": weight,
        "blend_results": blend_results,
        "blend_gate": blend_gate,
        "oof_slices": slice_scores(store, train, transformer_oof[slice_indices], slice_indices),
        "champion_metadata": champion["metadata"].tolist(),
        "model_a_validation_source": "validation checkpoint plus train labels only",
        "model_b_test_source": "test checkpoint plus train and validation labels",
        "validation_resolution": validation_resolution(val["label"], val_prediction, champion["val"], transformer_val),
    }
    return val_prediction, test_prediction, diagnostics
