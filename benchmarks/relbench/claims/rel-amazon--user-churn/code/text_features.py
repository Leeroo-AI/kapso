from __future__ import annotations

import fcntl
import gc
import hashlib
import json
import os
import re
import time
from pathlib import Path

import duckdb
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, logging as transformers_logging

from sequence_data import atomic_json, atomic_save, cache_root, data_root


MODERNBERT_MODEL = "answerdotai/ModernBERT-base"
MODERNBERT_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
QWEN_MODEL = "Qwen/Qwen3.5-0.8B"
TEXT_VERSION = "modernbert_8949b9_random_projection_v3_tokens96"
QWEN_VERSION = "qwen35_08b_review_attributes_v4"


def configure_transformers() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()


def projection(hidden: int, output: int, seed: int) -> torch.Tensor:
    generator = np.random.default_rng(seed)
    matrix = generator.standard_normal((hidden, output), dtype=np.float32)
    matrix /= np.sqrt(np.square(matrix).sum(axis=0, keepdims=True) + 1e-12)
    return torch.from_numpy(matrix).cuda()


def array_content_hash(path: Path) -> str:
    array = np.load(path, mmap_mode="r")
    digest = hashlib.sha256()
    for start in range(0, len(array), 1_000_000):
        digest.update(np.asarray(array[start : start + 1_000_000]).tobytes())
    return digest.hexdigest()


def register_artifact(name: str, path: Path, description: str, content_key: str, rebuild_hint: str) -> None:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.json.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            entries = []
        relative = str(path.relative_to(shared))
        record = {
            "name": name,
            "path": relative,
            "description": description,
            "content_key": content_key,
            "rebuild_hint": rebuild_hint,
        }
        entries = [item for item in entries if item.get("name") != name]
        entries.append(record)
        temporary = registry.with_suffix(f".json.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(entries, indent=2, sort_keys=True))
        os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_modernbert() -> tuple[AutoTokenizer, AutoModel]:
    configure_transformers()
    tokenizer = AutoTokenizer.from_pretrained(MODERNBERT_MODEL, revision=MODERNBERT_REVISION, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODERNBERT_MODEL,
        revision=MODERNBERT_REVISION,
        local_files_only=True,
        dtype=torch.bfloat16,
    ).eval().cuda()
    return tokenizer, model


def encode_text_batch(
    tokenizer: AutoTokenizer,
    model: AutoModel,
    first: list[str],
    second: list[str],
    matrix: torch.Tensor,
    max_length: int,
) -> np.ndarray:
    encoded = tokenizer(
        first,
        second,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded = {name: value.cuda(non_blocking=True) for name, value in encoded.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        hidden = model(**encoded).last_hidden_state[:, 0].float()
        hidden = torch.nn.functional.normalize(hidden, dim=1)
        reduced = torch.nn.functional.normalize(hidden @ matrix, dim=1)
    return reduced.cpu().numpy().astype(np.float16)


def prepare_review_embeddings() -> tuple[Path, Path]:
    root = cache_root() / TEXT_VERSION
    root.mkdir(parents=True, exist_ok=True)
    embedding_path = root / "review_embeddings.npy"
    hashes_path = root / "review_doc_hashes.npy"
    metadata_path = root / "review_metadata.json"
    if embedding_path.exists() and hashes_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        print(f"[text] reused {metadata['documents']:,} review embeddings at {metadata['documents_per_second']:.1f} docs/s", flush=True)
        return embedding_path, hashes_path
    connection = duckdb.connect()
    connection.execute(f"SET threads={int(os.environ.get('OMP_NUM_THREADS', '1'))}")
    connection.execute("SET preserve_insertion_order=false")
    review_path = data_root() / "db" / "review.parquet"
    count = int(
        connection.execute(
            f"SELECT count(DISTINCT hash(coalesce(summary, ''), coalesce(review_text, ''))) FROM read_parquet('{review_path}')"
        ).fetchone()[0]
    )
    query = f"""
        SELECT
            hash(coalesce(summary, ''), coalesce(review_text, ''))::UBIGINT AS doc_hash,
            any_value(coalesce(summary, '')) AS summary,
            any_value(coalesce(review_text, '')) AS body
        FROM read_parquet('{review_path}')
        GROUP BY doc_hash
        ORDER BY doc_hash
    """
    temporary_embedding = root / f"review_embeddings.{os.getpid()}.tmp.npy"
    temporary_hashes = root / f"review_doc_hashes.{os.getpid()}.tmp.npy"
    embeddings = np.lib.format.open_memmap(temporary_embedding, mode="w+", dtype=np.float16, shape=(count, 64))
    hashes = np.lib.format.open_memmap(temporary_hashes, mode="w+", dtype=np.uint64, shape=(count,))
    tokenizer, model = load_modernbert()
    matrix = projection(model.config.hidden_size, 64, 6401337)
    reader = connection.execute(query).fetch_record_batch(8192)
    cursor = 0
    checkpoint_time = time.time()
    encoding_seconds = 0.0
    confirmed = False
    for record in reader:
        record_hashes = record.column(0).to_numpy(zero_copy_only=False).astype(np.uint64)
        summaries = record.column(1).to_pylist()
        bodies = record.column(2).to_pylist()
        for start in range(0, len(summaries), 1024):
            end = min(start + 1024, len(summaries))
            before = time.time()
            output = encode_text_batch(tokenizer, model, summaries[start:end], bodies[start:end], matrix, 96)
            encoding_seconds += time.time() - before
            size = end - start
            embeddings[cursor : cursor + size] = output
            hashes[cursor : cursor + size] = record_hashes[start:end]
            cursor += size
            if not confirmed and cursor >= 1_000_000:
                rate = cursor / max(encoding_seconds, 1e-9)
                projected_minutes = count / rate / 60.0
                print(
                    f"[text] one-million probe {rate:.1f} docs/s projected={projected_minutes:.1f}min token_budget=96",
                    flush=True,
                )
                if rate < 1_200:
                    raise RuntimeError(f"ModernBERT throughput {rate:.1f} docs/s is below the safe fallback rate")
                confirmed = True
        if cursor % 1_048_576 < len(summaries):
            embeddings.flush()
            hashes.flush()
            print(f"[text] review cache {cursor:,}/{count:,} elapsed={time.time() - checkpoint_time:.1f}s", flush=True)
    if cursor != count:
        raise RuntimeError(f"review embedding count {cursor} != {count}")
    embeddings.flush()
    hashes.flush()
    del embeddings, hashes
    os.replace(temporary_embedding, embedding_path)
    os.replace(temporary_hashes, hashes_path)
    rate = count / max(encoding_seconds, 1e-9)
    atomic_json(
        metadata_path,
        {
            "documents": count,
            "documents_per_second": rate,
            "model": MODERNBERT_MODEL,
            "revision": MODERNBERT_REVISION,
            "dimensions": 64,
            "token_budget": 96,
            "content_hash": array_content_hash(hashes_path),
        },
    )
    register_artifact(
        "lane2 ModernBERT review embeddings",
        root,
        "FP16 64-dimensional embeddings and sorted content hashes for distinct review documents",
        TEXT_VERSION,
        "Run python main.py --prepare-cache with the sanitized rel-amazon cache and ModernBERT revision 8949b9",
    )
    print(f"[text] encoded {count:,} distinct reviews at {rate:.1f} docs/s", flush=True)
    del model, tokenizer, matrix
    gc.collect()
    torch.cuda.empty_cache()
    return embedding_path, hashes_path


def prepare_product_embeddings() -> Path:
    root = cache_root() / TEXT_VERSION
    root.mkdir(parents=True, exist_ok=True)
    embedding_path = root / "product_embeddings.npy"
    metadata_path = root / "product_metadata.json"
    if embedding_path.exists() and metadata_path.exists():
        return embedding_path
    product_path = data_root() / "db" / "product.parquet"
    connection = duckdb.connect()
    query = f"""
        SELECT product_id, coalesce(title, '') AS title, coalesce(description, '') AS description
        FROM read_parquet('{product_path}') ORDER BY product_id
    """
    count = int(connection.execute(f"SELECT count(*) FROM read_parquet('{product_path}')").fetchone()[0])
    reader = connection.execute(query).fetch_record_batch(8192)
    temporary = root / f"product_embeddings.{os.getpid()}.tmp.npy"
    embeddings = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=(count, 32))
    tokenizer, model = load_modernbert()
    matrix = projection(model.config.hidden_size, 32, 3201337)
    cursor = 0
    started = time.time()
    for record in reader:
        titles = record.column(1).to_pylist()
        descriptions = record.column(2).to_pylist()
        for start in range(0, len(titles), 1024):
            end = min(start + 1024, len(titles))
            output = encode_text_batch(tokenizer, model, titles[start:end], descriptions[start:end], matrix, 128)
            embeddings[cursor : cursor + len(output)] = output
            cursor += len(output)
    if cursor != count:
        raise RuntimeError(f"product embedding count {cursor} != {count}")
    embeddings.flush()
    del embeddings
    os.replace(temporary, embedding_path)
    atomic_json(metadata_path, {"products": count, "seconds": time.time() - started, "dimensions": 32})
    print(f"[text] encoded {count:,} products in {time.time() - started:.1f}s", flush=True)
    del model, tokenizer, matrix
    gc.collect()
    torch.cuda.empty_cache()
    return embedding_path


def prepare_event_doc_index(index: dict[str, np.ndarray], hashes_path: Path) -> Path:
    path = cache_root() / TEXT_VERSION / "event_doc_index.npy"
    metadata = path.with_suffix(".json")
    if path.exists() and metadata.exists():
        return path
    hashes = np.load(hashes_path, mmap_mode="r")
    event_hashes = index["doc_hash"]
    temporary = path.with_name(f"event_doc_index.{os.getpid()}.tmp.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(len(event_hashes),))
    for start in range(0, len(event_hashes), 1_000_000):
        end = min(start + 1_000_000, len(event_hashes))
        positions = np.searchsorted(hashes, event_hashes[start:end]).astype(np.int32)
        if np.any(hashes[positions] != event_hashes[start:end]):
            raise RuntimeError("review content hash lookup failed")
        output[start:end] = positions
    output.flush()
    del output
    os.replace(temporary, path)
    atomic_json(metadata, {"events": int(len(event_hashes)), "documents": int(len(hashes))})
    return path


def prepare_all_text(index: dict[str, np.ndarray]) -> dict[str, Path]:
    review_path, hashes_path = prepare_review_embeddings()
    product_path = prepare_product_embeddings()
    event_doc_path = prepare_event_doc_index(index, hashes_path)
    return {
        "review": review_path,
        "hashes": hashes_path,
        "product": product_path,
        "event_doc": event_doc_path,
    }


def prepare_debug_text(index: dict[str, np.ndarray]) -> dict[str, Path]:
    root = cache_root() / "debug_text_v1"
    root.mkdir(parents=True, exist_ok=True)
    review = root / "review_embeddings.npy"
    product = root / "product_embeddings.npy"
    event_doc = root / "event_doc_index.npy"
    hashes = root / "review_doc_hashes.npy"
    if not review.exists():
        atomic_save(review, np.zeros((1, 64), dtype=np.float16))
    if not product.exists():
        atomic_save(product, np.zeros((len(index["product_price"]), 32), dtype=np.float16))
    if not event_doc.exists():
        atomic_save(event_doc, np.zeros(len(index["doc_hash"]), dtype=np.int32))
    if not hashes.exists():
        atomic_save(hashes, np.zeros(1, dtype=np.uint64))
    return {"review": review, "product": product, "event_doc": event_doc, "hashes": hashes}


def qwen_panel_rows() -> dict[str, np.ndarray | list[str]]:
    train_path = data_root() / "tasks" / os.environ["RELBENCH_TASK"] / "train.parquet"
    review_path = data_root() / "db" / "review.parquet"
    connection = duckdb.connect()
    query = f"""
        WITH latest_origins AS (
            SELECT DISTINCT timestamp FROM read_parquet('{train_path}') ORDER BY timestamp DESC LIMIT 2
        ), ranked_seeds AS (
            SELECT t.*, row_number() OVER (PARTITION BY timestamp ORDER BY hash(customer_id, timestamp, 1337)) AS seed_rank
            FROM read_parquet('{train_path}') t
            WHERE timestamp IN (SELECT timestamp FROM latest_origins)
        ), seeds AS (
            SELECT * FROM ranked_seeds WHERE seed_rank <= 12500
        ), joined AS (
            SELECT
                s.timestamp,
                s.customer_id,
                s.churn,
                r.rating,
                r.verified,
                coalesce(r.summary, '') AS summary,
                coalesce(r.review_text, '') AS body,
                hash(coalesce(r.summary, ''), coalesce(r.review_text, ''))::UBIGINT AS doc_hash,
                row_number() OVER (
                    PARTITION BY s.timestamp, s.customer_id
                    ORDER BY r.review_time DESC, r.product_id DESC, doc_hash DESC
                ) AS event_rank
            FROM seeds s
            JOIN read_parquet('{review_path}') r
              ON r.customer_id = s.customer_id AND r.review_time <= s.timestamp
        )
        SELECT
            date_diff('day', TIMESTAMP '1970-01-01', timestamp)::INTEGER AS origin,
            customer_id::INTEGER AS customer,
            churn::INTEGER AS target,
            rating::FLOAT AS rating,
            verified::UTINYINT AS verified,
            length(summary)::INTEGER AS summary_length,
            length(body)::INTEGER AS body_length,
            doc_hash,
            summary,
            body
        FROM joined
        WHERE event_rank = 1
        ORDER BY origin, hash(customer_id, origin, 1337)
    """
    result = connection.execute(query).fetchnumpy()
    return {
        "origin": result["origin"].astype(np.int32),
        "customer": result["customer"].astype(np.int32),
        "target": result["target"].astype(np.int64),
        "rating": result["rating"].astype(np.float32),
        "verified": result["verified"].astype(np.float32),
        "summary_length": result["summary_length"].astype(np.float32),
        "body_length": result["body_length"].astype(np.float32),
        "doc_hash": result["doc_hash"].astype(np.uint64),
        "summary": result["summary"].tolist(),
        "body": result["body"].tolist(),
    }


def parse_integer_attributes(text: str) -> np.ndarray | None:
    names = ["sentiment", "specificity", "engagement", "intensity"]
    values = []
    for name in names:
        match = re.search(rf'"{name}"\s*:\s*([0-4])', text)
        if match is None:
            return None
        values.append(int(match.group(1)))
    return np.asarray(values, dtype=np.int8)


def parse_attributes(text: str) -> np.ndarray | None:
    integer = parse_integer_attributes(text)
    if integer is not None:
        return integer
    names = ["sentiment", "specificity", "engagement", "intensity"]
    values = []
    for name in names:
        match = re.search(rf'"{name}"\s*:\s*([0-4])', text)
        if match is not None:
            values.append(int(match.group(1)))
            continue
        word = re.search(rf'"{name}"\s*:\s*"(very negative|negative|mixed|neutral|positive|very positive|low|medium|high)"', text.lower())
        if word is None:
            return None
        mapping = {"very negative": 0, "negative": 1, "mixed": 2, "neutral": 2, "low": 1, "medium": 2, "positive": 3, "high": 3, "very positive": 4}
        values.append(mapping[word.group(1)])
    return np.asarray(values, dtype=np.int8)


def qwen_prompts(summaries: list[str], bodies: list[str]) -> list[str]:
    prefix = 'Read the book review. Return only {"sentiment":N,"specificity":N,"engagement":N,"intensity":N}. Every N must be one unquoted integer 0, 1, 2, 3, or 4. Sentiment runs very negative to very positive; the other scales run low to high. Do not use words or explanation. '
    return [prefix + "Summary: " + summary[:240] + " Review: " + body[:720] for summary, body in zip(summaries, bodies)]


def generate_qwen_attributes(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, summaries: list[str], bodies: list[str]) -> tuple[np.ndarray, int]:
    attributes = np.empty((len(summaries), 4), dtype=np.int8)
    parsed = 0
    prompts = qwen_prompts(summaries, bodies)
    tokenizer.padding_side = "left"
    for start in range(0, len(prompts), 128):
        batch = prompts[start : start + 128]
        chats = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for prompt in batch]
        encoded = tokenizer(chats, padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses = tokenizer.batch_decode(output[:, encoded.input_ids.shape[1] :], skip_special_tokens=True)
        for local, response in enumerate(responses):
            strict = parse_integer_attributes(response)
            value = strict if strict is not None else parse_attributes(response)
            if value is None:
                digits = [int(item) for item in re.findall(r"(?<!\d)([0-4])(?!\d)", response)[:4]]
                value = np.asarray(digits, dtype=np.int8) if len(digits) == 4 else np.full(4, 2, dtype=np.int8)
            if strict is not None:
                parsed += 1
            attributes[start + local] = value
    return attributes, parsed


def atomic_panel(path: Path, values: dict[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def qwen_gate(panel: dict[str, np.ndarray]) -> dict:
    origins = np.unique(panel["origin"])
    train_mask = panel["origin"] == origins[0]
    fold_mask = panel["origin"] == origins[-1]
    base = np.column_stack(
        (
            panel["rating"] / 5.0,
            panel["verified"],
            np.log1p(panel["summary_length"]),
            np.log1p(panel["body_length"]),
        )
    )
    attrs = panel["attributes"].astype(np.float32) / 4.0
    target = panel["target"]
    attribute_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300))
    base_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300))
    plus_model = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300))
    attribute_model.fit(attrs[train_mask], target[train_mask])
    base_model.fit(base[train_mask], target[train_mask])
    plus_model.fit(np.column_stack((base, attrs))[train_mask], target[train_mask])
    attribute_prediction = attribute_model.predict_proba(attrs[fold_mask])[:, 1]
    base_prediction = base_model.predict_proba(base[fold_mask])[:, 1]
    plus_prediction = plus_model.predict_proba(np.column_stack((base, attrs))[fold_mask])[:, 1]
    attribute_auc = roc_auc_score(target[fold_mask], attribute_prediction)
    base_auc = roc_auc_score(target[fold_mask], base_prediction)
    plus_auc = roc_auc_score(target[fold_mask], plus_prediction)
    generator = np.random.default_rng(1337)
    differences = np.empty(100, dtype=np.float64)
    fold_target = target[fold_mask]
    for draw in range(100):
        rows = generator.integers(0, len(fold_target), len(fold_target))
        differences[draw] = roc_auc_score(fold_target[rows], plus_prediction[rows]) - roc_auc_score(fold_target[rows], base_prediction[rows])
    return {
        "attribute_auc": float(attribute_auc),
        "base_auc": float(base_auc),
        "plus_auc": float(plus_auc),
        "paired_delta": float(plus_auc - base_auc),
        "paired_se": float(differences.std(ddof=1)),
        "prediction_rank_correlation": float(np.corrcoef(np.argsort(np.argsort(base_prediction)), np.argsort(np.argsort(plus_prediction)))[0, 1]),
        "kept": bool(plus_auc > base_auc),
        "train_rows": int(train_mask.sum()),
        "fold_rows": int(fold_mask.sum()),
    }


def ensure_qwen_measurement(debug: bool) -> tuple[dict, dict[str, np.ndarray]]:
    root = cache_root() / QWEN_VERSION
    root.mkdir(parents=True, exist_ok=True)
    panel_path = root / "panel.npz"
    gate_path = root / "gate.json"
    rows = qwen_panel_rows()
    if panel_path.exists():
        stored = dict(np.load(panel_path, allow_pickle=False))
        attributes = stored["attributes"]
        if len(attributes) != len(rows["origin"]):
            raise RuntimeError("Qwen panel cache length mismatch")
    else:
        attributes = np.full((len(rows["origin"]), 4), -1, dtype=np.int8)
    missing = np.flatnonzero(attributes[:, 0] < 0)
    target_count = min(32, len(missing)) if debug else len(missing)
    scored = 0
    parsed = 0
    if target_count:
        configure_transformers()
        tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL, local_files_only=True, dtype=torch.bfloat16).eval().cuda()
        probe, probe_parsed = generate_qwen_attributes(
            model,
            tokenizer,
            ["Wonderful story"],
            ["I loved the characters and clear writing, though the ending was rushed."],
        )
        if probe_parsed != 1 or np.any(probe < 0) or np.any(probe > 4):
            raise RuntimeError("Qwen integer-schema probe failed")
        chosen = missing[:target_count]
        started = time.time()
        generated, parsed = generate_qwen_attributes(
            model,
            tokenizer,
            [rows["summary"][index] for index in chosen],
            [rows["body"][index] for index in chosen],
        )
        attributes[chosen] = generated
        scored = len(chosen)
        print(
            f"[qwen] schema_probe=passed scored={scored:,} parsed={parsed:,} seconds={time.time() - started:.1f}",
            flush=True,
        )
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    panel_values = {name: value for name, value in rows.items() if name not in {"summary", "body"}}
    panel_values["attributes"] = attributes
    atomic_panel(panel_path, panel_values)
    if np.all(attributes[:, 0] >= 0):
        gate = qwen_gate(panel_values)
        atomic_json(gate_path, gate)
        register_artifact(
            "lane2 Qwen bounded review attributes",
            root,
            "Fixed-hash 25,000-document Qwen integer attributes and forward-fold gate",
            QWEN_VERSION,
            "Run a full main.py invocation with Qwen/Qwen3.5-0.8B available",
        )
        print(
            f"[qwen] attributes_auc={gate['attribute_auc']:.6f} base_auc={gate['base_auc']:.6f} plus_auc={gate['plus_auc']:.6f} paired_delta={gate['paired_delta']:+.6f} se={gate['paired_se']:.6f} kept={gate['kept']}",
            flush=True,
        )
    else:
        gate = {"kept": False, "scored": int(np.sum(attributes[:, 0] >= 0)), "required": 25000}
    return gate, panel_values
