from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import duckdb
import numpy as np


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_VERSION = "sentence-transformers-bge-small-en-v1.5-max256-v1"
DIMENSION = 384


def paths() -> tuple[Path, Path, Path]:
    source = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-amazon/db/product.parquet"
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    shared.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(f"{source.stat().st_size}:{source.stat().st_mtime_ns}:{MODEL_VERSION}".encode()).hexdigest()[:16]
    return source, shared / f"lane0_catalog_bge_{signature}.npy", shared / f"lane0_catalog_bge_{signature}.json"


def product_text(source: Path) -> list[str]:
    con = duckdb.connect()
    rows = con.execute(
        "SELECT product_id, title, brand, CASE WHEN category IS NULL OR len(category)=0 THEN '' ELSE category[len(category)] END, left(coalesce(description, ''), 1800) FROM read_parquet(?) ORDER BY product_id",
        [str(source)],
    ).fetchall()
    if len(rows) != 506012 or any(row[0] != i for i, row in enumerate(rows)):
        raise RuntimeError("product identifiers are not dense and ordered")
    return [" | ".join(str(value or "") for value in row[1:]) for row in rows]


def encode_bge(texts: list[str], target: Path) -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    model.max_seq_length = 256
    matrix = np.lib.format.open_memmap(str(target) + ".tmp", mode="w+", dtype=np.float16, shape=(len(texts), DIMENSION))
    chunk = 32768
    for start in range(0, len(texts), chunk):
        stop = min(start + chunk, len(texts))
        values = model.encode(
            texts[start:stop],
            batch_size=512,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=model.device,
        )
        matrix[start:stop] = values.astype(np.float16)
        matrix.flush()
        print(f"[embedding] encoded={stop}/{len(texts)} elapsed={time.time() - START:.1f}s", flush=True)
    del matrix
    Path(str(target) + ".tmp").replace(target)


def encode_fallback(texts: list[str], target: Path) -> None:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

    word = HashingVectorizer(n_features=192, ngram_range=(1, 2), alternate_sign=False, norm=None, dtype=np.float32)
    char = HashingVectorizer(analyzer="char_wb", n_features=192, ngram_range=(3, 5), alternate_sign=False, norm=None, dtype=np.float32)
    counts = hstack([word.transform(texts), char.transform(texts)], format="csr")
    values = TfidfTransformer(norm="l2").fit_transform(counts).toarray().astype(np.float16)
    np.save(str(target) + ".tmp", values)
    Path(str(target) + ".tmp.npy").replace(target)


def main() -> None:
    source, target, metadata = paths()
    if target.exists() and metadata.exists():
        print(f"[embedding] cache_hit={target} elapsed={time.time() - START:.1f}s")
        return
    texts = product_text(source)
    print(f"[embedding] catalog_loaded={len(texts)} elapsed={time.time() - START:.1f}s", flush=True)
    method = "bge"
    error = ""
    try:
        encode_bge(texts, target)
    except Exception as exc:
        method = "hashed_word_character_tfidf"
        error = f"{type(exc).__name__}: {exc}"
        print(f"[embedding] bge_failed={error}", flush=True)
        encode_fallback(texts, target)
    metadata.write_text(json.dumps({"model": MODEL_NAME, "version": MODEL_VERSION, "method": method, "shape": [len(texts), DIMENSION], "dtype": "float16", "normalized": True, "fallback_reason": error}, indent=2))
    print(f"[embedding] complete method={method} path={target} elapsed={time.time() - START:.1f}s", flush=True)


START = time.time()


if __name__ == "__main__":
    main()
