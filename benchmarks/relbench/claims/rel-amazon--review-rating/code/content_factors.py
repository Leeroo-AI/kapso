import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ProductMetadata:
    matrix: np.ndarray
    category: np.ndarray
    top_category: np.ndarray
    brand: np.ndarray
    title: np.ndarray
    category_frequency: np.ndarray
    brand_frequency: np.ndarray
    title_frequency: np.ndarray
    log_price: np.ndarray
    title_length: np.ndarray
    description_length: np.ndarray
    category_depth: np.ndarray


@dataclass
class ContentFactors:
    item_bias: np.ndarray
    item_factor: np.ndarray
    implicit_factor: np.ndarray
    mapped_item_bias: np.ndarray
    mapped_item_factor: np.ndarray
    mapped_implicit_factor: np.ndarray


def register_artifact(cache_dir, name, path, description, content_key, rebuild_hint):
    cache_dir = Path(cache_dir)
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        entries = json.loads(registry.read_text()) if registry.exists() else []
        if not any(x.get("content_key") == content_key for x in entries):
            entries.append(
                {
                    "name": name,
                    "path": str(Path(path).relative_to(cache_dir)),
                    "description": description,
                    "content_key": content_key,
                    "rebuild_hint": rebuild_hint,
                }
            )
            temp = registry.with_suffix(f".{os.getpid()}.tmp")
            temp.write_text(json.dumps(entries, indent=2))
            os.replace(temp, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def encode_products(product, cache_dir, debug=False):
    n_items = len(product)
    if debug:
        return np.zeros((n_items, 32), dtype=np.float32)
    cache_dir = Path(cache_dir)
    embedding_path = cache_dir / "amazon_books_minilm_l6v2_384_f16_v1.npy"
    pca_path = cache_dir / "amazon_books_minilm_l6v2_pca32_f16_v1.npy"
    lock_path = cache_dir / "amazon_books_minilm_l6v2_v1.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        valid_embedding = embedding_path.exists() and np.load(embedding_path, mmap_mode="r").shape == (n_items, 384)
        if not valid_embedding:
            from sentence_transformers import SentenceTransformer

            started = time.time()
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda:0", local_files_only=True)
            temp = embedding_path.with_suffix(f".{os.getpid()}.tmp.npy")
            output = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float16, shape=(n_items, 384))
            titles = product["title"].fillna("").astype(str).to_numpy()
            descriptions = product["description"].fillna("").astype(str).to_numpy()
            brands = product["brand"].fillna("").astype(str).to_numpy()
            batch_size = 2048
            for begin in range(0, n_items, batch_size):
                end = min(begin + batch_size, n_items)
                texts = [
                    f"{titles[j][:256]} [SEP] {brands[j][:96]} [SEP] {descriptions[j][:512]}"
                    for j in range(begin, end)
                ]
                output[begin:end] = model.encode(
                    texts,
                    batch_size=512,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float16)
            output.flush()
            del output, model
            os.replace(temp, embedding_path)
            print(f"[content] minilm rows={n_items} elapsed={time.time() - started:.1f}s", flush=True)
            register_artifact(
                cache_dir,
                "Amazon Books MiniLM product embeddings",
                embedding_path,
                "Normalized all-MiniLM-L6-v2 embeddings of title, brand, and description",
                "rel-amazon-books-product-minilm-l6v2-384-f16-v1",
                "Load the sanitized product table and encode title, brand, and description in product_id order",
            )
        valid_pca = pca_path.exists() and np.load(pca_path, mmap_mode="r").shape == (n_items, 32)
        if not valid_pca:
            from sklearn.decomposition import PCA

            started = time.time()
            embedding = np.asarray(np.load(embedding_path, mmap_mode="r"), dtype=np.float32)
            pca = PCA(n_components=32, svd_solver="randomized", iterated_power=3, random_state=1337)
            reduced = pca.fit_transform(embedding).astype(np.float16)
            temp = pca_path.with_suffix(f".{os.getpid()}.tmp.npy")
            np.save(temp, reduced)
            os.replace(temp, pca_path)
            print(f"[content] pca rows={n_items} elapsed={time.time() - started:.1f}s", flush=True)
            register_artifact(
                cache_dir,
                "Amazon Books MiniLM PCA-32",
                pca_path,
                "PCA-32 product content matrix in product_id order",
                "rel-amazon-books-product-minilm-l6v2-pca32-f16-v1",
                "Fit randomized PCA-32 with seed 1337 to the registered MiniLM embeddings",
            )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return np.asarray(np.load(pca_path, mmap_mode="r"), dtype=np.float32)


def product_metadata(product, cache_dir, debug=False):
    pca = encode_products(product, cache_dir, debug)
    categories = product["category"].to_numpy()
    top = []
    leaf = []
    depth = np.empty(len(product), dtype=np.float32)
    for j, value in enumerate(categories):
        if isinstance(value, (list, tuple, np.ndarray)) and len(value):
            top.append(str(value[0]))
            leaf.append(str(value[-1]))
            depth[j] = len(value)
        else:
            top.append("__missing__")
            leaf.append("__missing__")
            depth[j] = 0
    category, _ = pd.factorize(pd.Series(leaf), sort=True)
    top_category, _ = pd.factorize(pd.Series(top), sort=True)
    brand, _ = pd.factorize(product["brand"].fillna("__missing__"), sort=True)
    title, _ = pd.factorize(product["title"].fillna("__missing__"), sort=True)
    category = category.astype(np.int32)
    top_category = top_category.astype(np.int32)
    brand = brand.astype(np.int32)
    title = title.astype(np.int32)
    category_frequency = np.bincount(category)[category].astype(np.float32)
    brand_frequency = np.bincount(brand)[brand].astype(np.float32)
    title_frequency = np.bincount(title)[title].astype(np.float32)
    price = product["price"].fillna(product["price"].median()).to_numpy(dtype=np.float32)
    log_price = np.log1p(np.maximum(price, 0))
    title_length = product["title"].fillna("").str.len().to_numpy(dtype=np.float32)
    description_length = product["description"].fillna("").str.len().to_numpy(dtype=np.float32)
    numeric = np.column_stack(
        [
            log_price,
            np.log1p(category_frequency),
            np.log1p(brand_frequency),
            np.log1p(title_frequency),
            np.log1p(title_length),
            np.log1p(description_length),
            depth,
        ]
    ).astype(np.float32)
    numeric = (numeric - numeric.mean(0)) / (numeric.std(0) + 1e-5)
    hashed = []
    for code in (category, top_category, brand):
        value = code.astype(np.uint64)
        hashed.append(((value * np.uint64(11400714819323198485)) % np.uint64(104729)).astype(np.float32) / 52364.5 - 1.0)
        hashed.append(((value * np.uint64(14029467366897019727)) % np.uint64(104723)).astype(np.float32) / 52361.5 - 1.0)
    matrix = np.column_stack([pca, numeric] + hashed).astype(np.float32)
    return ProductMetadata(
        matrix=matrix,
        category=category,
        top_category=top_category,
        brand=brand,
        title=title,
        category_frequency=category_frequency,
        brand_frequency=brand_frequency,
        title_frequency=title_frequency,
        log_price=log_price,
        title_length=title_length,
        description_length=description_length,
        category_depth=depth,
    )


def ridge_content_factors(state, metadata, alpha=20.0):
    observed = state.item_count > 0
    design = metadata.matrix
    x = np.column_stack([np.ones(observed.sum(), dtype=np.float32), design[observed]])
    target = np.column_stack(
        [
            state.item_bias[observed],
            state.item_factor[observed],
            state.implicit_factor[observed],
        ]
    ).astype(np.float32)
    gram = x.T @ x
    gram.flat[:: gram.shape[0] + 1] += alpha
    gram[0, 0] -= alpha
    coef = np.linalg.solve(gram.astype(np.float64), (x.T @ target).astype(np.float64)).astype(np.float32)
    mapped = coef[0] + design @ coef[1:]
    factors = state.item_factor.shape[1]
    mapped_bias = np.clip(mapped[:, 0], -1.5, 1.5).astype(np.float32)
    mapped_item = mapped[:, 1 : 1 + factors].astype(np.float32)
    mapped_implicit = mapped[:, 1 + factors :].astype(np.float32)
    item_bias = state.item_bias.copy()
    item_factor = state.item_factor.copy()
    implicit_factor = state.implicit_factor.copy()
    item_bias[~observed] = mapped_bias[~observed]
    item_factor[~observed] = mapped_item[~observed]
    implicit_factor[~observed] = mapped_implicit[~observed]
    return ContentFactors(
        item_bias=item_bias,
        item_factor=item_factor,
        implicit_factor=implicit_factor,
        mapped_item_bias=mapped_bias,
        mapped_item_factor=mapped_item,
        mapped_implicit_factor=mapped_implicit,
    )
