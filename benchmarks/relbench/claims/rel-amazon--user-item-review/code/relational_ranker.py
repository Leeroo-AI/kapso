from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


N_PRODUCTS = 506012
N_CUSTOMERS = 1850193
SOURCE_NAMES = ("brand", "coreview", "category", "series", "semantic", "popularity", "global")
SOURCE_QUOTAS = (250, 400, 100, 50, 100, 80, 20)
SOURCE_WEIGHTS = (4.0, 1.0, 0.5, 3.0, 1.5, 0.5, 0.1)


def elapsed(start: float) -> str:
    return f"{time.time() - start:.1f}s"


def top_indices(values: np.ndarray, size: int) -> np.ndarray:
    size = min(size, len(values))
    if size == 0:
        return np.empty(0, dtype=np.int32)
    take = np.argpartition(values, len(values) - size)[-size:]
    return take[np.argsort(values[take], kind="stable")[::-1]].astype(np.int32)


def stable_union(parts: list[np.ndarray], limit: int) -> np.ndarray:
    result = []
    seen = set()
    for part in parts:
        for value in part:
            item = int(value)
            if item not in seen:
                seen.add(item)
                result.append(item)
                if len(result) == limit:
                    return np.asarray(result, dtype=np.int32)
    return np.asarray(result, dtype=np.int32)


def grouped_order(groups: np.ndarray, score: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((np.arange(len(groups), dtype=np.int64), -score, groups)).astype(np.int32)
    counts = np.bincount(groups, minlength=group_count)
    pointers = np.empty(group_count + 1, dtype=np.int64)
    pointers[0] = 0
    np.cumsum(counts, out=pointers[1:])
    return order, pointers


def series_prefix(value: str) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\([^)]*(?:book|vol|volume|part|series|edition)[^)]*\)", " ", text)
    text = re.sub(r"\b(?:book|volume|vol|part|number|no)\.?\s*#?\s*(?:\d+|[ivxlcdm]+)\b", " ", text)
    text = re.sub(r"\s[-:–—,#]+\s*(?:\d+|[ivxlcdm]+)\s*$", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    tokens = re.findall(r"[a-z0-9]+", text)
    return " ".join(tokens[:7]) if len(tokens) >= 2 else ""


@dataclass
class Catalog:
    brand: np.ndarray
    category: np.ndarray
    price: np.ndarray
    metadata_missing: np.ndarray
    brand_count: int
    category_count: int
    brand_values: np.ndarray
    brand_to_id: dict[str, int]
    series: np.ndarray
    series_order: np.ndarray
    series_pointer: np.ndarray
    brand_catalog_size: np.ndarray
    category_catalog_size: np.ndarray
    customer_name_missing: np.ndarray
    customer_name_frequency: np.ndarray
    embeddings: np.ndarray | None
    embeddings_small: np.ndarray | None
    embedding_path: Path | None

    @classmethod
    def load(cls, use_text: bool) -> "Catalog":
        start = time.time()
        root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-amazon/db"
        con = duckdb.connect()
        product = con.execute(
            "SELECT product_id, coalesce(brand, ''), CASE WHEN category IS NULL OR len(category)=0 THEN '' ELSE category[len(category)] END, title, price, ((brand IS NULL OR trim(brand)='')::INTEGER + (category IS NULL OR len(category)=0)::INTEGER * 2 + (description IS NULL OR trim(description)='')::INTEGER * 4) FROM read_parquet(?) ORDER BY product_id",
            [str(root / "product.parquet")],
        ).fetchdf()
        if len(product) != N_PRODUCTS or not np.array_equal(product.iloc[:, 0].to_numpy(), np.arange(N_PRODUCTS)):
            raise RuntimeError("catalog product identifiers violate dense-index contract")
        brand_text = product.iloc[:, 1].astype(str).to_numpy()
        category_text = product.iloc[:, 2].astype(str).to_numpy()
        brand = np.zeros(N_PRODUCTS, dtype=np.int32)
        category = np.zeros(N_PRODUCTS, dtype=np.int32)
        brand_valid = brand_text != ""
        category_valid = category_text != ""
        brand_codes, encoded_brands = pd.factorize(brand_text[brand_valid], sort=True)
        category_codes, _ = pd.factorize(category_text[category_valid], sort=True)
        brand[brand_valid] = brand_codes.astype(np.int32) + 1
        category[category_valid] = category_codes.astype(np.int32) + 1
        brand_values = np.concatenate([np.asarray([""], dtype=object), encoded_brands.astype(object)])
        brand_to_id = {str(value): i for i, value in enumerate(brand_values) if i}
        keys = np.asarray([f"{brand[i]}|{series_prefix(value)}" if series_prefix(value) else "" for i, value in enumerate(product.iloc[:, 3].tolist())], dtype=object)
        series_codes, _ = pd.factorize(keys, sort=True)
        series_codes = (series_codes + 1).astype(np.int32)
        series_sizes = np.bincount(series_codes)
        invalid = (series_codes == 0) | (series_sizes[series_codes] < 2) | (series_sizes[series_codes] > 120)
        series_codes[invalid] = 0
        surviving = np.unique(series_codes)
        remap = np.zeros(int(series_codes.max()) + 1, dtype=np.int32)
        remap[surviving] = np.arange(len(surviving), dtype=np.int32)
        series_codes = remap[series_codes]
        series_order, series_pointer = grouped_order(series_codes, np.zeros(N_PRODUCTS, dtype=np.float32), int(series_codes.max()) + 1)
        brand_catalog_size = np.bincount(brand, minlength=int(brand.max()) + 1).astype(np.float32)
        category_catalog_size = np.bincount(category, minlength=int(category.max()) + 1).astype(np.float32)
        customers = con.execute(
            "WITH names AS (SELECT customer_name, count(*) n FROM read_parquet(?) GROUP BY customer_name) SELECT c.customer_id, (c.customer_name IS NULL OR trim(c.customer_name)='')::UTINYINT, coalesce(names.n, 0) FROM read_parquet(?) c LEFT JOIN names USING(customer_name) ORDER BY customer_id",
            [str(root / "customer.parquet"), str(root / "customer.parquet")],
        ).fetchdf()
        customer_name_missing = np.ones(N_CUSTOMERS, dtype=np.float32)
        customer_name_frequency = np.zeros(N_CUSTOMERS, dtype=np.float32)
        ids = customers.iloc[:, 0].to_numpy(dtype=np.int64)
        valid = (ids >= 0) & (ids < N_CUSTOMERS)
        customer_name_missing[ids[valid]] = customers.iloc[:, 1].to_numpy(dtype=np.float32)[valid]
        customer_name_frequency[ids[valid]] = customers.iloc[:, 2].to_numpy(dtype=np.float32)[valid]
        embeddings = None
        embeddings_small = None
        embedding_path = None
        if use_text:
            shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
            matches = sorted(shared.glob("lane0_catalog_bge_*.npy"), key=lambda path: path.stat().st_mtime)
            if matches:
                embedding_path = matches[-1]
                embeddings = np.load(embedding_path, mmap_mode="r")
                if embeddings.shape != (N_PRODUCTS, 384):
                    raise RuntimeError(f"embedding shape is {embeddings.shape}")
                small = np.asarray(embeddings[:, :48], dtype=np.float32)
                norms = np.linalg.norm(small, axis=1, keepdims=True)
                embeddings_small = (small / np.maximum(norms, 1e-8)).astype(np.float16)
        print(f"[catalog] products={len(product)} brands={int(brand.max())} categories={int(category.max())} series={int(series_codes.max())} text={embeddings is not None} elapsed={elapsed(start)}", flush=True)
        return cls(
            brand=brand,
            category=category,
            price=product.iloc[:, 4].to_numpy(dtype=np.float32),
            metadata_missing=product.iloc[:, 5].to_numpy(dtype=np.float32),
            brand_count=int(brand.max()) + 1,
            category_count=int(category.max()) + 1,
            brand_values=brand_values,
            brand_to_id=brand_to_id,
            series=series_codes,
            series_order=series_order,
            series_pointer=series_pointer,
            brand_catalog_size=brand_catalog_size,
            category_catalog_size=category_catalog_size,
            customer_name_missing=customer_name_missing,
            customer_name_frequency=customer_name_frequency,
            embeddings=embeddings,
            embeddings_small=embeddings_small,
            embedding_path=embedding_path,
        )


@dataclass
class Histories:
    customer: np.ndarray
    product: np.ndarray
    time_days: np.ndarray
    rating: np.ndarray
    verified: np.ndarray
    detailed: np.ndarray
    text_length: np.ndarray
    summary_length: np.ndarray

    def slice(self, customer_id: int) -> slice:
        left = int(np.searchsorted(self.customer, customer_id, side="left"))
        right = int(np.searchsorted(self.customer, customer_id, side="right"))
        return slice(left, right)


@dataclass
class Snapshot:
    cutoff: pd.Timestamp
    all_count: np.ndarray
    count30: np.ndarray
    count91: np.ndarray
    count365: np.ndarray
    detail_all: np.ndarray
    detail30: np.ndarray
    detail91: np.ndarray
    detail365: np.ndarray
    season_detail: np.ndarray
    rating_mean: np.ndarray
    rating_std: np.ndarray
    verified_share: np.ndarray
    text_mean: np.ndarray
    first_age: np.ndarray
    last_age: np.ndarray
    brand_recent_detail: np.ndarray
    brand_detail_share: np.ndarray
    brand_low_history: np.ndarray
    category_recent_detail: np.ndarray
    category_detail_share: np.ndarray
    category_low_history: np.ndarray
    brand_pop_order: np.ndarray
    brand_pop_pointer: np.ndarray
    brand_low_order: np.ndarray
    brand_low_pointer: np.ndarray
    category_pop_order: np.ndarray
    category_pop_pointer: np.ndarray
    category_low_order: np.ndarray
    category_low_pointer: np.ndarray
    co_pointer: np.ndarray
    co_product: np.ndarray
    co_count: np.ndarray
    adjacent_pointer: np.ndarray
    adjacent_brand: np.ndarray
    popularity: np.ndarray
    global_padding: np.ndarray

    @classmethod
    def build(cls, catalog: Catalog, cutoff: pd.Timestamp) -> "Snapshot":
        start = time.time()
        shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
        cache = shared / f"lane0_snapshot_{cutoff.strftime('%Y%m%d')}_v3.npz"
        if cache.exists():
            data = np.load(cache, allow_pickle=False)
            values = {name: data[name] for name in data.files}
            print(f"[snapshot] cache_hit cutoff={cutoff.date()} bytes={cache.stat().st_size} elapsed={elapsed(start)}", flush=True)
            return cls(cutoff=cutoff, **values)
        event_path = shared / "lane0_events_detail_v1.parquet"
        product_path = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-amazon/db/product.parquet"
        con = duckdb.connect()
        con.execute("SET threads=11")
        con.execute("SET memory_limit='85GB'")
        con.execute("SET temp_directory='output_data_generic_exp_0/ducktmp'")
        season_start = cutoff - pd.DateOffset(years=1)
        season_stop = season_start + pd.Timedelta(days=91)
        query = """
            SELECT product_id,
                count(*) count_all,
                count(*) FILTER (WHERE review_time > ? - INTERVAL 30 DAY) count30,
                count(*) FILTER (WHERE review_time > ? - INTERVAL 91 DAY) count91,
                count(*) FILTER (WHERE review_time > ? - INTERVAL 365 DAY) count365,
                sum(detailed) detail_all,
                sum(detailed) FILTER (WHERE review_time > ? - INTERVAL 30 DAY) detail30,
                sum(detailed) FILTER (WHERE review_time > ? - INTERVAL 91 DAY) detail91,
                sum(detailed) FILTER (WHERE review_time > ? - INTERVAL 365 DAY) detail365,
                sum(detailed) FILTER (WHERE review_time > ? AND review_time <= ?) season_detail,
                avg(rating) rating_mean,
                stddev_pop(rating) rating_std,
                avg(verified::INTEGER) verified_share,
                avg(coalesce(text_length, 0)) text_mean,
                datediff('day', min(review_time), ?) first_age,
                datediff('day', max(review_time), ?) last_age
            FROM read_parquet(?) WHERE review_time <= ? GROUP BY product_id
        """
        params = [cutoff, cutoff, cutoff, cutoff, cutoff, cutoff, season_start, season_stop, cutoff, cutoff, str(event_path), cutoff]
        stats = con.execute(query, params).fetchdf()
        names = ["all_count", "count30", "count91", "count365", "detail_all", "detail30", "detail91", "detail365", "season_detail", "rating_mean", "rating_std", "verified_share", "text_mean", "first_age", "last_age"]
        arrays = {name: np.zeros(N_PRODUCTS, dtype=np.float32) for name in names}
        ids = stats.iloc[:, 0].to_numpy(dtype=np.int32)
        for position, name in enumerate(names, start=1):
            arrays[name][ids] = stats.iloc[:, position].fillna(0).to_numpy(dtype=np.float32)
        pop_score = 2.5 * np.log1p(arrays["detail30"]) + 2.0 * np.log1p(arrays["detail91"]) + np.log1p(arrays["detail365"]) + 0.25 * np.log1p(arrays["all_count"])
        hashed = ((np.arange(N_PRODUCTS, dtype=np.uint64) * np.uint64(11400714819323198485)) >> np.uint64(32)).astype(np.float32) / np.float32(2**32)
        low_score = np.where(arrays["all_count"] <= 2, 1_000_000.0 + hashed, pop_score)
        brand_pop_order, brand_pop_pointer = grouped_order(catalog.brand, pop_score, catalog.brand_count)
        brand_low_order, brand_low_pointer = grouped_order(catalog.brand, low_score, catalog.brand_count)
        category_pop_order, category_pop_pointer = grouped_order(catalog.category, pop_score, catalog.category_count)
        category_low_order, category_low_pointer = grouped_order(catalog.category, low_score, catalog.category_count)
        brand_recent_detail = np.bincount(catalog.brand, weights=arrays["detail91"], minlength=catalog.brand_count).astype(np.float32)
        brand_total_detail = np.bincount(catalog.brand, weights=arrays["detail_all"], minlength=catalog.brand_count).astype(np.float32)
        brand_total_count = np.bincount(catalog.brand, weights=arrays["all_count"], minlength=catalog.brand_count).astype(np.float32)
        brand_detail_share = brand_total_detail / np.maximum(brand_total_count, 1)
        brand_low_history = np.bincount(catalog.brand, weights=(arrays["all_count"] <= 2), minlength=catalog.brand_count).astype(np.float32)
        category_recent_detail = np.bincount(catalog.category, weights=arrays["detail91"], minlength=catalog.category_count).astype(np.float32)
        category_total_detail = np.bincount(catalog.category, weights=arrays["detail_all"], minlength=catalog.category_count).astype(np.float32)
        category_total_count = np.bincount(catalog.category, weights=arrays["all_count"], minlength=catalog.category_count).astype(np.float32)
        category_detail_share = category_total_detail / np.maximum(category_total_count, 1)
        category_low_history = np.bincount(catalog.category, weights=(arrays["all_count"] <= 2), minlength=catalog.category_count).astype(np.float32)
        co_query = """
            WITH capped AS (
                SELECT customer_id, product_id FROM (
                    SELECT customer_id, product_id, row_number() OVER (PARTITION BY customer_id ORDER BY review_time DESC) rn
                    FROM read_parquet(?) WHERE review_time <= ?
                ) WHERE rn <= 50 GROUP BY customer_id, product_id
            ), pairs AS (
                SELECT a.product_id src, b.product_id dst, count(*) cnt
                FROM capped a JOIN capped b USING(customer_id)
                WHERE a.product_id <> b.product_id
                GROUP BY a.product_id, b.product_id HAVING count(*) >= 2
            ), ranked AS (
                SELECT src, dst, cnt, row_number() OVER (PARTITION BY src ORDER BY cnt DESC, dst) rn FROM pairs
            )
            SELECT src, dst, cnt FROM ranked WHERE rn <= 40 ORDER BY src, rn
        """
        co = con.execute(co_query, [str(event_path), cutoff]).fetchdf()
        co_source = co.iloc[:, 0].to_numpy(dtype=np.int32)
        co_product = co.iloc[:, 1].to_numpy(dtype=np.int32)
        co_count = co.iloc[:, 2].to_numpy(dtype=np.float32)
        co_pointer = np.zeros(N_PRODUCTS + 1, dtype=np.int64)
        np.cumsum(np.bincount(co_source, minlength=N_PRODUCTS), out=co_pointer[1:])
        adjacent_query = """
            WITH capped AS (
                SELECT customer_id, brand FROM (
                    SELECT e.customer_id, p.brand, row_number() OVER (PARTITION BY e.customer_id ORDER BY e.review_time DESC) rn
                    FROM read_parquet(?) e JOIN read_parquet(?) p USING(product_id)
                    WHERE e.review_time <= ? AND p.brand IS NOT NULL AND trim(p.brand) <> ''
                ) WHERE rn <= 50 GROUP BY customer_id, brand
            ), pairs AS (
                SELECT a.brand src, b.brand dst, count(*) cnt
                FROM capped a JOIN capped b USING(customer_id) WHERE a.brand <> b.brand
                GROUP BY a.brand, b.brand HAVING count(*) >= 3
            ), ranked AS (
                SELECT src, dst, row_number() OVER(PARTITION BY src ORDER BY cnt DESC, dst) rn FROM pairs
            )
            SELECT src, dst FROM ranked WHERE rn <= 5 ORDER BY src, rn
        """
        adjacent = con.execute(adjacent_query, [str(event_path), str(product_path), cutoff]).fetchall()
        adjacent_pairs = [(catalog.brand_to_id.get(str(src), 0), catalog.brand_to_id.get(str(dst), 0)) for src, dst in adjacent]
        adjacent_pairs = [(src, dst) for src, dst in adjacent_pairs if src and dst]
        adjacent_pairs.sort()
        adjacent_source = np.asarray([value[0] for value in adjacent_pairs], dtype=np.int32)
        adjacent_brand = np.asarray([value[1] for value in adjacent_pairs], dtype=np.int32)
        adjacent_pointer = np.zeros(catalog.brand_count + 1, dtype=np.int64)
        np.cumsum(np.bincount(adjacent_source, minlength=catalog.brand_count), out=adjacent_pointer[1:])
        popularity = stable_union(
            [top_indices(arrays["detail30"], 80), top_indices(arrays["detail91"], 100), top_indices(arrays["detail365"], 100), top_indices(arrays["season_detail"], 100), top_indices(arrays["all_count"], 100)],
            400,
        )
        global_padding = stable_union([top_indices(pop_score, 3000), top_indices(arrays["all_count"], 3000)], 5000)
        values = dict(
            **arrays,
            brand_recent_detail=brand_recent_detail,
            brand_detail_share=brand_detail_share,
            brand_low_history=brand_low_history,
            category_recent_detail=category_recent_detail,
            category_detail_share=category_detail_share,
            category_low_history=category_low_history,
            brand_pop_order=brand_pop_order,
            brand_pop_pointer=brand_pop_pointer,
            brand_low_order=brand_low_order,
            brand_low_pointer=brand_low_pointer,
            category_pop_order=category_pop_order,
            category_pop_pointer=category_pop_pointer,
            category_low_order=category_low_order,
            category_low_pointer=category_low_pointer,
            co_pointer=co_pointer,
            co_product=co_product,
            co_count=co_count,
            adjacent_pointer=adjacent_pointer,
            adjacent_brand=adjacent_brand,
            popularity=popularity,
            global_padding=global_padding,
        )
        np.savez(cache, **values)
        print(f"[snapshot] built cutoff={cutoff.date()} reviewed={len(stats)} coreview_edges={len(co_product)} adjacent_edges={len(adjacent_brand)} elapsed={elapsed(start)}", flush=True)
        return cls(cutoff=cutoff, **values)

    def group_products(self, group: int, source: str, low: bool, limit: int) -> np.ndarray:
        if group <= 0:
            return np.empty(0, dtype=np.int32)
        if source == "brand":
            pointer = self.brand_low_pointer if low else self.brand_pop_pointer
            order = self.brand_low_order if low else self.brand_pop_order
        else:
            pointer = self.category_low_pointer if low else self.category_pop_pointer
            order = self.category_low_order if low else self.category_pop_order
        start = int(pointer[group])
        stop = min(int(pointer[group + 1]), start + limit)
        return order[start:stop]


def fetch_histories(customers: np.ndarray, cutoff: pd.Timestamp) -> Histories:
    start = time.time()
    users = pd.DataFrame({"customer_id": np.unique(customers.astype(np.int64))})
    con = duckdb.connect()
    con.execute("SET threads=11")
    con.register("seed_users", users)
    event_path = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane0_events_detail_v1.parquet"
    frame = con.execute(
        "SELECT e.customer_id, e.product_id, datediff('day', e.review_time, ?) age_days, e.rating, e.verified::UTINYINT, e.detailed, coalesce(e.text_length, 0), coalesce(e.summary_length, 0) FROM read_parquet(?) e JOIN seed_users s USING(customer_id) WHERE e.review_time <= ? ORDER BY e.customer_id, e.review_time DESC",
        [cutoff, str(event_path), cutoff],
    ).fetchdf()
    print(f"[history] cutoff={cutoff.date()} users={len(users)} events={len(frame)} elapsed={elapsed(start)}", flush=True)
    return Histories(
        customer=frame.iloc[:, 0].to_numpy(dtype=np.int64),
        product=frame.iloc[:, 1].to_numpy(dtype=np.int32),
        time_days=frame.iloc[:, 2].to_numpy(dtype=np.float32),
        rating=frame.iloc[:, 3].to_numpy(dtype=np.float32),
        verified=frame.iloc[:, 4].to_numpy(dtype=np.float32),
        detailed=frame.iloc[:, 5].to_numpy(dtype=np.float32),
        text_length=frame.iloc[:, 6].to_numpy(dtype=np.float32),
        summary_length=frame.iloc[:, 7].to_numpy(dtype=np.float32),
    )


class SemanticIndex:
    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.index = None
        if catalog.embeddings is None or catalog.embedding_path is None:
            return
        import faiss

        start = time.time()
        shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
        path = shared / f"lane0_hnsw_{catalog.embedding_path.stem}_m32_v1.faiss"
        if path.exists():
            self.index = faiss.read_index(str(path))
            self.index.hnsw.efSearch = 80
            print(f"[semantic] index_cache_hit ntotal={self.index.ntotal} elapsed={elapsed(start)}", flush=True)
            return
        index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        for begin in range(0, N_PRODUCTS, 32768):
            values = np.asarray(catalog.embeddings[begin:begin + 32768], dtype=np.float32)
            index.add(values)
        index.hnsw.efSearch = 80
        faiss.write_index(index, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)
        self.index = index
        print(f"[semantic] index_built ntotal={index.ntotal} elapsed={elapsed(start)}", flush=True)

    def search(self, profiles: np.ndarray, size: int = 240) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            return np.empty((len(profiles), 0), dtype=np.float32), np.empty((len(profiles), 0), dtype=np.int32)
        return self.index.search(np.ascontiguousarray(profiles, dtype=np.float32), size)


class CandidateBuilder:
    def __init__(self, catalog: Catalog, snapshot: Snapshot, semantic_index: SemanticIndex | None, semantic_enabled: bool):
        self.catalog = catalog
        self.snapshot = snapshot
        self.semantic_index = semantic_index
        self.semantic_enabled = semantic_enabled and semantic_index is not None and semantic_index.index is not None

    def profiles(self, histories: Histories, customers: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if not self.semantic_enabled or self.catalog.embeddings is None or self.catalog.embeddings_small is None:
            return None, None, None, None
        dimension = self.catalog.embeddings.shape[1]
        long_profiles = np.zeros((len(customers), dimension), dtype=np.float32)
        recent_small = np.zeros((len(customers), self.catalog.embeddings_small.shape[1]), dtype=np.float32)
        long_small = np.zeros_like(recent_small)
        for row, customer in enumerate(customers):
            section = histories.slice(int(customer))
            products = histories.product[section][:50]
            ages = histories.time_days[section][:50]
            if len(products) == 0:
                continue
            weights = np.power(0.5, ages / 365.0).astype(np.float32)
            vectors = np.asarray(self.catalog.embeddings[products], dtype=np.float32)
            profile = np.average(vectors, axis=0, weights=weights)
            profile /= max(float(np.linalg.norm(profile)), 1e-8)
            long_profiles[row] = profile
            long_slice = np.asarray(self.catalog.embeddings_small[products], dtype=np.float32)
            long_value = np.average(long_slice, axis=0, weights=weights)
            long_small[row] = long_value / max(float(np.linalg.norm(long_value)), 1e-8)
            recent_value = np.mean(long_slice[:5], axis=0)
            recent_small[row] = recent_value / max(float(np.linalg.norm(recent_value)), 1e-8)
        distances, neighbors = self.semantic_index.search(long_profiles)
        return long_small, recent_small, distances, neighbors

    def user_context(self, customer: int, histories: Histories) -> dict:
        section = histories.slice(customer)
        products = histories.product[section]
        ages = histories.time_days[section]
        ratings = histories.rating[section]
        verified = histories.verified[section]
        detailed = histories.detailed[section]
        text_length = histories.text_length[section]
        summary_length = histories.summary_length[section]
        recent_products = []
        seen_recent = set()
        for item in products:
            value = int(item)
            if value not in seen_recent:
                seen_recent.add(value)
                recent_products.append(value)
                if len(recent_products) == 50:
                    break
        brands = self.catalog.brand[products] if len(products) else np.empty(0, dtype=np.int32)
        categories = self.catalog.category[products] if len(products) else np.empty(0, dtype=np.int32)
        brand_affinity = {}
        category_affinity = {}
        for brand, category, age in zip(brands[:100], categories[:100], ages[:100]):
            brand = int(brand)
            category = int(category)
            weight = float(0.5 ** (float(age) / 365.0))
            if brand:
                old = brand_affinity.get(brand, [0.0, 1e9])
                brand_affinity[brand] = [old[0] + weight, min(old[1], float(age))]
            if category:
                old = category_affinity.get(category, [0.0, 1e9])
                category_affinity[category] = [old[0] + weight, min(old[1], float(age))]
        user_features = np.asarray([
            math.log1p(len(products)),
            math.log1p(int(np.sum(ages <= 30))) if len(ages) else 0,
            math.log1p(int(np.sum(ages <= 91))) if len(ages) else 0,
            math.log1p(int(np.sum(ages <= 365))) if len(ages) else 0,
            math.log1p(float(ages[0])) if len(ages) else math.log1p(3650),
            float(np.mean(ratings)) if len(ratings) else 0,
            float(np.std(ratings)) if len(ratings) else 0,
            float(np.mean(verified)) if len(verified) else 0,
            float(np.mean(detailed)) if len(detailed) else 0,
            math.log1p(float(np.mean(text_length))) if len(text_length) else 0,
            math.log1p(float(np.std(text_length))) if len(text_length) else 0,
            math.log1p(float(np.mean(summary_length))) if len(summary_length) else 0,
            math.log1p(len(set(int(value) for value in brands if value))),
            math.log1p(len(set(int(value) for value in categories if value))),
            math.log1p(float(np.mean(self.catalog.price[products]))) if len(products) else 0,
            math.log1p(float(np.std(self.catalog.price[products]))) if len(products) else 0,
            float(self.catalog.customer_name_missing[customer]) if 0 <= customer < N_CUSTOMERS else 1,
            math.log1p(float(self.catalog.customer_name_frequency[customer])) if 0 <= customer < N_CUSTOMERS else 0,
        ], dtype=np.float32)
        return {
            "products": products,
            "ages": ages,
            "recent": recent_products,
            "seen": set(int(value) for value in products),
            "brand_affinity": brand_affinity,
            "category_affinity": category_affinity,
            "top_brands": sorted(brand_affinity, key=lambda value: (-brand_affinity[value][0], brand_affinity[value][1]))[:12],
            "top_categories": sorted(category_affinity, key=lambda value: (-category_affinity[value][0], category_affinity[value][1]))[:3],
            "features": user_features,
        }

    def source_lists(self, context: dict, semantic_distance: np.ndarray | None, semantic_neighbor: np.ndarray | None) -> tuple[list[list[int]], dict[int, tuple[float, float]], dict[int, float]]:
        seen = context["seen"]
        brand_scores = {}
        for brand_position, brand in enumerate(context["top_brands"]):
            affinity = context["brand_affinity"][brand][0] / (1 + brand_position)
            popular = self.snapshot.group_products(brand, "brand", False, 45)
            low = self.snapshot.group_products(brand, "brand", True, 45)
            for source_position, product in enumerate(np.concatenate([popular, low])):
                item = int(product)
                if item not in seen:
                    score = affinity + 0.2 * math.log1p(float(self.snapshot.detail91[item])) + 0.15 / (1 + source_position)
                    brand_scores[item] = max(brand_scores.get(item, -1e9), score)
        brand_list = sorted(brand_scores, key=lambda item: (-brand_scores[item], item))[:1200]
        co_scores = {}
        co_extra = {}
        for recent_position, source in enumerate(context["recent"][:30]):
            left = int(self.snapshot.co_pointer[source])
            right = int(self.snapshot.co_pointer[source + 1])
            decay = 1.0 / math.sqrt(1 + recent_position)
            for position in range(left, right):
                item = int(self.snapshot.co_product[position])
                if item in seen:
                    continue
                strength = float(self.snapshot.co_count[position])
                co_scores[item] = co_scores.get(item, 0.0) + decay * math.log1p(strength)
                previous = co_extra.get(item, (0.0, 0.0))
                co_extra[item] = (previous[0] + strength, max(previous[1], strength))
        co_list = sorted(co_scores, key=lambda item: (-co_scores[item], item))[:1200]
        category_scores = {}
        for category_position, category in enumerate(context["top_categories"]):
            affinity = context["category_affinity"][category][0] / (1 + category_position)
            for source_position, product in enumerate(np.concatenate([self.snapshot.group_products(category, "category", False, 55), self.snapshot.group_products(category, "category", True, 35)])):
                item = int(product)
                if item not in seen:
                    category_scores[item] = max(category_scores.get(item, -1e9), affinity + 0.1 / (1 + source_position))
        for brand_position, brand in enumerate(context["top_brands"][:6]):
            left = int(self.snapshot.adjacent_pointer[brand])
            right = int(self.snapshot.adjacent_pointer[brand + 1])
            for adjacent_position, adjacent in enumerate(self.snapshot.adjacent_brand[left:right]):
                adjacent = int(adjacent)
                for source_position, product in enumerate(np.concatenate([self.snapshot.group_products(adjacent, "brand", False, 12), self.snapshot.group_products(adjacent, "brand", True, 6)])):
                    item = int(product)
                    if item not in seen:
                        score = 0.5 / (1 + brand_position) + 0.2 / (1 + adjacent_position) + 0.05 / (1 + source_position)
                        category_scores[item] = max(category_scores.get(item, -1e9), score)
        category_list = sorted(category_scores, key=lambda item: (-category_scores[item], item))[:500]
        series_scores = {}
        for position, source in enumerate(context["recent"][:30]):
            code = int(self.catalog.series[source])
            if code == 0:
                continue
            left = int(self.catalog.series_pointer[code])
            right = int(self.catalog.series_pointer[code + 1])
            for product in self.catalog.series_order[left:right]:
                item = int(product)
                if item not in seen:
                    series_scores[item] = max(series_scores.get(item, 0), 1.0 / math.sqrt(1 + position))
        series_list = sorted(series_scores, key=lambda item: (-series_scores[item], item))[:300]
        semantic_scores = {}
        if semantic_neighbor is not None and semantic_distance is not None:
            categories = set(context["top_categories"])
            for item, score in zip(semantic_neighbor, semantic_distance):
                item = int(item)
                if item < 0 or item in seen:
                    continue
                if self.snapshot.all_count[item] <= 5 or int(self.catalog.category[item]) in categories:
                    semantic_scores[item] = float(score)
                    if len(semantic_scores) >= 160:
                        break
        semantic_list = sorted(semantic_scores, key=lambda item: (-semantic_scores[item], item))
        popularity_list = [int(item) for item in self.snapshot.popularity if int(item) not in seen]
        global_list = [int(item) for item in self.snapshot.global_padding if int(item) not in seen]
        return [brand_list, co_list, category_list, series_list, semantic_list, popularity_list, global_list], co_extra, semantic_scores

    def merge_candidates(self, source_lists: list[list[int]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        source_rank = {}
        heuristic = {}
        for source, items in enumerate(source_lists):
            for rank, item in enumerate(items):
                ranks = source_rank.setdefault(item, np.zeros(len(SOURCE_NAMES), dtype=np.float32))
                ranks[source] = 1.0 / (rank + 1.0)
                heuristic[item] = heuristic.get(item, 0.0) + SOURCE_WEIGHTS[source] / (5.0 + rank)
        selected = []
        selected_set = set()
        positions = [0] * len(SOURCE_NAMES)
        for source, quota in enumerate(SOURCE_QUOTAS):
            items = source_lists[source]
            source_selected = sum(1 for value in selected if source_rank[value][source] > 0)
            while positions[source] < len(items) and source_selected < quota:
                item = items[positions[source]]
                positions[source] += 1
                if item not in selected_set:
                    selected_set.add(item)
                    selected.append(item)
                    source_selected += 1
                if len(selected) == 1000:
                    break
            if len(selected) == 1000:
                break
        source = 0
        stalled = 0
        while len(selected) < 1000 and stalled < len(SOURCE_NAMES):
            items = source_lists[source]
            if positions[source] < len(items):
                item = items[positions[source]]
                positions[source] += 1
                if item not in selected_set:
                    selected_set.add(item)
                    selected.append(item)
                stalled = 0
            else:
                stalled += 1
            source = (source + 1) % len(SOURCE_NAMES)
        selected.sort(key=lambda item: (-heuristic.get(item, 0.0), item))
        products = np.asarray(selected, dtype=np.int32)
        ranks = np.vstack([source_rank[item] for item in selected]).astype(np.float32) if selected else np.empty((0, len(SOURCE_NAMES)), dtype=np.float32)
        flags = (ranks > 0).astype(np.float32)
        heuristic_values = np.asarray([heuristic[item] for item in selected], dtype=np.float32)
        return products, flags, ranks, heuristic_values

    def features(self, customer: int, context: dict, products: np.ndarray, flags: np.ndarray, ranks: np.ndarray, heuristic: np.ndarray, co_extra: dict[int, tuple[float, float]], semantic_scores: dict[int, float], long_small: np.ndarray | None, recent_small: np.ndarray | None) -> np.ndarray:
        snapshot = self.snapshot
        catalog = self.catalog
        brand = catalog.brand[products]
        category = catalog.category[products]
        base = np.column_stack([
            np.log1p(snapshot.all_count[products]),
            np.log1p(snapshot.count30[products]),
            np.log1p(snapshot.count91[products]),
            np.log1p(snapshot.count365[products]),
            np.log1p(snapshot.detail_all[products]),
            np.log1p(snapshot.detail30[products]),
            np.log1p(snapshot.detail91[products]),
            np.log1p(snapshot.detail365[products]),
            snapshot.detail30[products] / np.maximum(snapshot.detail91[products], 1),
            np.log1p(snapshot.first_age[products]),
            np.log1p(snapshot.last_age[products]),
            snapshot.rating_mean[products],
            snapshot.rating_std[products],
            snapshot.verified_share[products],
            np.log1p(catalog.price[products]),
            catalog.metadata_missing[products],
            np.log1p(catalog.brand_catalog_size[brand]),
            np.log1p(catalog.category_catalog_size[category]),
            np.log1p(snapshot.brand_recent_detail[brand]),
            snapshot.brand_detail_share[brand],
            np.log1p(snapshot.brand_low_history[brand]),
            np.log1p(snapshot.category_recent_detail[category]),
            snapshot.category_detail_share[category],
            np.log1p(snapshot.category_low_history[category]),
        ]).astype(np.float32)
        brand_affinity = np.asarray([context["brand_affinity"].get(int(value), (0, 3650))[0] for value in brand], dtype=np.float32)
        brand_recency = np.asarray([context["brand_affinity"].get(int(value), (0, 3650))[1] for value in brand], dtype=np.float32)
        category_affinity = np.asarray([context["category_affinity"].get(int(value), (0, 3650))[0] for value in category], dtype=np.float32)
        category_recency = np.asarray([context["category_affinity"].get(int(value), (0, 3650))[1] for value in category], dtype=np.float32)
        co_sum = np.asarray([co_extra.get(int(value), (0, 0))[0] for value in products], dtype=np.float32)
        co_max = np.asarray([co_extra.get(int(value), (0, 0))[1] for value in products], dtype=np.float32)
        price_mean = math.expm1(float(context["features"][14]))
        price_std = max(math.expm1(float(context["features"][15])), 1.0)
        price_deviation = np.abs(catalog.price[products] - price_mean) / price_std
        semantic_long = np.asarray([semantic_scores.get(int(value), 0.0) for value in products], dtype=np.float32)
        semantic_recent = semantic_long.copy()
        if long_small is not None and recent_small is not None and catalog.embeddings_small is not None and len(products):
            vectors = np.asarray(catalog.embeddings_small[products], dtype=np.float32)
            semantic_long = vectors @ long_small
            semantic_recent = vectors @ recent_small
        series_score = np.asarray([max((ranks[i, 3]), 0) for i in range(len(products))], dtype=np.float32)
        novelty = 1.0 / np.sqrt(1.0 + snapshot.all_count[products])
        cross = np.column_stack([
            np.log1p(brand_affinity),
            np.log1p(brand_recency),
            np.log1p(category_affinity),
            np.log1p(category_recency),
            np.log1p(co_sum),
            np.log1p(co_max),
            np.log1p(price_deviation),
            semantic_long,
            semantic_recent,
            series_score,
            novelty,
            heuristic,
        ]).astype(np.float32)
        users = np.repeat(context["features"][None, :], len(products), axis=0)
        return np.column_stack([users, base, cross, flags, ranks]).astype(np.float32)


FEATURE_NAMES = (
    "u_count_all", "u_count30", "u_count91", "u_count365", "u_days_last", "u_rating_mean", "u_rating_std", "u_verified_share", "u_detail_share", "u_text_mean", "u_text_std", "u_summary_mean", "u_distinct_brands", "u_distinct_categories", "u_price_mean", "u_price_std", "u_name_missing", "u_name_frequency",
    "p_count_all", "p_count30", "p_count91", "p_count365", "p_detail_all", "p_detail30", "p_detail91", "p_detail365", "p_acceleration", "p_first_age", "p_last_age", "p_rating_mean", "p_rating_std", "p_verified_share", "p_price", "p_metadata_missing", "brand_catalog_size", "category_catalog_size", "brand_recent_detail", "brand_detail_share", "brand_low_history", "category_recent_detail", "category_detail_share", "category_low_history",
    "cross_brand_affinity", "cross_brand_recency", "cross_category_affinity", "cross_category_recency", "cross_coreview_sum", "cross_coreview_max", "cross_price_deviation", "semantic_long", "semantic_recent", "series_score", "product_novelty", "heuristic_blend",
    "flag_brand", "flag_coreview", "flag_category", "flag_series", "flag_semantic", "flag_popularity", "flag_global", "rank_brand", "rank_coreview", "rank_category", "rank_series", "rank_semantic", "rank_popularity", "rank_global",
)


@dataclass
class RankingData:
    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    candidates: list[np.ndarray]
    truths: list[set[int]]
    recall200: float
    recall800: float
    recall800_no_semantic: float
    strata: dict


def build_ranking_data(catalog: Catalog, snapshot: Snapshot, semantic_index: SemanticIndex | None, seeds: pd.DataFrame, truths: list, semantic_enabled: bool, negative_limit: int, training: bool) -> RankingData:
    start = time.time()
    customers = seeds["customer_id"].to_numpy(dtype=np.int64)
    histories = fetch_histories(customers, snapshot.cutoff)
    builder = CandidateBuilder(catalog, snapshot, semantic_index, semantic_enabled)
    long_small, recent_small, semantic_distances, semantic_neighbors = builder.profiles(histories, customers)
    feature_parts = []
    label_parts = []
    groups = []
    candidates = []
    truth_sets = [set(int(value) for value in values) for values in truths]
    kept_truths = []
    hits200 = 0
    hits800 = 0
    hits800_no_semantic = 0
    truth_total = 0
    strata = {}
    for row, customer in enumerate(customers):
        context = builder.user_context(int(customer), histories)
        distance = semantic_distances[row] if semantic_distances is not None else None
        neighbors = semantic_neighbors[row] if semantic_neighbors is not None else None
        sources, co_extra, semantic_scores = builder.source_lists(context, distance, neighbors)
        products, flags, ranks, heuristic = builder.merge_candidates(sources)
        sources_without_semantic = [values if index != 4 else [] for index, values in enumerate(sources)]
        products_without_semantic, _, _, _ = builder.merge_candidates(sources_without_semantic)
        truth = truth_sets[row]
        truth_total += len(truth)
        hits200 += sum(int(value) in truth for value in products[:200])
        hits800 += sum(int(value) in truth for value in products[:800])
        hits800_no_semantic += sum(int(value) in truth for value in products_without_semantic[:800])
        history_count = len(context["products"])
        history_bin = "cold" if history_count == 0 else "1-5" if history_count <= 5 else "6-20" if history_count <= 20 else ">20"
        record = strata.setdefault(history_bin, [0, 0, 0])
        record[0] += 1
        record[1] += len(truth)
        record[2] += sum(int(value) in truth for value in products[:800])
        if training:
            labels = np.asarray([int(int(value) in truth) for value in products], dtype=np.int8)
            positives = np.flatnonzero(labels)
            negatives = np.flatnonzero(labels == 0)
            hard = negatives[:negative_limit]
            if len(negatives) > negative_limit:
                source_choices = []
                for source in range(len(SOURCE_NAMES)):
                    source_values = negatives[flags[negatives, source] > 0]
                    source_choices.extend(source_values[:8].tolist())
                ordered = list(dict.fromkeys(source_choices + hard.tolist()))[:negative_limit]
                hard = np.asarray(ordered, dtype=np.int64)
            chosen = np.sort(np.concatenate([positives, hard]))
            if len(positives) and len(chosen):
                values = builder.features(int(customer), context, products[chosen], flags[chosen], ranks[chosen], heuristic[chosen], co_extra, semantic_scores, None if long_small is None else long_small[row], None if recent_small is None else recent_small[row])
                feature_parts.append(values)
                label_parts.append(labels[chosen])
                groups.append(len(chosen))
                candidates.append(products)
                kept_truths.append(truth)
        else:
            values = builder.features(int(customer), context, products, flags, ranks, heuristic, co_extra, semantic_scores, None if long_small is None else long_small[row], None if recent_small is None else recent_small[row])
            feature_parts.append(values)
            label_parts.append(np.asarray([int(int(value) in truth) for value in products], dtype=np.int8))
            groups.append(len(products))
            candidates.append(products)
            kept_truths.append(truth)
        if (row + 1) % 2000 == 0:
            print(f"[candidates] cutoff={snapshot.cutoff.date()} rows={row + 1}/{len(customers)} recall800={hits800 / max(truth_total, 1):.5f} elapsed={elapsed(start)}", flush=True)
    features = np.concatenate(feature_parts) if feature_parts else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    labels = np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int8)
    strata_out = {name: {"rows": value[0], "truth": value[1], "recall800": value[2] / max(value[1], 1)} for name, value in strata.items()}
    print(f"[candidates] complete cutoff={snapshot.cutoff.date()} seeds={len(seeds)} feature_rows={len(features)} recall200={hits200 / max(truth_total, 1):.6f} recall800={hits800 / max(truth_total, 1):.6f} no_semantic={hits800_no_semantic / max(truth_total, 1):.6f} strata={json.dumps(strata_out, sort_keys=True)} elapsed={elapsed(start)}", flush=True)
    return RankingData(features, labels, np.asarray(groups, dtype=np.int32), candidates, kept_truths, hits200 / max(truth_total, 1), hits800 / max(truth_total, 1), hits800_no_semantic / max(truth_total, 1), strata_out)


def train_ranker(data: RankingData, rounds: int, validation: RankingData | None = None) -> lgb.Booster:
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "lambdarank_truncation_level": 10,
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "num_threads": 11,
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
        "deterministic": True,
        "force_col_wise": True,
    }
    train_set = lgb.Dataset(data.features, label=data.labels, group=data.groups, feature_name=list(FEATURE_NAMES), free_raw_data=False)
    valid_sets = None
    callbacks = [lgb.log_evaluation(0)]
    if validation is not None:
        valid_set = lgb.Dataset(validation.features, label=validation.labels, group=validation.groups, feature_name=list(FEATURE_NAMES), reference=train_set, free_raw_data=False)
        valid_sets = [valid_set]
        callbacks.append(lgb.early_stopping(60, first_metric_only=True, verbose=False))
    return lgb.train(params, train_set, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)


def blended_order(model_scores: np.ndarray, group: int, blend_weight: float) -> np.ndarray:
    model_order = np.argsort(model_scores, kind="stable")[::-1]
    model_rank = np.empty(group, dtype=np.float32)
    model_rank[model_order] = 1.0 - np.arange(group, dtype=np.float32) / max(group, 1)
    heuristic_rank = 1.0 - np.arange(group, dtype=np.float32) / max(group, 1)
    blend = blend_weight * model_rank + (1.0 - blend_weight) * heuristic_rank
    return np.argsort(blend, kind="stable")[::-1]


def ranking_map(model: lgb.Booster, data: RankingData, blend_weight: float = 1.0) -> float:
    scores = model.predict(data.features, num_iteration=model.best_iteration or model.current_iteration())
    offset = 0
    values = []
    candidate_index = 0
    for group in data.groups:
        group = int(group)
        products = data.candidates[candidate_index]
        labels = data.labels[offset:offset + group]
        local_scores = scores[offset:offset + group]
        order = blended_order(local_scores, group, blend_weight)[:10]
        truth_size = len(data.truths[candidate_index])
        hits = labels[order]
        precision = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        values.append(float(np.sum(precision * hits) / min(max(truth_size, 1), 10)))
        offset += group
        candidate_index += 1
    return float(np.mean(values)) if values else 0.0


def infer_split(catalog: Catalog, snapshot: Snapshot, semantic_index: SemanticIndex | None, seeds: pd.DataFrame, model: lgb.Booster, semantic_enabled: bool, truths: list | None, collect_training: bool, chunk_size: int, blend_weight: float) -> tuple[np.ndarray, list[RankingData]]:
    output = np.empty((len(seeds), 10), dtype=np.int64)
    collected = []
    empty_truths = [[] for _ in range(chunk_size)]
    for start in range(0, len(seeds), chunk_size):
        stop = min(start + chunk_size, len(seeds))
        chunk = seeds.iloc[start:stop].reset_index(drop=True)
        chunk_truths = truths[start:stop] if truths is not None else empty_truths[:stop - start]
        data = build_ranking_data(catalog, snapshot, semantic_index, chunk, chunk_truths, semantic_enabled, 96, False)
        scores = model.predict(data.features, num_iteration=model.best_iteration or model.current_iteration())
        offset = 0
        for row, (products, group) in enumerate(zip(data.candidates, data.groups)):
            group = int(group)
            local = scores[offset:offset + group]
            order = blended_order(local, group, blend_weight)
            ranked = products[order]
            chosen = []
            chosen_set = set()
            for product in np.concatenate([ranked, snapshot.global_padding]):
                value = int(product)
                if value not in chosen_set:
                    chosen_set.add(value)
                    chosen.append(value)
                    if len(chosen) == 10:
                        break
            output[start + row] = chosen
            offset += group
        if collect_training and truths is not None:
            sampled = sample_inference_data(data, 96)
            collected.append(sampled)
        print(f"[inference] cutoff={snapshot.cutoff.date()} rows={stop}/{len(seeds)} elapsed_chunk_complete", flush=True)
        del data, scores
        gc.collect()
    return output, collected


def sample_inference_data(data: RankingData, negative_limit: int) -> RankingData:
    feature_parts = []
    label_parts = []
    groups = []
    candidates = []
    truths = []
    offset = 0
    for index, group in enumerate(data.groups):
        group = int(group)
        labels = data.labels[offset:offset + group]
        positives = np.flatnonzero(labels)
        negatives = np.flatnonzero(labels == 0)[:negative_limit]
        chosen = np.sort(np.concatenate([positives, negatives]))
        if len(positives):
            feature_parts.append(data.features[offset:offset + group][chosen])
            label_parts.append(labels[chosen])
            groups.append(len(chosen))
            candidates.append(data.candidates[index][chosen])
            truths.append(data.truths[index])
        offset += group
    return RankingData(np.concatenate(feature_parts) if feature_parts else np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), np.concatenate(label_parts) if label_parts else np.empty(0, dtype=np.int8), np.asarray(groups, dtype=np.int32), candidates, truths, data.recall200, data.recall800, data.recall800_no_semantic, data.strata)


def combine_data(parts: list[RankingData]) -> RankingData:
    return RankingData(
        features=np.concatenate([part.features for part in parts]),
        labels=np.concatenate([part.labels for part in parts]),
        groups=np.concatenate([part.groups for part in parts]),
        candidates=sum((part.candidates for part in parts), []),
        truths=sum((part.truths for part in parts), []),
        recall200=float(np.mean([part.recall200 for part in parts])),
        recall800=float(np.mean([part.recall800 for part in parts])),
        recall800_no_semantic=float(np.mean([part.recall800_no_semantic for part in parts])),
        strata={},
    )


def validate_predictions(values: np.ndarray, expected_rows: int) -> None:
    if values.shape != (expected_rows, 10):
        raise RuntimeError(f"prediction shape {values.shape} != {(expected_rows, 10)}")
    if not np.issubdtype(values.dtype, np.integer):
        raise RuntimeError(f"prediction dtype {values.dtype} is not integer")
    if values.min() < 0 or values.max() >= N_PRODUCTS:
        raise RuntimeError("prediction identifier outside catalog range")
    if np.any(np.diff(np.sort(values, axis=1), axis=1) == 0):
        raise RuntimeError("duplicate product identifiers within prediction row")
