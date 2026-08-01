from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb


NUM_CLASSES = 46
ROLE_COLUMNS = ("SOLDTOPARTY", "SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY")
HEADER_COLUMNS = (
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
)


@dataclass
class GroupKeys:
    name: str
    n_keys: int
    kind: str
    values: np.ndarray
    indptr: np.ndarray | None = None

    def block(self, rows: np.ndarray) -> np.ndarray | list[np.ndarray]:
        if self.kind == "single":
            return self.values[rows]
        if self.kind == "multi":
            return [np.unique(self.values[row]) for row in rows]
        return [self.values[self.indptr[row] : self.indptr[row + 1]] for row in rows]


@dataclass
class RelationalBundle:
    split_rows: dict[str, np.ndarray]
    split_labels: dict[str, np.ndarray]
    timestamps_ns: np.ndarray
    days: np.ndarray
    months: np.ndarray
    static_features: sp.csr_matrix
    groups: list[GroupKeys]
    static_feature_count: int


class GroupHistory:
    def __init__(self, n_keys: int, short_decay: int, long_decay: int) -> None:
        shape = (n_keys, NUM_CLASSES)
        self.short_decay = short_decay
        self.long_decay = long_decay
        self.raw = np.zeros(shape, dtype=np.float32)
        self.short = np.zeros(shape, dtype=np.float32)
        self.long = np.zeros(shape, dtype=np.float32)
        self.last_probs = np.zeros(shape, dtype=np.float32)
        self.last_day = np.full(n_keys, -100000, dtype=np.int32)

    def selected(self, block_keys: np.ndarray | list[np.ndarray], day: int) -> tuple[np.ndarray, ...]:
        if isinstance(block_keys, np.ndarray):
            keys = block_keys.astype(np.int64, copy=False)
            raw = self.raw[keys].copy()
            ages = np.maximum(day - self.last_day[keys], 0).astype(np.float32)
            present = self.last_day[keys] > -10000
            short = self.short[keys] * np.exp(-ages[:, None] / float(self.short_decay))
            long = self.long[keys] * np.exp(-ages[:, None] / float(self.long_decay))
            last = self.last_probs[keys].copy()
            ages[~present] = 9999.0
            last[~present] = 0.0
            return raw, short, long, last, ages
        n_rows = len(block_keys)
        raw = np.zeros((n_rows, NUM_CLASSES), dtype=np.float32)
        short = np.zeros_like(raw)
        long = np.zeros_like(raw)
        last = np.zeros_like(raw)
        ages_out = np.full(n_rows, 9999.0, dtype=np.float32)
        for index, keys_value in enumerate(block_keys):
            keys = np.asarray(keys_value, dtype=np.int64)
            if keys.size == 0:
                continue
            key_days = self.last_day[keys]
            present = key_days > -10000
            if not np.any(present):
                continue
            keys = keys[present]
            ages = np.maximum(day - self.last_day[keys], 0).astype(np.float32)
            raw[index] = self.raw[keys].sum(axis=0)
            short[index] = (self.short[keys] * np.exp(-ages[:, None] / float(self.short_decay))).sum(axis=0)
            long[index] = (self.long[keys] * np.exp(-ages[:, None] / float(self.long_decay))).sum(axis=0)
            weights = np.exp(-ages / 45.0)
            last[index] = (self.last_probs[keys] * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-8)
            ages_out[index] = float(ages.min())
        return raw, short, long, last, ages_out

    def update(self, block_keys: np.ndarray | list[np.ndarray], values: np.ndarray, day: int) -> None:
        if isinstance(block_keys, np.ndarray):
            keys = block_keys.astype(np.int64, copy=False)
            expanded = values
        else:
            key_parts = []
            value_parts = []
            for index, keys_value in enumerate(block_keys):
                keys_row = np.asarray(keys_value, dtype=np.int64)
                if keys_row.size:
                    key_parts.append(keys_row)
                    value_parts.append(np.repeat(values[index : index + 1], keys_row.size, axis=0))
            if not key_parts:
                return
            keys = np.concatenate(key_parts)
            expanded = np.concatenate(value_parts, axis=0)
        support = expanded.sum(axis=1)
        keep = support > 0
        if not np.any(keep):
            return
        keys = keys[keep]
        expanded = expanded[keep]
        unique_keys, inverse = np.unique(keys, return_inverse=True)
        additions = np.zeros((len(unique_keys), NUM_CLASSES), dtype=np.float32)
        np.add.at(additions, inverse, expanded)
        prior_days = self.last_day[unique_keys]
        ages = np.maximum(day - prior_days, 0).astype(np.float32)
        present = prior_days > -10000
        short_factor = np.where(present, np.exp(-ages / float(self.short_decay)), 0.0).astype(np.float32)
        long_factor = np.where(present, np.exp(-ages / float(self.long_decay)), 0.0).astype(np.float32)
        self.short[unique_keys] *= short_factor[:, None]
        self.long[unique_keys] *= long_factor[:, None]
        self.raw[unique_keys] += additions
        self.short[unique_keys] += additions
        self.long[unique_keys] += additions
        addition_support = additions.sum(axis=1, keepdims=True)
        self.last_probs[unique_keys] = additions / np.maximum(addition_support, 1e-8)
        self.last_day[unique_keys] = day


class HistoryState:
    def __init__(self, groups: list[GroupKeys], short_decay: int, long_decay: int) -> None:
        self.groups = [GroupHistory(group.n_keys, short_decay, long_decay) for group in groups]
        self.short_decay = short_decay
        self.long_decay = long_decay

    def block_values(self, group_index: int, block_keys: np.ndarray | list[np.ndarray], day: int) -> tuple[np.ndarray, ...]:
        return self.groups[group_index].selected(block_keys, day)

    def update(self, group_blocks: list[np.ndarray | list[np.ndarray]], values: np.ndarray, day: int) -> None:
        for group_index, block_keys in enumerate(group_blocks):
            self.groups[group_index].update(block_keys, values, day)


@dataclass(frozen=True)
class RollforwardConfig:
    block_days: int
    weight_function: str
    blend_strength: float
    enabled: bool


def _hash_values(values: pd.Series | np.ndarray) -> np.ndarray:
    series = pd.Series(values, copy=False).astype("string").fillna("__NA__")
    return pd.util.hash_pandas_object(series, index=False, categorize=True).to_numpy(dtype=np.uint64)


def _combine_hashes(parts: list[np.ndarray]) -> np.ndarray:
    result = np.full(len(parts[0]), np.uint64(1469598103934665603), dtype=np.uint64)
    for index, part in enumerate(parts):
        result ^= part + np.uint64(1099511628211 * (index + 1))
        result *= np.uint64(1099511628211)
    return result


def _factorize_hash(values: np.ndarray) -> tuple[np.ndarray, int]:
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int32), int(codes.max()) + 1


def _artifact_register(cache_root: Path, artifact_path: Path) -> None:
    registry = cache_root / "artifacts.json"
    lock_path = cache_root / "artifacts.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        else:
            records = []
        name = "lane3-sales-shipcond-relational-bundle-v2"
        if not any(record.get("name") == name for record in records):
            records.append(
                {
                    "name": name,
                    "path": str(artifact_path.relative_to(cache_root)),
                    "description": "Allowed-header, item, party-geography, hashed static features and causal key indices",
                    "content_key": "rel-salt-sales-shipcond-lane3-rollforward-v2",
                    "rebuild_hint": "Delete the lane3_rollforward_v2 directory and run main.py again",
                }
            )
            temporary = registry.with_suffix(f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def load_or_build_bundle(task, dataset, cache_root: Path) -> RelationalBundle:
    cache_dir = cache_root / "lane3_rollforward_v2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "bundle.joblib"
    if cache_path.exists():
        bundle = joblib.load(cache_path)
        print(f"[features] loaded relational bundle rows={len(bundle.days)} static={bundle.static_features.shape}")
        return bundle
    bundle = build_bundle(task, dataset)
    temporary = cache_dir / f"bundle.{os.getpid()}.tmp"
    joblib.dump(bundle, temporary, compress=0)
    os.replace(temporary, cache_path)
    _artifact_register(cache_root, cache_path)
    print(f"[features] cached relational bundle rows={len(bundle.days)} static={bundle.static_features.shape}")
    return bundle


def build_bundle(task, dataset) -> RelationalBundle:
    train_table = task.get_table("train", mask_input_cols=False).df
    val_table = task.get_table("val", mask_input_cols=False).df
    test_table = task.get_table("test").df
    split_tables = {"train": train_table, "val": val_table, "test": test_table}
    split_rows = {}
    split_labels = {
        "train": train_table[task.target_col].to_numpy(dtype=np.int32),
        "val": val_table[task.target_col].to_numpy(dtype=np.int32),
    }
    master_ids_parts = []
    offset = 0
    for name in ("train", "val", "test"):
        table = split_tables[name]
        rows = np.arange(offset, offset + len(table), dtype=np.int32)
        split_rows[name] = rows
        master_ids_parts.append(table[task.entity_col].to_numpy(dtype=np.int64))
        offset += len(table)
    master_ids = np.concatenate(master_ids_parts)
    if len(np.unique(master_ids)) != len(master_ids):
        raise RuntimeError("seed document identifiers are not unique")
    db = dataset.get_db(upto_test_timestamp=False)
    header_columns = ["SALESDOCUMENT", "CREATIONTIMESTAMP", *HEADER_COLUMNS]
    header = db.table_dict["salesdocument"].df[header_columns].set_index("SALESDOCUMENT")
    documents = header.loc[master_ids].reset_index()
    timestamps = pd.to_datetime(documents["CREATIONTIMESTAMP"])
    task_timestamps = pd.concat(
        [pd.to_datetime(split_tables[name][task.time_col]) for name in ("train", "val", "test")],
        ignore_index=True,
    )
    if not np.array_equal(timestamps.to_numpy(), task_timestamps.to_numpy()):
        raise RuntimeError("task and header timestamps are not aligned")
    items_columns = [
        "SALESDOCUMENT",
        "SALESDOCUMENTITEM",
        "SALESDOCUMENTITEMCATEGORY",
        "PRODUCT",
        *ROLE_COLUMNS,
        "CREATIONTIMESTAMP",
    ]
    items = db.table_dict["salesdocumentitem"].df[items_columns]
    document_index = pd.Series(np.arange(len(master_ids), dtype=np.int32), index=master_ids)
    item_rows = document_index.reindex(items["SALESDOCUMENT"].to_numpy()).to_numpy()
    keep = ~pd.isna(item_rows)
    items = items.loc[keep].copy()
    item_rows = item_rows[keep].astype(np.int32)
    item_times = pd.to_datetime(items["CREATIONTIMESTAMP"]).to_numpy()
    allowed = item_times <= timestamps.to_numpy()[item_rows]
    items = items.loc[allowed].copy()
    item_rows = item_rows[allowed]
    items["_document_row"] = item_rows
    first_items = items.groupby("_document_row", sort=False).first().reindex(np.arange(len(master_ids)))
    if first_items[list(ROLE_COLUMNS)].isna().any().any():
        raise RuntimeError("at least one seed document has no causal item row")
    item_count = np.bincount(item_rows, minlength=len(master_ids)).astype(np.float32)
    product_hash_all = _hash_values(items["PRODUCT"])
    product_unique_hash, product_codes_all = np.unique(product_hash_all, return_inverse=True)
    product_pairs = np.stack([item_rows, product_codes_all.astype(np.int32)], axis=1)
    product_pairs = np.unique(product_pairs, axis=0)
    product_pairs = product_pairs[np.argsort(product_pairs[:, 0], kind="stable")]
    product_rows = product_pairs[:, 0].astype(np.int32)
    product_codes = product_pairs[:, 1].astype(np.int32)
    product_indptr = np.zeros(len(master_ids) + 1, dtype=np.int64)
    product_indptr[1:] = np.cumsum(np.bincount(product_rows, minlength=len(master_ids)))
    unique_products = np.diff(product_indptr).astype(np.float32)
    category_hash_all = _hash_values(items["SALESDOCUMENTITEMCATEGORY"])
    category_pairs = np.unique(np.stack([item_rows, (category_hash_all % 64).astype(np.int32)], axis=1), axis=0)
    unique_categories = np.bincount(category_pairs[:, 0], minlength=len(master_ids)).astype(np.float32)
    role_values = first_items[list(ROLE_COLUMNS)].to_numpy(dtype=np.int64)
    customer = db.table_dict["customer"].df[["CUSTOMER", "ADDRESSID"]]
    address = db.table_dict["address"].df[["ADDRESSID", "COUNTRY", "REGION"]]
    geography = customer.merge(address, on="ADDRESSID", how="left").set_index("CUSTOMER")
    countries = []
    regions = []
    for role_index in range(len(ROLE_COLUMNS)):
        joined = geography.reindex(role_values[:, role_index])
        countries.append(joined["COUNTRY"].astype("string").fillna("__NA__").to_numpy())
        regions.append(joined["REGION"].astype("string").fillna("__NA__").to_numpy())
    country_values = np.stack(countries, axis=1)
    region_values = np.stack(regions, axis=1)
    party_unique, party_inverse = np.unique(role_values.reshape(-1), return_inverse=True)
    party_codes = party_inverse.reshape(role_values.shape).astype(np.int32)
    document_type_hash = _hash_values(documents["SALESDOCUMENTTYPE"])
    document_type_codes, document_type_count = _factorize_hash(document_type_hash)
    party_document_codes = party_codes.astype(np.int64) * document_type_count + document_type_codes[:, None]
    party_document_codes = party_document_codes.astype(np.int32)
    commercial_hash = _combine_hashes([_hash_values(documents[column]) for column in HEADER_COLUMNS if column != "ORGANIZATIONDIVISION"])
    commercial_codes, commercial_count = _factorize_hash(commercial_hash)
    geography_hash = _combine_hashes(
        [
            _hash_values(country_values[:, 1]),
            _hash_values(region_values[:, 1]),
            _hash_values(documents["SALESORGANIZATION"]),
            document_type_hash,
        ]
    )
    geography_codes, geography_count = _factorize_hash(geography_hash)
    groups = [
        GroupKeys("sold_party", len(party_unique), "single", party_codes[:, 0]),
        GroupKeys("ship_party", len(party_unique), "single", party_codes[:, 1]),
        GroupKeys("bill_party", len(party_unique), "single", party_codes[:, 2]),
        GroupKeys("payer_party", len(party_unique), "single", party_codes[:, 3]),
        GroupKeys("party_document_type", len(party_unique) * document_type_count, "multi", party_document_codes),
        GroupKeys("product", len(product_unique_hash), "csr", product_codes, product_indptr),
        GroupKeys("commercial_cohort", commercial_count, "single", commercial_codes),
        GroupKeys("geography_cohort", geography_count, "single", geography_codes),
    ]
    static_features = build_static_features(
        documents,
        timestamps,
        first_items,
        item_count,
        unique_products,
        unique_categories,
        role_values,
        country_values,
        region_values,
        product_rows,
        product_codes,
        product_unique_hash,
        category_pairs,
    )
    timestamps_ns = timestamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    days = timestamps.to_numpy(dtype="datetime64[D]").astype(np.int64).astype(np.int32)
    months = (timestamps.dt.year.to_numpy() * 100 + timestamps.dt.month.to_numpy()).astype(np.int32)
    return RelationalBundle(
        split_rows=split_rows,
        split_labels=split_labels,
        timestamps_ns=timestamps_ns,
        days=days,
        months=months,
        static_features=static_features,
        groups=groups,
        static_feature_count=static_features.shape[1],
    )


def build_static_features(
    documents: pd.DataFrame,
    timestamps: pd.Series,
    first_items: pd.DataFrame,
    item_count: np.ndarray,
    unique_products: np.ndarray,
    unique_categories: np.ndarray,
    role_values: np.ndarray,
    country_values: np.ndarray,
    region_values: np.ndarray,
    product_rows: np.ndarray,
    product_codes: np.ndarray,
    product_unique_hash: np.ndarray,
    category_pairs: np.ndarray,
) -> sp.csr_matrix:
    n_rows = len(documents)
    day_fraction = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60.0
    day_of_year = timestamps.dt.dayofyear.to_numpy()
    numerical = np.column_stack(
        [
            (timestamps.dt.year.to_numpy() - 2018).astype(np.float32),
            np.sin(2 * np.pi * timestamps.dt.month.to_numpy() / 12.0),
            np.cos(2 * np.pi * timestamps.dt.month.to_numpy() / 12.0),
            np.sin(2 * np.pi * timestamps.dt.dayofweek.to_numpy() / 7.0),
            np.cos(2 * np.pi * timestamps.dt.dayofweek.to_numpy() / 7.0),
            np.sin(2 * np.pi * day_of_year / 365.25),
            np.cos(2 * np.pi * day_of_year / 365.25),
            np.sin(2 * np.pi * day_fraction / 24.0),
            np.cos(2 * np.pi * day_fraction / 24.0),
            np.log1p(item_count),
            np.log1p(unique_products),
            np.log1p(unique_categories),
            item_count / np.maximum(unique_products, 1.0),
        ]
    ).astype(np.float32)
    role_diversity = np.array([len(np.unique(row)) for row in role_values], dtype=np.float32)
    role_equalities = np.column_stack(
        [
            (role_values[:, left] == role_values[:, right]).astype(np.float32)
            for left in range(len(ROLE_COLUMNS))
            for right in range(left + 1, len(ROLE_COLUMNS))
        ]
    )
    numerical = np.column_stack([numerical, role_diversity, role_equalities])
    blocks = [sp.csr_matrix(numerical)]
    onehot_specs: list[tuple[np.ndarray, int]] = []
    sizes = {
        "SALESDOCUMENTTYPE": 32,
        "SALESORGANIZATION": 64,
        "DISTRIBUTIONCHANNEL": 8,
        "ORGANIZATIONDIVISION": 8,
        "BILLINGCOMPANYCODE": 64,
        "TRANSACTIONCURRENCY": 64,
    }
    for column in HEADER_COLUMNS:
        onehot_specs.append((_hash_values(documents[column]), sizes[column]))
    onehot_specs.append((_hash_values(first_items["SALESDOCUMENTITEMCATEGORY"]), 64))
    for role_index in range(len(ROLE_COLUMNS)):
        onehot_specs.append((_hash_values(role_values[:, role_index]), 4096))
        onehot_specs.append((_hash_values(country_values[:, role_index]), 256))
        onehot_specs.append((_hash_values(region_values[:, role_index]), 1024))
    onehot_specs.append(
        (
            _combine_hashes(
                [
                    _hash_values(documents["SALESDOCUMENTTYPE"]),
                    _hash_values(documents["SALESORGANIZATION"]),
                    _hash_values(documents["DISTRIBUTIONCHANNEL"]),
                ]
            ),
            1024,
        )
    )
    categorical_rows = []
    categorical_columns = []
    offset = 0
    row_index = np.arange(n_rows, dtype=np.int32)
    for hashes, size in onehot_specs:
        categorical_rows.append(row_index)
        categorical_columns.append(offset + (hashes % size).astype(np.int32))
        offset += size
    categorical = sp.csr_matrix(
        (
            np.ones(sum(len(value) for value in categorical_rows), dtype=np.float32),
            (np.concatenate(categorical_rows), np.concatenate(categorical_columns)),
        ),
        shape=(n_rows, offset),
    )
    blocks.append(categorical)
    product_buckets = (product_unique_hash[product_codes] % 16384).astype(np.int32)
    product_values = 1.0 / np.sqrt(np.maximum(unique_products[product_rows], 1.0))
    product_matrix = sp.csr_matrix(
        (product_values.astype(np.float32), (product_rows, product_buckets)),
        shape=(n_rows, 16384),
    )
    product_matrix.sum_duplicates()
    blocks.append(product_matrix)
    category_matrix = sp.csr_matrix(
        (
            np.ones(len(category_pairs), dtype=np.float32),
            (category_pairs[:, 0], category_pairs[:, 1]),
        ),
        shape=(n_rows, 64),
    )
    blocks.append(category_matrix)
    result = sp.hstack(blocks, format="csr", dtype=np.float32)
    result.sort_indices()
    return result


def group_blocks(bundle: RelationalBundle, rows: np.ndarray) -> list[np.ndarray | list[np.ndarray]]:
    return [group.block(rows) for group in bundle.groups]


def posterior_features(counts: np.ndarray, top_classes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support = counts.sum(axis=1)
    denominator = support + 2.0
    top = (counts[:, top_classes] + 2.0 / NUM_CLASSES) / denominator[:, None]
    probabilities = (counts + 2.0 / NUM_CLASSES) / denominator[:, None]
    confidence = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(axis=1) / math.log(NUM_CLASSES)
    return top.astype(np.float32), support.astype(np.float32), confidence.astype(np.float32), entropy.astype(np.float32)


def dynamic_features(
    bundle: RelationalBundle,
    rows: np.ndarray,
    day: int,
    true_history: HistoryState,
    pseudo_history: HistoryState | None,
    blend_strength: float,
    top_classes: np.ndarray,
) -> np.ndarray:
    blocks = group_blocks(bundle, rows)
    feature_parts = []
    for group_index, block_keys in enumerate(blocks):
        true_raw, true_short, true_long, true_last, true_age = true_history.block_values(group_index, block_keys, day)
        if pseudo_history is None:
            pseudo_raw = np.zeros_like(true_raw)
            pseudo_short = np.zeros_like(true_short)
            pseudo_long = np.zeros_like(true_long)
            pseudo_last = np.zeros_like(true_last)
            pseudo_age = np.full(len(rows), 9999.0, dtype=np.float32)
        else:
            pseudo_raw, pseudo_short, pseudo_long, pseudo_last, pseudo_age = pseudo_history.block_values(group_index, block_keys, day)
        raw = true_raw + blend_strength * pseudo_raw
        short = true_short + blend_strength * pseudo_short
        long = true_long + blend_strength * pseudo_long
        raw_top, raw_support, raw_confidence, raw_entropy = posterior_features(raw, top_classes)
        short_top, short_support, short_confidence, short_entropy = posterior_features(short, top_classes)
        long_top, long_support, long_confidence, long_entropy = posterior_features(long, top_classes)
        true_last_weight = np.exp(-np.minimum(true_age, 9999.0) / 180.0)
        pseudo_last_weight = blend_strength * np.exp(-np.minimum(pseudo_age, 9999.0) / 45.0)
        last_denominator = true_last_weight + pseudo_last_weight
        last = (
            true_last * true_last_weight[:, None] + pseudo_last * pseudo_last_weight[:, None]
        ) / np.maximum(last_denominator[:, None], 1e-8)
        pseudo_support = pseudo_long.sum(axis=1)
        _, _, _, pseudo_entropy = posterior_features(pseudo_long, top_classes)
        summaries = np.column_stack(
            [
                np.log1p(raw_support),
                np.log1p(short_support),
                np.log1p(long_support),
                raw_confidence,
                short_confidence,
                long_confidence,
                raw_entropy,
                short_entropy,
                long_entropy,
                np.log1p(np.minimum(true_age, 9999.0)),
                np.log1p(pseudo_support),
                pseudo_entropy,
                np.log1p(np.minimum(pseudo_age, 9999.0)),
            ]
        ).astype(np.float32)
        feature_parts.extend([raw_top, short_top, long_top, last[:, top_classes].astype(np.float32), summaries])
    return np.concatenate(feature_parts, axis=1).astype(np.float32)


def one_hot_labels(labels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(labels), NUM_CLASSES), dtype=np.float32)
    result[np.arange(len(labels)), labels.astype(np.int64)] = 1.0
    return result


def make_labeled_features(
    bundle: RelationalBundle,
    rows: np.ndarray,
    labels: np.ndarray,
    top_classes: np.ndarray,
    true_history: HistoryState | None = None,
) -> tuple[np.ndarray, HistoryState]:
    if true_history is None:
        true_history = HistoryState(bundle.groups, 60, 365)
    order = np.lexsort((rows, bundle.days[rows]))
    sorted_rows = rows[order]
    sorted_labels = labels[order]
    feature_count = len(bundle.groups) * (4 * len(top_classes) + 13)
    result_sorted = np.empty((len(rows), feature_count), dtype=np.float32)
    start = 0
    while start < len(rows):
        day = int(bundle.days[sorted_rows[start]])
        end = start + 1
        while end < len(rows) and bundle.days[sorted_rows[end]] == day:
            end += 1
        day_rows = sorted_rows[start:end]
        result_sorted[start:end] = dynamic_features(bundle, day_rows, day, true_history, None, 0.0, top_classes)
        true_history.update(group_blocks(bundle, day_rows), one_hot_labels(sorted_labels[start:end]), day)
        start = end
    result = np.empty_like(result_sorted)
    result[order] = result_sorted
    return result, true_history


def update_true_history(bundle: RelationalBundle, rows: np.ndarray, labels: np.ndarray) -> HistoryState:
    state = HistoryState(bundle.groups, 60, 365)
    order = np.lexsort((rows, bundle.days[rows]))
    sorted_rows = rows[order]
    sorted_labels = labels[order]
    start = 0
    while start < len(rows):
        day = int(bundle.days[sorted_rows[start]])
        end = start + 1
        while end < len(rows) and bundle.days[sorted_rows[end]] == day:
            end += 1
        day_rows = sorted_rows[start:end]
        state.update(group_blocks(bundle, day_rows), one_hot_labels(sorted_labels[start:end]), day)
        start = end
    return state


def assemble_matrix(bundle: RelationalBundle, rows: np.ndarray, dynamic: np.ndarray) -> sp.csr_matrix:
    matrix = sp.hstack([bundle.static_features[rows], sp.csr_matrix(dynamic)], format="csr", dtype=np.float32)
    matrix.sort_indices()
    return matrix


def xgboost_parameters(seed: int) -> dict:
    return {
        "objective": "multi:softprob",
        "num_class": NUM_CLASSES,
        "max_depth": 8,
        "eta": 0.07,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "min_child_weight": 8,
        "tree_method": "hist",
        "device": "cuda",
        "max_bin": 256,
        "eval_metric": "merror",
        "seed": seed,
        "nthread": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "verbosity": 0,
    }


def train_booster(
    matrix: sp.csr_matrix,
    labels: np.ndarray,
    rounds: int,
    seed: int,
    eval_matrix: sp.csr_matrix | None = None,
    eval_labels: np.ndarray | None = None,
) -> xgb.Booster:
    training = xgb.QuantileDMatrix(matrix, label=labels, max_bin=256)
    evaluations = []
    early_stopping = None
    if eval_matrix is not None:
        evaluation = xgb.QuantileDMatrix(eval_matrix, label=eval_labels, ref=training, max_bin=256)
        evaluations = [(evaluation, "internal_forward")]
        early_stopping = 50
    booster = xgb.train(
        xgboost_parameters(seed),
        training,
        num_boost_round=rounds,
        evals=evaluations,
        early_stopping_rounds=early_stopping,
        verbose_eval=False,
    )
    return booster


def predict_booster(booster: xgb.Booster, matrix: sp.csr_matrix) -> np.ndarray:
    iteration_range = (0, booster.best_iteration + 1) if getattr(booster, "best_iteration", None) is not None else (0, 0)
    prediction = booster.inplace_predict(matrix, iteration_range=iteration_range)
    return np.asarray(prediction, dtype=np.float32)


def reliability_weights(predictions: np.ndarray, function_name: str) -> np.ndarray:
    top = np.max(predictions, axis=1)
    if function_name == "squared_margin":
        weights = np.maximum(0.0, (top - 0.45) / 0.45) ** 2
    elif function_name == "probability_margin":
        partitioned = np.partition(predictions, -2, axis=1)
        margin = partitioned[:, -1] - partitioned[:, -2]
        weights = np.maximum(0.0, (margin - 0.10) / 0.50)
    elif function_name == "hard_confidence":
        weights = np.where(top >= 0.75, 0.5, 0.0)
    else:
        raise ValueError(function_name)
    return np.minimum(weights, 0.5).astype(np.float32)


def predict_window(
    booster: xgb.Booster,
    bundle: RelationalBundle,
    rows: np.ndarray,
    true_history: HistoryState,
    top_classes: np.ndarray,
    config: RollforwardConfig,
) -> np.ndarray:
    _row_id = np.arange(len(rows), dtype=np.int64)
    order = np.lexsort((_row_id, bundle.timestamps_ns[rows]))
    sorted_rows = rows[order]
    sorted_days = bundle.days[sorted_rows]
    predictions_sorted = np.empty((len(rows), NUM_CLASSES), dtype=np.float32)
    pseudo_history = HistoryState(bundle.groups, 30, 90) if config.enabled else None
    first_day = int(sorted_days[0])
    block_ids = (sorted_days - first_day) // max(config.block_days, 1)
    start = 0
    while start < len(rows):
        block_id = block_ids[start]
        end = start + 1
        while end < len(rows) and block_ids[end] == block_id:
            end += 1
        block_rows = sorted_rows[start:end]
        state_day = int(sorted_days[start])
        dynamic = dynamic_features(
            bundle,
            block_rows,
            state_day,
            true_history,
            pseudo_history,
            config.blend_strength if config.enabled else 0.0,
            top_classes,
        )
        matrix = assemble_matrix(bundle, block_rows, dynamic)
        predictions = predict_booster(booster, matrix)
        predictions_sorted[start:end] = predictions
        if config.enabled:
            weights = reliability_weights(predictions, config.weight_function)
            pseudo_history.update(group_blocks(bundle, block_rows), predictions * weights[:, None], int(sorted_days[end - 1]))
        start = end
    restored = np.empty_like(predictions_sorted)
    restored[order] = predictions_sorted
    if not np.all(np.isfinite(restored)):
        raise RuntimeError("rollforward produced non-finite predictions")
    return restored


def simulation_grid(debug: bool) -> list[RollforwardConfig]:
    if debug:
        return [RollforwardConfig(7, "squared_margin", 0.5, True)]
    return [
        RollforwardConfig(1, "squared_margin", 0.5, True),
        RollforwardConfig(7, "squared_margin", 0.5, True),
        RollforwardConfig(1, "probability_margin", 0.5, True),
        RollforwardConfig(1, "hard_confidence", 0.5, True),
        RollforwardConfig(1, "squared_margin", 0.25, True),
        RollforwardConfig(1, "squared_margin", 1.0, True),
    ]


def monthly_accuracy(predictions: np.ndarray, labels: np.ndarray, months: np.ndarray) -> dict[int, float]:
    classes = predictions.argmax(axis=1)
    return {
        int(month): float(np.mean(classes[months == month] == labels[months == month]))
        for month in np.unique(months)
    }


def select_rollforward(
    booster: xgb.Booster,
    bundle: RelationalBundle,
    simulation_rows: np.ndarray,
    simulation_labels: np.ndarray,
    simulation_history: HistoryState,
    top_classes: np.ndarray,
    debug: bool,
) -> tuple[RollforwardConfig, dict]:
    frozen = RollforwardConfig(7 if debug else 1, "squared_margin", 0.0, False)
    months = bundle.months[simulation_rows]
    baseline_by_block = {}
    for block_days in sorted({config.block_days for config in simulation_grid(debug)}):
        baseline_config = RollforwardConfig(block_days, "squared_margin", 0.0, False)
        baseline_prediction = predict_window(booster, bundle, simulation_rows, simulation_history, top_classes, baseline_config)
        baseline_by_block[block_days] = monthly_accuracy(baseline_prediction, simulation_labels, months)
    baseline_monthly = baseline_by_block[frozen.block_days]
    records = []
    chosen = frozen
    chosen_value = -np.inf
    ordered_months = sorted(baseline_monthly)
    for config in simulation_grid(debug):
        candidate_baseline = baseline_by_block[config.block_days]
        prediction = predict_window(booster, bundle, simulation_rows, simulation_history, top_classes, config)
        scores = monthly_accuracy(prediction, simulation_labels, months)
        deltas = {month: scores[month] - candidate_baseline[month] for month in ordered_months}
        mean_delta = float(np.mean(list(deltas.values())))
        late_months = ordered_months[2:5]
        late_delta = float(np.mean([deltas[month] for month in late_months])) if late_months else mean_delta
        month_one_delta = float(deltas[ordered_months[0]])
        qualifies = mean_delta >= 0.003 and late_delta >= 0.003 and month_one_delta >= -0.001
        record = {
            "block_days": config.block_days,
            "weight_function": config.weight_function,
            "blend_strength": config.blend_strength,
            "frozen_monthly_accuracy": candidate_baseline,
            "monthly_accuracy": scores,
            "monthly_delta": deltas,
            "mean_delta": mean_delta,
            "months_3_5_delta": late_delta,
            "month_1_delta": month_one_delta,
            "qualifies": qualifies,
        }
        records.append(record)
        value = mean_delta + late_delta
        if qualifies and value > chosen_value:
            chosen = config
            chosen_value = value
    diagnostics = {
        "frozen_monthly_accuracy": baseline_monthly,
        "frozen_monthly_accuracy_by_block": baseline_by_block,
        "candidates": records,
        "selected": {
            "enabled": chosen.enabled,
            "block_days": chosen.block_days,
            "weight_function": chosen.weight_function,
            "blend_strength": chosen.blend_strength,
        },
    }
    return chosen, diagnostics


def stable_top_classes(train_labels: np.ndarray, count: int = 20) -> np.ndarray:
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    return np.argsort(-counts, kind="stable")[:count].astype(np.int32)


def sample_chronological(rows: np.ndarray, days: np.ndarray, count: int) -> np.ndarray:
    if len(rows) <= count:
        return rows
    positions = np.linspace(0, len(rows) - 1, count, dtype=np.int64)
    order = np.lexsort((rows, days[rows]))
    return rows[order[positions]]


def save_diagnostics(path: Path, diagnostics: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / f"metrics.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
    os.replace(temporary, path / "metrics.json")


def elapsed_message(start_time: float, phase: str) -> None:
    print(f"[timing] {phase}: {time.time() - start_time:.1f}s elapsed")
