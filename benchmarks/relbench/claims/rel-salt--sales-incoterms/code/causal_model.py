from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numba
import numpy as np
import pandas as pd
from numba import njit, prange


# Configuration

SEED = 20260801
N_CLASSES = 13
HALF_LIVES = np.array([30.0, 90.0, 365.0], dtype=np.float64)
TAIL_DAYS = 273.75
ALL_WEIGHT_HALF_LIFE = 180.0
TAIL_WEIGHT_HALF_LIFE = 3650.0
DAY_NS = 86_400_000_000_000.0
BASE_NS = pd.Timestamp("2018-01-01").value
EPS_DAY = 1.0 / 86_400.0
HIERARCHIES = [
    ("sold", ["SOLDTOPARTY"]),
    ("sold_doc_org", ["SOLDTOPARTY", "SALESDOCUMENTTYPE", "SALESORGANIZATION"]),
    ("ship_area", ["SHIPTOPARTY", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION"]),
    ("doc_org_shipcountry", ["SALESDOCUMENTTYPE", "SALESORGANIZATION", "SHIP_COUNTRY"]),
]
RAW_CATEGORICAL = [
    "SALESDOCUMENTTYPE",
    "SALESORGANIZATION",
    "DISTRIBUTIONCHANNEL",
    "ORGANIZATIONDIVISION",
    "BILLINGCOMPANYCODE",
    "TRANSACTIONCURRENCY",
    "SOLDTOPARTY",
    "SHIPTOPARTY",
    "BILLTOPARTY",
    "PAYERPARTY",
    "ITEM_CATEGORY_FIRST",
    "ITEM_CATEGORY_MODE",
    "ITEM_CATEGORY_LAST",
    "PRODUCT_FIRST",
    "PRODUCT_MODE",
    "PRODUCT_LAST",
    "SOLD_COUNTRY",
    "SOLD_REGION",
    "SHIP_COUNTRY",
    "SHIP_REGION",
    "BILL_COUNTRY",
    "BILL_REGION",
    "PAYER_COUNTRY",
    "PAYER_REGION",
]
BASE_NUMERIC = [
    "ITEM_COUNT",
    "UNIQUE_ITEM_CATEGORIES",
    "UNIQUE_PRODUCTS",
    "ITEM_CATEGORY_MODE_SHARE",
    "PRODUCT_MODE_SHARE",
    "ITEM_POSITION_MIN",
    "ITEM_POSITION_MAX",
]


# Kernels

@njit(cache=True, parallel=True)
def lookup_last_positions(unique_keys, starts, ends, sorted_times, query_keys, query_times):
    n = len(query_keys)
    positions = np.full(n, -1, dtype=np.int64)
    group_starts = np.full(n, -1, dtype=np.int64)
    for i in prange(n):
        key = query_keys[i]
        g = np.searchsorted(unique_keys, key)
        if g < len(unique_keys) and unique_keys[g] == key:
            lo = starts[g]
            hi = ends[g]
            p = np.searchsorted(sorted_times[lo:hi], query_times[i], side="right")
            if p > 0:
                positions[i] = lo + p - 1
            group_starts[i] = lo
    return positions, group_starts


@njit(cache=True, parallel=True)
def build_cumulatives(sorted_labels, sorted_times, starts, ends):
    n = len(sorted_labels)
    counts = np.zeros((n, N_CLASSES), dtype=np.float32)
    decayed = np.zeros((len(HALF_LIVES), n, N_CLASSES), dtype=np.float32)
    for g in prange(len(starts)):
        count_acc = np.zeros(N_CLASSES, dtype=np.float32)
        decay_acc = np.zeros((len(HALF_LIVES), N_CLASSES), dtype=np.float32)
        for i in range(starts[g], ends[g]):
            label = sorted_labels[i]
            count_acc[label] += 1.0
            counts[i] = count_acc
            for h in range(len(HALF_LIVES)):
                decay_acc[h, label] += np.exp(sorted_times[i] / HALF_LIVES[h])
                decayed[h, i] = decay_acc[h]
    return counts, decayed


# Data

def log_phase(start, name, detail=""):
    elapsed = time.time() - start
    suffix = f" {detail}" if detail else ""
    print(f"[causal_dual] phase={name} elapsed={elapsed:.1f}s{suffix}", flush=True)


def path_sql(path):
    return str(path).replace("'", "''")


def safe_frame_query(base):
    header = path_sql(base / "db" / "salesdocument.parquet")
    item = path_sql(base / "db" / "salesdocumentitem.parquet")
    customer = path_sql(base / "db" / "customer.parquet")
    address = path_sql(base / "db" / "address.parquet")
    return f"""
    WITH item_rows AS (
        SELECT SALESDOCUMENT, ID, SALESDOCUMENTITEM, SALESDOCUMENTITEMCATEGORY,
               PRODUCT, SOLDTOPARTY, SHIPTOPARTY, BILLTOPARTY, PAYERPARTY
        FROM read_parquet('{item}')
    ),
    item_basic AS (
        SELECT SALESDOCUMENT, count(*) AS ITEM_COUNT,
               count(DISTINCT SALESDOCUMENTITEMCATEGORY) AS UNIQUE_ITEM_CATEGORIES,
               count(DISTINCT PRODUCT) AS UNIQUE_PRODUCTS,
               arg_min(SALESDOCUMENTITEMCATEGORY, ID) AS ITEM_CATEGORY_FIRST,
               mode(SALESDOCUMENTITEMCATEGORY) AS ITEM_CATEGORY_MODE,
               arg_max(SALESDOCUMENTITEMCATEGORY, ID) AS ITEM_CATEGORY_LAST,
               arg_min(PRODUCT, ID) AS PRODUCT_FIRST,
               mode(PRODUCT) AS PRODUCT_MODE,
               arg_max(PRODUCT, ID) AS PRODUCT_LAST,
               min(SOLDTOPARTY) AS SOLDTOPARTY,
               min(SHIPTOPARTY) AS SHIPTOPARTY,
               min(BILLTOPARTY) AS BILLTOPARTY,
               min(PAYERPARTY) AS PAYERPARTY,
               min(try_cast(SALESDOCUMENTITEM AS INTEGER)) AS ITEM_POSITION_MIN,
               max(try_cast(SALESDOCUMENTITEM AS INTEGER)) AS ITEM_POSITION_MAX
        FROM item_rows GROUP BY SALESDOCUMENT
    ),
    category_share AS (
        SELECT SALESDOCUMENT, max(n) / sum(n) AS ITEM_CATEGORY_MODE_SHARE
        FROM (SELECT SALESDOCUMENT, SALESDOCUMENTITEMCATEGORY, count(*) AS n
              FROM item_rows GROUP BY SALESDOCUMENT, SALESDOCUMENTITEMCATEGORY)
        GROUP BY SALESDOCUMENT
    ),
    product_share AS (
        SELECT SALESDOCUMENT, max(n) / sum(n) AS PRODUCT_MODE_SHARE
        FROM (SELECT SALESDOCUMENT, PRODUCT, count(*) AS n
              FROM item_rows GROUP BY SALESDOCUMENT, PRODUCT)
        GROUP BY SALESDOCUMENT
    ),
    geo AS (
        SELECT c.CUSTOMER, a.COUNTRY, a.REGION
        FROM read_parquet('{customer}') c
        LEFT JOIN read_parquet('{address}') a ON c.ADDRESSID = a.ADDRESSID
    )
    SELECT h.SALESDOCUMENT, h.CREATIONTIMESTAMP, h.SALESDOCUMENTTYPE,
           h.SALESORGANIZATION, h.DISTRIBUTIONCHANNEL, h.ORGANIZATIONDIVISION,
           h.BILLINGCOMPANYCODE, h.TRANSACTIONCURRENCY,
           i.ITEM_COUNT, i.UNIQUE_ITEM_CATEGORIES, i.UNIQUE_PRODUCTS,
           i.ITEM_CATEGORY_FIRST, i.ITEM_CATEGORY_MODE, i.ITEM_CATEGORY_LAST,
           i.PRODUCT_FIRST, i.PRODUCT_MODE, i.PRODUCT_LAST,
           i.SOLDTOPARTY, i.SHIPTOPARTY, i.BILLTOPARTY, i.PAYERPARTY,
           i.ITEM_POSITION_MIN, i.ITEM_POSITION_MAX,
           cs.ITEM_CATEGORY_MODE_SHARE, ps.PRODUCT_MODE_SHARE,
           gs.COUNTRY AS SOLD_COUNTRY, gs.REGION AS SOLD_REGION,
           gh.COUNTRY AS SHIP_COUNTRY, gh.REGION AS SHIP_REGION,
           gb.COUNTRY AS BILL_COUNTRY, gb.REGION AS BILL_REGION,
           gp.COUNTRY AS PAYER_COUNTRY, gp.REGION AS PAYER_REGION
    FROM read_parquet('{header}') h
    LEFT JOIN item_basic i ON h.SALESDOCUMENT = i.SALESDOCUMENT
    LEFT JOIN category_share cs ON h.SALESDOCUMENT = cs.SALESDOCUMENT
    LEFT JOIN product_share ps ON h.SALESDOCUMENT = ps.SALESDOCUMENT
    LEFT JOIN geo gs ON i.SOLDTOPARTY = gs.CUSTOMER
    LEFT JOIN geo gh ON i.SHIPTOPARTY = gh.CUSTOMER
    LEFT JOIN geo gb ON i.BILLTOPARTY = gb.CUSTOMER
    LEFT JOIN geo gp ON i.PAYERPARTY = gp.CUSTOMER
    ORDER BY h.SALESDOCUMENT
    """


def load_document_frame(base, shared):
    cache_dir = shared / "lane3_causal_drift"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / "legal_document_frame_v2.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path), True
    connection = duckdb.connect()
    connection.execute("SET threads TO 11")
    connection.execute("SET preserve_insertion_order=false")
    frame = connection.sql(safe_frame_query(base)).df()
    temporary = cache_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, cache_path)
    return frame, False


def hash_category(values, buckets):
    strings = values.astype("string").fillna("__MISSING__")
    hashed = pd.util.hash_pandas_object(strings, index=False, categorize=False).to_numpy(dtype=np.uint64)
    return (hashed % np.uint64(buckets - 1) + np.uint64(1)).astype(np.int32)


def dense_category(values):
    strings = values.astype("string").fillna("__MISSING__")
    codes, _ = pd.factorize(strings, sort=True)
    return (codes + 1).astype(np.int32)


def row_hash(frame, columns):
    block = frame[columns].copy()
    for column in columns:
        block[column] = block[column].astype("string").fillna("__MISSING__")
    return pd.util.hash_pandas_object(block, index=False, categorize=False).to_numpy(dtype=np.uint64)


def time_days(values):
    return (pd.to_datetime(values).astype("int64").to_numpy(dtype=np.float64) - BASE_NS) / DAY_NS


def read_splits(base):
    task_dir = base / "tasks" / "sales-incoterms"
    train = pd.read_parquet(
        task_dir / "train.parquet",
        columns=["CREATIONTIMESTAMP", "SALESDOCUMENT", "HEADERINCOTERMSCLASSIFICATION"],
    )
    validation = pd.read_parquet(
        task_dir / "val.parquet", columns=["CREATIONTIMESTAMP", "SALESDOCUMENT"]
    )
    test = pd.read_parquet(
        task_dir / "test.parquet", columns=["CREATIONTIMESTAMP", "SALESDOCUMENT"]
    )
    train = train.rename(columns={"HEADERINCOTERMSCLASSIFICATION": "label"})
    train["label"] = train["label"].astype(np.int8)
    return train, validation, test


def read_validation_labels(base):
    path = base / "tasks" / "sales-incoterms" / "val.parquet"
    labels = pd.read_parquet(path, columns=["HEADERINCOTERMSCLASSIFICATION"])
    return labels.iloc[:, 0].to_numpy(dtype=np.int8)


class FeatureEngine:
    def __init__(self, frame):
        self.frame = frame
        self.ids = frame["SALESDOCUMENT"].to_numpy(dtype=np.int64)
        self.days = time_days(frame["CREATIONTIMESTAMP"])
        if len(self.ids) == 0 or self.ids[0] != 0 or not np.array_equal(self.ids, np.arange(len(self.ids))):
            self.id_to_position = pd.Series(np.arange(len(self.ids), dtype=np.int64), index=self.ids)
        else:
            self.id_to_position = None
        category_arrays = []
        native_columns = []
        for column in RAW_CATEGORICAL:
            high_cardinality = column.endswith("PARTY") or column.startswith("PRODUCT")
            if high_cardinality:
                buckets = 131071 if column.endswith("PARTY") else 32767
                category_arrays.append(hash_category(frame[column], buckets))
            else:
                category_arrays.append(dense_category(frame[column]))
                native_columns.append(column)
        numeric_arrays = []
        for column in BASE_NUMERIC:
            numeric_arrays.append(pd.to_numeric(frame[column], errors="coerce").fillna(0).to_numpy(dtype=np.float32))
        timestamps = pd.to_datetime(frame["CREATIONTIMESTAMP"])
        numeric_arrays.extend(
            [
                timestamps.dt.year.to_numpy(dtype=np.float32) - 2018.0,
                timestamps.dt.month.to_numpy(dtype=np.float32),
                timestamps.dt.day.to_numpy(dtype=np.float32),
                timestamps.dt.dayofweek.to_numpy(dtype=np.float32),
                timestamps.dt.isocalendar().week.to_numpy(dtype=np.float32),
                timestamps.dt.hour.to_numpy(dtype=np.float32),
                self.days.astype(np.float32),
                (frame["SOLDTOPARTY"].to_numpy() == frame["SHIPTOPARTY"].to_numpy()).astype(np.float32),
                (frame["SOLDTOPARTY"].to_numpy() == frame["BILLTOPARTY"].to_numpy()).astype(np.float32),
                (frame["SOLDTOPARTY"].to_numpy() == frame["PAYERPARTY"].to_numpy()).astype(np.float32),
                (frame["SHIPTOPARTY"].to_numpy() == frame["BILLTOPARTY"].to_numpy()).astype(np.float32),
                (frame["SHIPTOPARTY"].to_numpy() == frame["PAYERPARTY"].to_numpy()).astype(np.float32),
                (frame["BILLTOPARTY"].to_numpy() == frame["PAYERPARTY"].to_numpy()).astype(np.float32),
            ]
        )
        self.base = np.column_stack(category_arrays + numeric_arrays).astype(np.float32)
        self.base_names = RAW_CATEGORICAL + BASE_NUMERIC + [
            "YEAR",
            "MONTH",
            "DAY",
            "DAY_OF_WEEK",
            "ISO_WEEK",
            "HOUR",
            "TIME_TREND",
            "SOLD_EQ_SHIP",
            "SOLD_EQ_BILL",
            "SOLD_EQ_PAYER",
            "SHIP_EQ_BILL",
            "SHIP_EQ_PAYER",
            "BILL_EQ_PAYER",
        ]
        self.base_categorical = [RAW_CATEGORICAL.index(column) for column in native_columns]
        self.keys = {name: row_hash(frame, columns) for name, columns in HIERARCHIES}

    def positions(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        if self.id_to_position is None:
            return ids
        return self.id_to_position.loc[ids].to_numpy(dtype=np.int64)

    def _index(self, keys, days, labels=None):
        order = np.lexsort((days, keys))
        sorted_keys = keys[order]
        sorted_days = days[order].astype(np.float64)
        changes = np.r_[True, sorted_keys[1:] != sorted_keys[:-1]]
        starts = np.flatnonzero(changes).astype(np.int64)
        ends = np.r_[starts[1:], len(order)].astype(np.int64)
        unique_keys = sorted_keys[starts]
        if labels is None:
            return unique_keys, starts, ends, sorted_days, order
        return unique_keys, starts, ends, sorted_days, labels[order], order

    def _history_block(self, name, query_positions, cutoffs, label_positions, labels, query_days):
        history_keys = self.keys[name][label_positions]
        index = self._index(history_keys, self.days[label_positions], labels)
        unique_keys, starts, ends, sorted_days, sorted_labels, _ = index
        positions, group_starts = lookup_last_positions(
            unique_keys, starts, ends, sorted_days, self.keys[name][query_positions], cutoffs
        )
        counts, decayed = build_cumulatives(sorted_labels, sorted_days, starts, ends)
        n = len(query_positions)
        valid = positions >= 0
        last_label = np.zeros(n, dtype=np.float32)
        mode_label = np.zeros(n, dtype=np.float32)
        support = np.zeros(n, dtype=np.float32)
        staleness = np.full(n, 9999.0, dtype=np.float32)
        purity = np.zeros(n, dtype=np.float32)
        switched = np.zeros(n, dtype=np.float32)
        distributions = []
        if valid.any():
            selected_counts = counts[positions[valid]]
            selected_support = selected_counts.sum(axis=1)
            last_label[valid] = sorted_labels[positions[valid]].astype(np.float32) + 1.0
            mode_label[valid] = selected_counts.argmax(axis=1).astype(np.float32) + 1.0
            support[valid] = selected_support
            staleness[valid] = np.maximum(0.0, query_days[valid] - sorted_days[positions[valid]])
            purity[valid] = selected_counts.max(axis=1) / np.maximum(selected_support, 1.0)
            previous = positions[valid] - 1
            can_switch = previous >= group_starts[valid]
            switch_values = np.zeros(valid.sum(), dtype=np.float32)
            switch_values[can_switch] = (
                sorted_labels[positions[valid][can_switch]] != sorted_labels[previous[can_switch]]
            ).astype(np.float32)
            switched[valid] = switch_values
        for h, half_life in enumerate(HALF_LIVES):
            values = np.zeros((n, N_CLASSES), dtype=np.float32)
            if valid.any():
                selected = decayed[h, positions[valid]].astype(np.float64)
                selected *= np.exp(-cutoffs[valid, None] / half_life)
                selected /= np.maximum(selected.sum(axis=1, keepdims=True), 1e-12)
                values[valid] = selected.astype(np.float32)
            distributions.append(values)
        event_index = self._index(self.keys[name], self.days)
        event_unique, event_starts, event_ends, event_days, _ = event_index
        event_last, event_group_starts = lookup_last_positions(
            event_unique,
            event_starts,
            event_ends,
            event_days,
            self.keys[name][query_positions],
            query_days - EPS_DAY,
        )
        cutoff_last, _ = lookup_last_positions(
            event_unique,
            event_starts,
            event_ends,
            event_days,
            self.keys[name][query_positions],
            cutoffs,
        )
        bridge_count = np.zeros(n, dtype=np.float32)
        bridge_gap = np.full(n, 9999.0, dtype=np.float32)
        has_event = event_last >= 0
        if has_event.any():
            left_rank = np.where(
                cutoff_last[has_event] >= 0,
                cutoff_last[has_event] - event_group_starts[has_event] + 1,
                0,
            )
            right_rank = event_last[has_event] - event_group_starts[has_event] + 1
            local_count = np.maximum(0, right_rank - left_rank)
            bridge_count[has_event] = local_count.astype(np.float32)
            active = np.zeros(n, dtype=bool)
            active[np.flatnonzero(has_event)[local_count > 0]] = True
            bridge_gap[active] = query_days[active] - event_days[event_last[active]]
        span = np.maximum(query_days - cutoffs, 1.0)
        block = [
            last_label[:, None],
            mode_label[:, None],
            np.log1p(support)[:, None],
            np.minimum(staleness, 9999.0)[:, None],
            np.log1p(np.minimum(staleness, 9999.0))[:, None],
            purity[:, None],
            switched[:, None],
            valid.astype(np.float32)[:, None],
            np.log1p(bridge_count)[:, None],
            (bridge_count / span)[:, None],
            np.log1p(np.minimum(bridge_gap, 9999.0))[:, None],
        ] + distributions
        matrix = np.concatenate(block, axis=1).astype(np.float32)
        metadata = {
            "last_label": last_label.astype(np.int16) - 1,
            "support": support,
            "staleness": staleness,
        }
        del counts, decayed
        return matrix, metadata

    def make_features(self, query_ids, cutoffs, label_ids, labels, return_gate=False):
        query_positions = self.positions(query_ids)
        label_positions = self.positions(label_ids)
        query_days = self.days[query_positions]
        pieces = [self.base[query_positions]]
        names = list(self.base_names)
        categorical = list(self.base_categorical)
        fine_metadata = None
        for name, _ in HIERARCHIES:
            block, metadata = self._history_block(
                name, query_positions, cutoffs, label_positions, labels, query_days
            )
            offset = sum(piece.shape[1] for piece in pieces)
            categorical.extend([offset, offset + 1])
            names.extend(
                [
                    f"{name}_last_label",
                    f"{name}_mode_label",
                    f"{name}_log_support",
                    f"{name}_staleness",
                    f"{name}_log_staleness",
                    f"{name}_purity",
                    f"{name}_switch",
                    f"{name}_has_history",
                    f"{name}_log_bridge_count",
                    f"{name}_bridge_rate",
                    f"{name}_log_bridge_gap",
                ]
            )
            for half_life in HALF_LIVES.astype(int):
                names.extend([f"{name}_decay_{half_life}_class_{c}" for c in range(N_CLASSES)])
            pieces.append(block)
            if name == "sold_doc_org":
                fine_metadata = metadata
        pieces.append((query_days - cutoffs).astype(np.float32)[:, None])
        names.append("LABEL_EMBARGO_DAYS")
        matrix = np.concatenate(pieces, axis=1)
        if return_gate:
            return matrix, names, categorical, fine_metadata
        return matrix, names, categorical, None


# Modeling

@dataclass
class DualModel:
    all_model: object
    tail_model: object
    source_prior: np.ndarray


def lgb_params():
    threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "11")))
    return {
        "objective": "multiclass",
        "num_class": N_CLASSES,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_data_in_leaf": 75,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "max_cat_threshold": 64,
        "cat_l2": 10.0,
        "cat_smooth": 20.0,
        "verbosity": -1,
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "num_threads": threads,
        "force_col_wise": True,
    }


def class_prior(labels, weights):
    counts = np.bincount(labels, weights=weights, minlength=N_CLASSES).astype(np.float64)
    counts += 1e-3
    return counts / counts.sum()


def fit_dual(x, y, query_days, cutoff, categorical, rounds):
    age = np.maximum(0.0, cutoff - query_days)
    all_weights = np.power(0.5, age / ALL_WEIGHT_HALF_LIFE)
    tail_mask = query_days >= cutoff - TAIL_DAYS
    if tail_mask.sum() < 1000:
        tail_mask = query_days >= np.quantile(query_days, 0.5)
    tail_weights = np.power(0.5, age[tail_mask] / TAIL_WEIGHT_HALF_LIFE)
    all_data = lgb.Dataset(
        x,
        label=y,
        weight=all_weights,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    all_model = lgb.train(
        lgb_params(),
        all_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    tail_data = lgb.Dataset(
        x[tail_mask],
        label=y[tail_mask],
        weight=tail_weights,
        categorical_feature=categorical,
        free_raw_data=True,
    )
    tail_model = lgb.train(
        lgb_params(),
        tail_data,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    all_prior = class_prior(y, all_weights)
    tail_prior = class_prior(y[tail_mask], tail_weights)
    return DualModel(all_model, tail_model, np.stack([all_prior, tail_prior]))


def make_training_features(engine, labels_frame, views):
    ids = labels_frame["SALESDOCUMENT"].to_numpy(dtype=np.int64)
    labels = labels_frame["label"].to_numpy(dtype=np.int8)
    positions = engine.positions(ids)
    days = engine.days[positions]
    query_ids = []
    cutoffs = []
    for view in range(views):
        mixed = ids.astype(np.uint64) * np.uint64(11400714819323198485)
        salt = np.uint64(SEED + view * 1000003)
        offsets = ((mixed ^ salt) % np.uint64(154)).astype(np.float64)
        query_ids.append(ids)
        cutoffs.append(np.minimum(days - offsets, days - EPS_DAY))
    all_ids = np.concatenate(query_ids)
    all_cutoffs = np.concatenate(cutoffs)
    x, names, categorical, _ = engine.make_features(all_ids, all_cutoffs, ids, labels)
    y = np.tile(labels, views)
    qdays = np.tile(days, views)
    return x, y, qdays, names, categorical


def predict_dual(model, x, rounds, tail_blend):
    p_all = model.all_model.predict(x, num_iteration=rounds)
    p_tail = model.tail_model.predict(x, num_iteration=rounds)
    prediction = (1.0 - tail_blend) * p_all + tail_blend * p_tail
    prediction /= np.maximum(prediction.sum(axis=1, keepdims=True), 1e-12)
    source_prior = (1.0 - tail_blend) * model.source_prior[0] + tail_blend * model.source_prior[1]
    return prediction, source_prior, p_all, p_tail


def saerens_ratio(scores, source_prior):
    source = np.maximum(source_prior.astype(np.float64), 1e-8)
    target = source.copy()
    for _ in range(50):
        posterior = scores * (target / source)[None, :]
        posterior /= np.maximum(posterior.sum(axis=1, keepdims=True), 1e-15)
        updated = posterior.mean(axis=0)
        if np.max(np.abs(updated - target)) < 1e-7:
            target = updated
            break
        target = updated
    shrink = len(scores) / (len(scores) + 2000.0)
    target = shrink * target + (1.0 - shrink) * source
    return np.clip(target / source, 0.5, 2.0)


def causal_prior_correction(scores, days, source_prior, strength):
    if strength <= 0.0:
        return scores.copy()
    result = np.empty_like(scores, dtype=np.float64)
    day_batches = np.floor(days).astype(np.int64)
    order = np.argsort(day_batches, kind="stable")
    sorted_batches = day_batches[order]
    ends = np.r_[np.flatnonzero(sorted_batches[1:] != sorted_batches[:-1]) + 1, len(order)]
    start = 0
    for end in ends:
        cumulative = scores[order[:end]]
        ratio = saerens_ratio(cumulative, source_prior)
        applied = 1.0 + strength * (ratio - 1.0)
        batch_index = order[start:end]
        corrected = scores[batch_index] * applied[None, :]
        corrected /= np.maximum(corrected.sum(axis=1, keepdims=True), 1e-15)
        result[batch_index] = corrected
        start = end
    return result.astype(np.float64)


def month_codes(frame):
    return pd.to_datetime(frame["CREATIONTIMESTAMP"]).dt.strftime("%Y-%m").to_numpy()


def monthly_accuracies(scores, labels, months):
    prediction = scores.argmax(axis=1)
    return {
        month: float(np.mean(prediction[months == month] == labels[months == month]))
        for month in np.unique(months)
    }


def accuracy_by_origin(origin_records, score_getter):
    result = []
    monthly = []
    for record in origin_records:
        scores = score_getter(record)
        values = monthly_accuracies(scores, record["labels"], record["months"])
        result.append(float(np.mean(list(values.values()))))
        monthly.extend(values.values())
    return np.array(result), np.array(monthly)


def expert_probabilities(last_labels):
    result = np.full((len(last_labels), N_CLASSES), 0.03 / N_CLASSES, dtype=np.float64)
    valid = last_labels >= 0
    result[np.flatnonzero(valid), last_labels[valid]] += 0.97
    return result


def segment_codes(metadata):
    support = metadata["support"]
    staleness = metadata["staleness"]
    support_bin = np.digitize(support, [2.5, 5.5, 10.5, 25.5]).astype(np.int8)
    stale_bin = np.digitize(staleness, [30.0, 90.0]).astype(np.int8)
    return support_bin * 3 + stale_bin


def apply_gate(scores, metadata, segments, alpha):
    if alpha <= 0.0 or not segments:
        return scores.copy()
    codes = segment_codes(metadata)
    active = np.isin(codes, np.array(sorted(segments), dtype=np.int8)) & (metadata["last_label"] >= 0)
    result = scores.copy()
    if active.any():
        expert = expert_probabilities(metadata["last_label"][active])
        result[active] = (1.0 - alpha) * result[active] + alpha * expert
        result[active] /= np.maximum(result[active].sum(axis=1, keepdims=True), 1e-15)
    return result


def select_internal_rules(engine, train, debug, start):
    if debug:
        return {
            "rounds": 80,
            "tail_blend": 0.25,
            "em_strength": 0.0,
            "gate_segments": set(),
            "gate_alpha": 0.0,
            "diagnostics": {"mode": "debug_defaults"},
        }
    origins = [pd.Timestamp("2019-08-01"), pd.Timestamp("2019-11-01")]
    records = []
    for origin in origins:
        end = min(origin + pd.DateOffset(months=3), pd.Timestamp("2020-02-01"))
        fit_frame = train[pd.to_datetime(train["CREATIONTIMESTAMP"]) < origin].copy()
        holdout = train[
            (pd.to_datetime(train["CREATIONTIMESTAMP"]) >= origin)
            & (pd.to_datetime(train["CREATIONTIMESTAMP"]) < end)
        ].copy()
        x_train, y_train, train_days, _, categorical = make_training_features(engine, fit_frame, 1)
        cutoff = (origin.value - BASE_NS) / DAY_NS
        dual = fit_dual(x_train, y_train, train_days, cutoff, categorical, 1200)
        holdout_ids = holdout["SALESDOCUMENT"].to_numpy(dtype=np.int64)
        holdout_days = engine.days[engine.positions(holdout_ids)]
        holdout_cutoffs = np.full(len(holdout), cutoff - EPS_DAY, dtype=np.float64)
        x_holdout, _, _, metadata = engine.make_features(
            holdout_ids,
            holdout_cutoffs,
            fit_frame["SALESDOCUMENT"].to_numpy(dtype=np.int64),
            fit_frame["label"].to_numpy(dtype=np.int8),
            return_gate=True,
        )
        predictions = {}
        for rounds in [800, 1000, 1200]:
            predictions[rounds] = (
                dual.all_model.predict(x_holdout, num_iteration=rounds),
                dual.tail_model.predict(x_holdout, num_iteration=rounds),
            )
        records.append(
            {
                "origin": origin.strftime("%Y-%m-%d"),
                "labels": holdout["label"].to_numpy(dtype=np.int8),
                "months": month_codes(holdout),
                "days": holdout_days,
                "metadata": metadata,
                "predictions": predictions,
                "priors": dual.source_prior,
            }
        )
        del x_train, x_holdout, dual
        gc.collect()
        log_phase(start, f"internal_origin_{origin.strftime('%Y%m%d')}", f"fit={len(fit_frame)} holdout={len(holdout)}")
    candidates = []
    for rounds in [800, 1000, 1200]:
        baseline_worst = None
        baseline_month = []
        for record in records:
            baseline_month.extend(
                monthly_accuracies(record["predictions"][rounds][0], record["labels"], record["months"]).values()
            )
        baseline_worst = min(baseline_month)
        for blend in [0.0, 0.25, 0.5, 0.75]:
            origin_scores, monthly_scores = accuracy_by_origin(
                records,
                lambda record, r=rounds, b=blend: (1.0 - b) * record["predictions"][r][0]
                + b * record["predictions"][r][1],
            )
            regression = max(0.0, baseline_worst - float(monthly_scores.min()))
            objective = float(monthly_scores.mean() - regression)
            candidates.append((objective, float(monthly_scores.mean()), float(monthly_scores.min()), rounds, blend, origin_scores))
    candidates.sort(key=lambda value: (value[0], value[1], value[2], -value[4], -value[3]), reverse=True)
    _, blend_mean, blend_worst, selected_rounds, selected_blend, blend_origins = candidates[0]
    for record in records:
        p_all, p_tail = record["predictions"][selected_rounds]
        record["base"] = (1.0 - selected_blend) * p_all + selected_blend * p_tail
        record["source_prior"] = (1.0 - selected_blend) * record["priors"][0] + selected_blend * record["priors"][1]
    base_origins, _ = accuracy_by_origin(records, lambda record: record["base"])
    em_candidates = []
    for strength in [0.25, 0.5, 1.0]:
        for record in records:
            record[f"em_{strength}"] = causal_prior_correction(
                record["base"], record["days"], record["source_prior"], strength
            )
        origin_scores, monthly_scores = accuracy_by_origin(records, lambda record, s=strength: record[f"em_{s}"])
        if np.all(origin_scores > base_origins):
            em_candidates.append((float(monthly_scores.mean()), float(monthly_scores.min()), strength, origin_scores))
    if em_candidates:
        em_candidates.sort(reverse=True)
        _, _, em_strength, em_origins = em_candidates[0]
    else:
        em_strength = 0.0
        em_origins = base_origins
    for record in records:
        record["corrected"] = record["base"] if em_strength == 0.0 else record[f"em_{em_strength}"]
        record["segments"] = segment_codes(record["metadata"])
    qualified = set()
    segment_diagnostics = {}
    for segment in range(15):
        winning_months = 0
        support_total = 0
        deltas = []
        for record in records:
            model_labels = record["corrected"].argmax(axis=1)
            expert_labels = record["metadata"]["last_label"]
            for month in np.unique(record["months"]):
                mask = (record["segments"] == segment) & (record["months"] == month) & (expert_labels >= 0)
                count = int(mask.sum())
                support_total += count
                if count >= 100:
                    delta = float(np.mean(expert_labels[mask] == record["labels"][mask]) - np.mean(model_labels[mask] == record["labels"][mask]))
                    deltas.append(delta)
                    if delta > 0.01:
                        winning_months += 1
        if winning_months >= 2 and support_total >= 500:
            qualified.add(segment)
        segment_diagnostics[str(segment)] = {
            "support": support_total,
            "winning_months": winning_months,
            "mean_delta": float(np.mean(deltas)) if deltas else None,
        }
    gate_candidates = []
    corrected_origins, _ = accuracy_by_origin(records, lambda record: record["corrected"])
    for alpha in [0.25, 0.5, 0.75]:
        origin_scores, monthly_scores = accuracy_by_origin(
            records,
            lambda record, a=alpha: apply_gate(record["corrected"], record["metadata"], qualified, a),
        )
        if float(monthly_scores.mean()) > float(np.mean([np.mean(list(monthly_accuracies(r["corrected"], r["labels"], r["months"]).values())) for r in records])) and np.all(origin_scores >= corrected_origins - 0.001):
            gate_candidates.append((float(monthly_scores.mean()), float(monthly_scores.min()), alpha, origin_scores))
    if gate_candidates:
        gate_candidates.sort(reverse=True)
        _, _, gate_alpha, gate_origins = gate_candidates[0]
    else:
        gate_alpha = 0.0
        gate_origins = corrected_origins
    diagnostics = {
        "origins": [record["origin"] for record in records],
        "rounds": selected_rounds,
        "tail_blend": selected_blend,
        "blend_month_mean": blend_mean,
        "blend_worst_month": blend_worst,
        "blend_origin_scores": blend_origins.tolist(),
        "em_strength": em_strength,
        "em_origin_scores": em_origins.tolist(),
        "gate_segments": sorted(qualified),
        "gate_alpha": gate_alpha,
        "gate_origin_scores": gate_origins.tolist(),
        "segment_diagnostics": segment_diagnostics,
    }
    return {
        "rounds": selected_rounds,
        "tail_blend": selected_blend,
        "em_strength": em_strength,
        "gate_segments": qualified,
        "gate_alpha": gate_alpha,
        "diagnostics": diagnostics,
    }


def fit_and_predict(engine, labels_frame, query_frame, cutoff_timestamp, rules, views, start, phase):
    x_train, y_train, train_days, _, categorical = make_training_features(engine, labels_frame, views)
    cutoff = (pd.Timestamp(cutoff_timestamp).value - BASE_NS) / DAY_NS
    dual = fit_dual(x_train, y_train, train_days, cutoff, categorical, rules["rounds"])
    log_phase(start, f"{phase}_models", f"rows={len(y_train)} features={x_train.shape[1]}")
    del x_train, y_train, train_days
    gc.collect()
    query_ids = query_frame["SALESDOCUMENT"].to_numpy(dtype=np.int64)
    query_days = engine.days[engine.positions(query_ids)]
    cutoffs = np.full(len(query_ids), cutoff - EPS_DAY, dtype=np.float64)
    x_query, _, _, metadata = engine.make_features(
        query_ids,
        cutoffs,
        labels_frame["SALESDOCUMENT"].to_numpy(dtype=np.int64),
        labels_frame["label"].to_numpy(dtype=np.int8),
        return_gate=True,
    )
    scores, source_prior, _, _ = predict_dual(dual, x_query, rules["rounds"], rules["tail_blend"])
    scores = causal_prior_correction(scores, query_days, source_prior, rules["em_strength"])
    scores = apply_gate(scores, metadata, rules["gate_segments"], rules["gate_alpha"])
    scores /= np.maximum(scores.sum(axis=1, keepdims=True), 1e-15)
    log_phase(start, f"{phase}_predictions", f"rows={len(scores)}")
    return scores.astype(np.float32), metadata


def strata_counts(metadata):
    codes = segment_codes(metadata)
    unique, counts = np.unique(codes, return_counts=True)
    return {str(int(key)): int(value) for key, value in zip(unique, counts)}


# Orchestration

def run():
    warnings.filterwarnings("ignore")
    assigned_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "11")))
    numba.set_num_threads(min(assigned_threads, numba.config.NUMBA_NUM_THREADS))
    start = time.time()
    debug = "--debug" in sys.argv
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "shared_cache"))
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "kapso_output"))
    output.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = Path("output_data_generic_exp_3")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    train, validation, test = read_splits(root)
    frame, cached = load_document_frame(root, shared)
    log_phase(start, "legal_document_frame", f"rows={len(frame)} cached={cached} numba_threads={numba.get_num_threads()}")
    engine = FeatureEngine(frame)
    log_phase(start, "base_features", f"features={engine.base.shape[1]}")
    if debug:
        train_for_rules = train.sort_values("CREATIONTIMESTAMP").tail(30000).copy()
    else:
        train_for_rules = train
    feature_probe_frame = train_for_rules.sort_values("CREATIONTIMESTAMP").tail(min(30000, len(train_for_rules))).copy()
    probe_start = time.time()
    probe_x, _, _, _, _ = make_training_features(engine, feature_probe_frame, 1)
    probe_rate = len(feature_probe_frame) / max(time.time() - probe_start, 1e-6)
    del probe_x
    gc.collect()
    log_phase(start, "feature_rate", f"rows_per_second={probe_rate:.1f}")
    rules = select_internal_rules(engine, train_for_rules, debug, start)
    log_phase(
        start,
        "internal_selection",
        f"rounds={rules['rounds']} tail={rules['tail_blend']:.2f} em={rules['em_strength']:.2f} gate={rules['gate_alpha']:.2f}",
    )
    if debug:
        model_a_labels = train.sort_values("CREATIONTIMESTAMP").tail(30000).copy()
        views = 1
    else:
        model_a_labels = train
        views = 1
    validation_scores, validation_metadata = fit_and_predict(
        engine,
        model_a_labels,
        validation,
        "2020-02-01",
        rules,
        views,
        start,
        "model_a",
    )
    validation_path = output / "val_predictions.npy"
    np.save(validation_path, validation_scores)
    validation_digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    del validation_scores
    gc.collect()
    validation_labels = read_validation_labels(root)
    validation_labeled = validation.copy()
    validation_labeled["label"] = validation_labels
    combined = pd.concat([train, validation_labeled], ignore_index=True)
    if debug:
        model_b_labels = combined.sort_values("CREATIONTIMESTAMP").tail(30000).copy()
    else:
        model_b_labels = combined
    test_scores, test_metadata = fit_and_predict(
        engine,
        model_b_labels,
        test,
        "2020-07-01",
        rules,
        views,
        start,
        "model_b",
    )
    np.save(output / "test_predictions.npy", test_scores)
    if hashlib.sha256(validation_path.read_bytes()).hexdigest() != validation_digest:
        raise RuntimeError("validation predictions changed after Model B refit")
    diagnostics = {
        "debug": debug,
        "elapsed_seconds": time.time() - start,
        "feature_rows_per_second": probe_rate,
        "rules": rules["diagnostics"],
        "validation_strata_counts": strata_counts(validation_metadata),
        "test_strata_counts": strata_counts(test_metadata),
        "validation_fit": "train_labels_only_model_a",
        "test_fit": "train_plus_validation_model_b",
    }
    (diagnostics_dir / "internal_metrics.json").write_text(json.dumps(diagnostics, indent=2))
    log_phase(
        start,
        "complete",
        f"val={(71470, N_CLASSES)} test={test_scores.shape} debug={debug}",
    )
