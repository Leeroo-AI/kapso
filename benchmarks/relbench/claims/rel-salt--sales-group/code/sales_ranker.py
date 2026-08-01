import fcntl
import gc
import hashlib
import json
import math
import os
import pickle
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd


VERSION = "lane0_causal_ltr_v7"
ORIGINS = (
    (pd.Timestamp("2018-11-01"), pd.Timestamp("2019-04-01")),
    (pd.Timestamp("2019-04-01"), pd.Timestamp("2019-09-01")),
    (pd.Timestamp("2019-09-01"), pd.Timestamp("2020-02-01")),
)
V = pd.Timestamp("2020-02-01")
T = pd.Timestamp("2020-07-01")
HALF_LOG = math.log(2.0)
ROLE_NAMES = ("sold", "payer", "ship", "bill")
ROLE_COLUMNS = tuple(f"{name}_parties" for name in ROLE_NAMES)
STAT_COUNT = 0
STAT_SHARE14 = 1
STAT_SHARE60 = 2
STAT_SHARE180 = 3
STAT_WINDOW30 = 4
STAT_WINDOW90 = 5
STAT_WINDOW180 = 6
STAT_AGE = 7
STAT_RANK = 8


def phase(message, started):
    print(f"[phase] {message} elapsed_seconds={time.time() - started:.1f}")


def normalize_value(value):
    if value is None:
        return "__MISSING__"
    if isinstance(value, float) and np.isnan(value):
        return "__MISSING__"
    return value


def stable_code(value):
    raw = str(normalize_value(value)).encode("utf-8", errors="replace")
    return float(zlib.crc32(raw) % 1000003)


def sequence_values(value):
    if isinstance(value, (tuple, list, np.ndarray)):
        return value
    return ()


def softmax_rows(values):
    maximum = values.max(axis=1, keepdims=True)
    exp = np.exp(np.clip(values - maximum, -40.0, 0.0))
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


@dataclass
class SupportRecord:
    labels: np.ndarray
    stats: np.ndarray
    sorted_labels: np.ndarray
    sorted_positions: np.ndarray
    total_count: float
    distinct_labels: float

    def top(self, count):
        return self.labels[:count]

    def match(self, candidates):
        result = np.zeros((len(candidates), 9), dtype=np.float32)
        result[:, STAT_AGE] = 3650.0
        result[:, STAT_RANK] = 128.0
        positions = np.searchsorted(self.sorted_labels, candidates)
        valid = positions < len(self.sorted_labels)
        if valid.any():
            clipped = np.minimum(positions, len(self.sorted_labels) - 1)
            valid &= self.sorted_labels[clipped] == candidates
            result[valid] = self.stats[self.sorted_positions[clipped[valid]]]
        return result

    def anchor(self):
        if len(self.labels) == 0:
            return -1, 3650.0
        position = int(np.argmin(self.stats[:, STAT_AGE]))
        return int(self.labels[position]), float(self.stats[position, STAT_AGE])


@dataclass
class SupportIndex:
    records: dict

    def get(self, key):
        return self.records.get(normalize_value(key))


@dataclass
class FrozenState:
    cutoff: pd.Timestamp
    supports: dict
    transitions: SupportIndex

    @property
    def global_record(self):
        return self.supports["global"].get("__GLOBAL__")


@dataclass
class PairData:
    features: np.ndarray
    candidates: np.ndarray
    posterior_log: np.ndarray
    groups: np.ndarray
    documents: np.ndarray
    feature_names: list
    query_frame: pd.DataFrame


def build_support_index(frame, key_column, cutoff, keep):
    if len(frame) == 0:
        return SupportIndex({})
    selected = frame[[key_column, "label", "time"]].copy()
    selected = selected.dropna(subset=[key_column, "label", "time"])
    selected[key_column] = selected[key_column].map(normalize_value)
    selected["label"] = selected["label"].astype(np.int16)
    age = np.maximum((cutoff - selected["time"]).dt.total_seconds().to_numpy() / 86400.0, 0.0)
    selected["weight14"] = np.exp(-HALF_LOG * age / 14.0).astype(np.float32)
    selected["weight60"] = np.exp(-HALF_LOG * age / 60.0).astype(np.float32)
    selected["weight180"] = np.exp(-HALF_LOG * age / 180.0).astype(np.float32)
    selected["window30"] = (age <= 30.0).astype(np.float32)
    selected["window90"] = (age <= 90.0).astype(np.float32)
    selected["window180"] = (age <= 180.0).astype(np.float32)
    selected["age"] = age.astype(np.float32)
    grouped = selected.groupby([key_column, "label"], sort=False, observed=True).agg(
        count=("label", "size"),
        weight14=("weight14", "sum"),
        weight60=("weight60", "sum"),
        weight180=("weight180", "sum"),
        window30=("window30", "sum"),
        window90=("window90", "sum"),
        window180=("window180", "sum"),
        age=("age", "min"),
    ).reset_index()
    sum_columns = ["weight14", "weight60", "weight180", "window30", "window90", "window180"]
    totals = grouped.groupby(key_column, sort=False, observed=True)[sum_columns].transform("sum")
    grouped[sum_columns] = grouped[sum_columns] / np.maximum(totals, 1e-12)
    grouped = grouped.sort_values([key_column, "weight60", "age"], ascending=[True, False, True], kind="mergesort")
    grouped["rank"] = grouped.groupby(key_column, sort=False, observed=True).cumcount().astype(np.float32)
    records = {}
    for key, part in grouped.groupby(key_column, sort=False, observed=True):
        total_count = float(part["count"].sum())
        distinct = float(len(part))
        part = part.iloc[:keep]
        labels = part["label"].to_numpy(np.int16, copy=True)
        stats = part[["count", "weight14", "weight60", "weight180", "window30", "window90", "window180", "age", "rank"]].to_numpy(np.float32, copy=True)
        order = np.argsort(labels, kind="stable")
        records[normalize_value(key)] = SupportRecord(labels, stats, labels[order], order.astype(np.int16), total_count, distinct)
    return SupportIndex(records)


def build_transitions(history, cutoff):
    sold = history[["sold_parties", "time", "label"]].explode("sold_parties")
    sold = sold.dropna(subset=["sold_parties", "time", "label"]).rename(columns={"sold_parties": "party"})
    sold = sold.sort_values(["party", "time"], kind="mergesort")
    batches = sold.drop_duplicates(["party", "time"], keep="last").copy()
    batches["previous_label"] = batches.groupby("party", sort=False)["label"].shift(1)
    batches["previous_time"] = batches.groupby("party", sort=False)["time"].shift(1)
    prior = batches[["party", "time", "previous_label", "previous_time"]]
    pairs = sold.merge(prior, on=["party", "time"], how="left", validate="many_to_one")
    pairs = pairs.dropna(subset=["previous_label", "previous_time"])
    gaps = (pairs["time"] - pairs["previous_time"]).dt.total_seconds().to_numpy() / 86400.0
    pairs["gap"] = np.select([gaps <= 7, gaps <= 30, gaps <= 90, gaps <= 180], [0, 1, 2, 3], default=4).astype(np.int8)
    pairs["transition_key"] = list(zip(pairs["previous_label"].astype(np.int16), pairs["gap"]))
    return build_support_index(pairs.rename(columns={"label": "next_label", "next_label": "label"}), "transition_key", cutoff, 32)


def unique_tuple(values):
    return tuple(pd.unique(values.dropna()).tolist())


def combine_key(frame, columns):
    values = frame[list(columns)].fillna("__MISSING__").astype(str)
    return values.agg("\x1f".join, axis=1)


class SalesGroupPipeline:
    def __init__(self, output, cache, debug):
        self.output = Path(output)
        self.cache = Path(cache)
        self.debug = bool(debug)
        self.started = time.time()
        self.data_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / os.environ["RELBENCH_DATASET"]
        self.cache_root = self.cache / VERSION
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.diagnostics = {"version": VERSION, "debug": self.debug, "origins": {}}
        self.models = []

    def run(self):
        documents, train, val_input, test_input, snapshot_ok = self.load_extracted()
        self.diagnostics["snapshot_items_available"] = bool(snapshot_ok)
        phase("extracted relational tables", self.started)
        train_history = self.attach_documents(train, documents)
        origin_frames = []
        states = []
        for origin, end in ORIGINS:
            state = self.load_state(train_history, origin, f"train_{origin:%Y%m%d}")
            target = train_history[(train_history["time"] >= origin) & (train_history["time"] < end)].copy()
            if self.debug:
                target = self.sample_frame(target, 3334)
            states.append(state)
            origin_frames.append(target)
        recall40, recall_strata = self.measure_recall(origin_frames, states, 40)
        mean_recall = float(np.mean(recall40))
        cap = 64 if mean_recall < 0.99 else 40
        self.diagnostics["candidate_recall_at_40"] = recall40
        self.diagnostics["candidate_recall_strata_at_40"] = recall_strata
        self.diagnostics["candidate_cap"] = cap
        print(f"[recon] candidate_recall_at_40={recall40} mean={mean_recall:.6f} selected_cap={cap}")
        pair_origins = []
        pair_started = time.time()
        for index, (target, state) in enumerate(zip(origin_frames, states)):
            pair = self.build_pairs(target, state, cap)
            pair_origins.append(pair)
            labels = target["label"].to_numpy(np.int16)
            recall = self.pair_recall(pair, labels)
            rate = len(target) / max(time.time() - pair_started, 1e-9)
            self.diagnostics["origins"][str(index)] = {"queries": int(len(target)), "candidate_recall": recall}
            print(f"[matrix] origin={index} queries={len(target)} pairs={len(pair.candidates)} recall={recall:.6f} cumulative_queries_per_second={rate:.1f}")
        phase("built internal origin matrices", self.started)
        rounds, diagnostic_model, selection_model = self.select_iterations(pair_origins, origin_frames)
        self.models.extend([diagnostic_model, selection_model])
        blend, pool = self.select_postprocessing(pair_origins, origin_frames, states, diagnostic_model, selection_model)
        self.diagnostics["selected_rounds"] = int(rounds)
        self.diagnostics["posterior_blend"] = float(blend)
        self.diagnostics["forward_pool_weight"] = float(pool)
        model_a = self.refit_model(pair_origins, origin_frames, rounds, "model_a")
        self.models.append(model_a)
        phase("trained model A", self.started)
        val_frame = self.attach_documents(val_input, documents)
        state_v = self.load_state(train_history, V, "train_20200201")
        val_pairs = self.build_pairs(val_frame, state_v, cap)
        val_margin = model_a.predict(val_pairs.features, num_iteration=rounds)
        val_scores = self.compose_scores(val_pairs, val_margin, state_v, blend)
        val_scores = self.forward_pool(val_scores, val_pairs.query_frame, pool)
        self.validate_scores(val_scores, len(val_input), "validation")
        val_path = self.output / "val_predictions.npy"
        np.save(val_path, val_scores.astype(np.float32))
        frozen_hash = self.file_hash(val_path)
        np.save(self.cache_root / "model_a_val_predictions.npy", val_scores.astype(np.float32))
        print(f"[provenance] model_a_validation_frozen sha256={frozen_hash}")
        del val_scores, val_margin
        gc.collect()
        phase("froze model-A validation predictions before validation-label access", self.started)
        val_labels = self.load_validation_labels()
        val_history = self.attach_documents(val_labels, documents)
        train_val_history = pd.concat([train_history, val_history], ignore_index=True)
        val_targets = val_history["label"].to_numpy(np.int16)
        model_b_pairs = pair_origins + [val_pairs]
        model_b_frames = origin_frames + [val_history]
        model_b = self.refit_model(model_b_pairs, model_b_frames, rounds, "model_b")
        self.models.append(model_b)
        phase("trained model B with V-to-T supervision", self.started)
        state_t = self.load_state(train_val_history, T, "trainval_20200701")
        test_frame = self.attach_documents(test_input, documents)
        test_pairs = self.build_pairs(test_frame, state_t, cap)
        test_margin = model_b.predict(test_pairs.features, num_iteration=rounds)
        test_scores = self.compose_scores(test_pairs, test_margin, state_t, blend)
        test_scores = self.forward_pool(test_scores, test_pairs.query_frame, pool)
        self.validate_scores(test_scores, len(test_input), "test")
        if self.file_hash(val_path) != frozen_hash:
            raise RuntimeError("validation predictions changed after model-B training")
        val_frozen = np.load(val_path, allow_pickle=False)
        np.save(self.output / "val_predictions.npy", val_frozen.astype(np.float32))
        np.save(self.output / "test_predictions.npy", test_scores.astype(np.float32))
        np.save(self.cache_root / "model_b_test_predictions.npy", test_scores.astype(np.float32))
        self.validate_artifacts(len(val_input), len(test_input), frozen_hash)
        self.diagnostics["model_a_validation_sha256"] = frozen_hash
        self.diagnostics["val_shape"] = list(val_frozen.shape)
        self.diagnostics["test_shape"] = list(test_scores.shape)
        self.register_artifacts()
        phase("saved and validated final artifacts", self.started)
        return self.diagnostics

    def load_extracted(self):
        cache_path = self.cache_root / "document_features.pkl"
        train_path = self.data_root / "tasks" / os.environ["RELBENCH_TASK"] / "train.parquet"
        val_path = self.data_root / "tasks" / os.environ["RELBENCH_TASK"] / "val.parquet"
        test_path = self.data_root / "tasks" / os.environ["RELBENCH_TASK"] / "test.parquet"
        train = pd.read_parquet(train_path, columns=["CREATIONTIMESTAMP", "SALESDOCUMENT", "SALESGROUP"]).rename(columns={"CREATIONTIMESTAMP": "time", "SALESDOCUMENT": "document", "SALESGROUP": "label"})
        train["label"] = train["label"].astype(np.int16)
        train["row_index"] = np.arange(len(train), dtype=np.int32)
        val_input = pd.read_parquet(val_path, columns=["CREATIONTIMESTAMP", "SALESDOCUMENT"]).rename(columns={"CREATIONTIMESTAMP": "time", "SALESDOCUMENT": "document"})
        val_input["row_index"] = np.arange(len(val_input), dtype=np.int32)
        test_input = pd.read_parquet(test_path, columns=["CREATIONTIMESTAMP", "SALESDOCUMENT"]).rename(columns={"CREATIONTIMESTAMP": "time", "SALESDOCUMENT": "document"})
        test_input["row_index"] = np.arange(len(test_input), dtype=np.int32)
        if cache_path.exists():
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
            print(f"[cache] loaded extracted document features {cache_path}")
            return payload["documents"], train, val_input, test_input, payload["snapshot_ok"]
        needed = set(train["document"].tolist()) | set(val_input["document"].tolist()) | set(test_input["document"].tolist())
        header_columns = ["SALESDOCUMENT", "SALESDOCUMENTTYPE", "SALESORGANIZATION", "DISTRIBUTIONCHANNEL", "ORGANIZATIONDIVISION", "BILLINGCOMPANYCODE", "TRANSACTIONCURRENCY", "CREATIONTIMESTAMP"]
        headers = pd.read_parquet(self.data_root / "db" / "salesdocument.parquet", columns=header_columns)
        headers = headers[headers["SALESDOCUMENT"].isin(needed)].copy()
        headers = headers.rename(columns={"SALESDOCUMENT": "document", "SALESDOCUMENTTYPE": "doc_type", "SALESORGANIZATION": "sales_org", "DISTRIBUTIONCHANNEL": "channel", "ORGANIZATIONDIVISION": "division", "BILLINGCOMPANYCODE": "billing_company", "TRANSACTIONCURRENCY": "currency", "CREATIONTIMESTAMP": "header_time"})
        item_columns = ["SALESDOCUMENT", "SALESDOCUMENTITEM", "SALESDOCUMENTITEMCATEGORY", "PRODUCT", "SOLDTOPARTY", "SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY", "CREATIONTIMESTAMP"]
        needed_documents = pd.DataFrame({"document": np.fromiter(needed, dtype=np.int64)})
        connection = duckdb.connect()
        connection.register("needed_documents", needed_documents)
        item_path = str(self.data_root / "db" / "salesdocumentitem.parquet")
        item_features = connection.execute(
            """
            SELECT
                i.SALESDOCUMENT AS document,
                count(*)::INTEGER AS item_count,
                count(DISTINCT i.PRODUCT)::INTEGER AS distinct_products,
                count(DISTINCT i.SALESDOCUMENTITEMCATEGORY)::INTEGER AS distinct_categories,
                min(i.CREATIONTIMESTAMP) AS item_time_min,
                max(i.CREATIONTIMESTAMP) AS item_time_max,
                list(DISTINCT i.PRODUCT ORDER BY i.PRODUCT) FILTER (WHERE i.PRODUCT IS NOT NULL) AS product_set,
                list(DISTINCT i.SALESDOCUMENTITEMCATEGORY ORDER BY i.SALESDOCUMENTITEMCATEGORY) FILTER (WHERE i.SALESDOCUMENTITEMCATEGORY IS NOT NULL) AS category_set,
                list(DISTINCT i.SOLDTOPARTY ORDER BY i.SOLDTOPARTY) FILTER (WHERE i.SOLDTOPARTY IS NOT NULL) AS sold_parties,
                list(DISTINCT i.PAYERPARTY ORDER BY i.PAYERPARTY) FILTER (WHERE i.PAYERPARTY IS NOT NULL) AS payer_parties,
                list(DISTINCT i.SHIPTOPARTY ORDER BY i.SHIPTOPARTY) FILTER (WHERE i.SHIPTOPARTY IS NOT NULL) AS ship_parties,
                list(DISTINCT i.BILLTOPARTY ORDER BY i.BILLTOPARTY) FILTER (WHERE i.BILLTOPARTY IS NOT NULL) AS bill_parties
            FROM read_parquet(?) AS i
            SEMI JOIN needed_documents AS n ON i.SALESDOCUMENT = n.document
            GROUP BY i.SALESDOCUMENT
            """,
            [item_path],
        ).fetch_df()
        connection.close()
        documents = headers.merge(item_features, on="document", how="left", validate="one_to_one")
        snapshot_ok = bool(((documents["item_time_min"] == documents["header_time"]) & (documents["item_time_max"] == documents["header_time"])).all())
        if not snapshot_ok:
            raise RuntimeError("current item timestamps do not match header timestamps")
        customers = pd.read_parquet(self.data_root / "db" / "customer.parquet", columns=["CUSTOMER", "ADDRESSID"])
        addresses = pd.read_parquet(self.data_root / "db" / "address.parquet", columns=["ADDRESSID", "COUNTRY", "REGION"])
        customer_geo = customers.merge(addresses, on="ADDRESSID", how="left", validate="many_to_one").set_index("CUSTOMER")
        country_map = customer_geo["COUNTRY"]
        region_map = customer_geo["REGION"]
        for role in ROLE_NAMES:
            first = documents[f"{role}_parties"].map(lambda value: sequence_values(value)[0] if len(sequence_values(value)) else np.nan)
            documents[f"{role}_country"] = first.map(country_map)
            documents[f"{role}_region"] = first.map(region_map)
            documents[f"{role}_party_count"] = documents[f"{role}_parties"].map(lambda value: len(sequence_values(value))).astype(np.int16)
        documents["sales_area"] = combine_key(documents, ["sales_org", "channel", "division"])
        documents["sold_geo"] = combine_key(documents, ["sold_country", "sold_region"])
        object_columns = ["doc_type", "sales_org", "channel", "division", "billing_company", "currency", "sales_area", "sold_geo", "sold_country", "sold_region"]
        for column in object_columns:
            documents[column] = documents[column].fillna("__MISSING__")
        documents["party_agreement"] = documents.apply(self.party_agreement, axis=1).astype(np.float32)
        documents = documents.drop(columns=["item_time_min", "item_time_max", "header_time"])
        if len(documents) != len(needed):
            raise RuntimeError(f"document feature join lost rows: {len(documents)} != {len(needed)}")
        payload = {"documents": documents, "snapshot_ok": snapshot_ok}
        temporary = cache_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
        print(f"[cache] wrote extracted document features {cache_path}")
        return documents, train, val_input, test_input, snapshot_ok

    def party_agreement(self, row):
        role_sets = []
        for column in ROLE_COLUMNS:
            value = row[column]
            role_sets.append(set(sequence_values(value)))
        populated = [value for value in role_sets if value]
        if len(populated) < 2:
            return 1.0
        common = set.intersection(*populated)
        union = set.union(*populated)
        return len(common) / max(len(union), 1)

    def attach_documents(self, split, documents):
        original = split["document"].to_numpy(copy=True)
        result = split.merge(documents, on="document", how="left", sort=False, validate="many_to_one")
        result = result.sort_values("row_index", kind="stable").reset_index(drop=True)
        if len(result) != len(split) or not np.array_equal(result["document"].to_numpy(), original):
            raise RuntimeError("document join changed task row order")
        if result["item_count"].isna().any():
            raise RuntimeError("item aggregation missing for task documents")
        return result

    def sample_frame(self, frame, count):
        if len(frame) <= count:
            return frame.copy()
        positions = np.linspace(0, len(frame) - 1, count, dtype=np.int64)
        return frame.iloc[positions].copy().reset_index(drop=True)

    def state_cache_path(self, tag):
        return self.cache_root / f"state_{tag}.pkl"

    def load_state(self, history, cutoff, tag):
        path = self.state_cache_path(tag)
        if path.exists():
            with path.open("rb") as handle:
                state = pickle.load(handle)
            print(f"[cache] loaded frozen state tag={tag}")
            return state
        legal = history[history["time"] < cutoff].copy()
        supports = {}
        for role, column in zip(ROLE_NAMES, ROLE_COLUMNS):
            expanded = legal[[column, "label", "time"]].explode(column).rename(columns={column: "support_key"})
            supports[role] = build_support_index(expanded, "support_key", cutoff, 64)
        product = legal[["product_set", "label", "time"]].explode("product_set").rename(columns={"product_set": "support_key"})
        category = legal[["category_set", "label", "time"]].explode("category_set").rename(columns={"category_set": "support_key"})
        supports["product"] = build_support_index(product, "support_key", cutoff, 32)
        supports["category"] = build_support_index(category, "support_key", cutoff, 32)
        singular = {
            "area": "sales_area",
            "doc_type": "doc_type",
            "geo": "sold_geo",
            "billing": "billing_company",
            "currency": "currency",
        }
        for name, column in singular.items():
            expanded = legal[[column, "label", "time"]].rename(columns={column: "support_key"})
            supports[name] = build_support_index(expanded, "support_key", cutoff, 64)
        global_frame = legal[["label", "time"]].copy()
        global_frame["support_key"] = "__GLOBAL__"
        supports["global"] = build_support_index(global_frame, "support_key", cutoff, 502)
        transitions = self.make_transition_index(legal, cutoff)
        state = FrozenState(cutoff, supports, transitions)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
        print(f"[cache] wrote frozen state tag={tag} history_rows={len(legal)}")
        return state

    def make_transition_index(self, history, cutoff):
        sold = history[["sold_parties", "time", "label"]].explode("sold_parties")
        sold = sold.dropna(subset=["sold_parties", "time", "label"]).rename(columns={"sold_parties": "party"})
        sold = sold.sort_values(["party", "time"], kind="mergesort")
        batches = sold.drop_duplicates(["party", "time"], keep="last").copy()
        batches["previous_label"] = batches.groupby("party", sort=False)["label"].shift(1)
        batches["previous_time"] = batches.groupby("party", sort=False)["time"].shift(1)
        pairs = sold.merge(batches[["party", "time", "previous_label", "previous_time"]], on=["party", "time"], how="left", validate="many_to_one")
        pairs = pairs.dropna(subset=["previous_label", "previous_time"])
        gaps = (pairs["time"] - pairs["previous_time"]).dt.total_seconds().to_numpy() / 86400.0
        pairs["gap_bucket"] = np.select([gaps <= 7, gaps <= 30, gaps <= 90, gaps <= 180], [0, 1, 2, 3], default=4).astype(np.int8)
        pairs["transition_key"] = list(zip(pairs["previous_label"].astype(np.int16), pairs["gap_bucket"]))
        pairs = pairs.rename(columns={"label": "next_label"})
        pairs["label"] = pairs["next_label"].astype(np.int16)
        return build_support_index(pairs, "transition_key", cutoff, 32)

    def role_records(self, row, state, role):
        values = getattr(row, f"{role}_parties")
        return [record for value in sequence_values(values) if (record := state.supports[role].get(value)) is not None]

    def source_records(self, values, index):
        return [record for value in sequence_values(values) if (record := index.get(value)) is not None]

    def sold_anchor(self, records):
        anchor = -1
        age = 3650.0
        for record in records:
            label, candidate_age = record.anchor()
            if candidate_age < age:
                anchor = label
                age = candidate_age
        return anchor, age

    def gap_bucket(self, age):
        if age <= 7:
            return 0
        if age <= 30:
            return 1
        if age <= 90:
            return 2
        if age <= 180:
            return 3
        return 4

    def candidates_for_row(self, row, state, cap):
        accumulator = {}
        forced = set()

        def add(records, top, weight, force=False):
            for record in records:
                labels = record.top(top)
                for rank, label in enumerate(labels):
                    value = int(label)
                    accumulator[value] = accumulator.get(value, 0.0) + weight / (rank + 1.0)
                    if force:
                        forced.add(value)

        roles = {role: self.role_records(row, state, role) for role in ROLE_NAMES}
        add(roles["sold"], 12, 30.0, True)
        for role in ("payer", "ship", "bill"):
            add(roles[role], 4, 8.0)
        anchor, anchor_age = self.sold_anchor(roles["sold"])
        transition = None
        if anchor >= 0:
            forced.add(anchor)
            accumulator[anchor] = accumulator.get(anchor, 0.0) + 40.0
            transition = state.transitions.get((np.int16(anchor), self.gap_bucket(anchor_age)))
            if transition is not None:
                add([transition], 8, 15.0, True)
        products = self.source_records(row.product_set, state.supports["product"])
        categories = self.source_records(row.category_set, state.supports["category"])
        add(products, 4, 2.5)
        add(categories, 4, 2.5)
        singular_records = {
            "area": state.supports["area"].get(row.sales_area),
            "doc_type": state.supports["doc_type"].get(row.doc_type),
            "geo": state.supports["geo"].get(row.sold_geo),
            "billing": state.supports["billing"].get(row.billing_company),
            "currency": state.supports["currency"].get(row.currency),
        }
        add([singular_records["area"]] if singular_records["area"] else [], 12, 4.0)
        add([singular_records["doc_type"]] if singular_records["doc_type"] else [], 8, 1.5)
        add([singular_records["geo"]] if singular_records["geo"] else [], 8, 2.5)
        add([singular_records["billing"]] if singular_records["billing"] else [], 8, 1.5)
        add([singular_records["currency"]] if singular_records["currency"] else [], 8, 1.5)
        add([state.global_record], max(cap, 24), 0.2)
        forced_order = sorted(forced, key=lambda label: accumulator.get(label, 0.0), reverse=True)
        remaining = sorted((label for label in accumulator if label not in forced), key=lambda label: accumulator[label], reverse=True)
        labels = (forced_order + remaining)[:cap]
        if len(labels) < cap:
            for label in state.global_record.labels:
                value = int(label)
                if value not in accumulator:
                    accumulator[value] = 0.0
                    labels.append(value)
                if len(labels) == cap:
                    break
        candidates = np.asarray(labels, dtype=np.int16)
        source_score = np.asarray([accumulator[int(label)] for label in candidates], dtype=np.float32)
        return candidates, source_score, roles, products, categories, singular_records, transition, anchor, anchor_age

    def measure_recall(self, frames, states, cap):
        recalls = []
        all_strata = {}
        for origin_index, (frame, state) in enumerate(zip(frames, states)):
            hit = np.zeros(len(frame), dtype=bool)
            strata = {"cold": [0, 0], "age_0_30": [0, 0], "age_31_90": [0, 0], "age_91_180": [0, 0], "age_181_plus": [0, 0]}
            for index, row in enumerate(frame.itertuples(index=False)):
                candidates, _, roles, _, _, _, _, _, age = self.candidates_for_row(row, state, cap)
                target = int(row.label)
                hit[index] = bool(np.any(candidates == target))
                if not roles["sold"]:
                    key = "cold"
                elif age <= 30:
                    key = "age_0_30"
                elif age <= 90:
                    key = "age_31_90"
                elif age <= 180:
                    key = "age_91_180"
                else:
                    key = "age_181_plus"
                strata[key][0] += 1
                strata[key][1] += int(hit[index])
            recall = float(hit.mean()) if len(hit) else 0.0
            recalls.append(recall)
            all_strata[str(origin_index)] = {key: {"count": value[0], "recall": value[1] / max(value[0], 1)} for key, value in strata.items()}
        return recalls, all_strata

    def aggregate_records(self, records, candidates):
        if not records:
            result = np.zeros((len(candidates), 9), dtype=np.float32)
            result[:, STAT_AGE] = 3650.0
            result[:, STAT_RANK] = 128.0
            return result, result.copy(), 0.0, 0.0
        matched = np.stack([record.match(candidates) for record in records], axis=0)
        maximum = matched.max(axis=0)
        maximum[:, STAT_AGE] = matched[:, :, STAT_AGE].min(axis=0)
        maximum[:, STAT_RANK] = matched[:, :, STAT_RANK].min(axis=0)
        mean = matched.mean(axis=0)
        total_count = float(sum(record.total_count for record in records))
        distinct = float(max(record.distinct_labels for record in records))
        return maximum, mean, total_count, distinct

    def pair_features(self, row, state, cap):
        candidates, source_score, roles, products, categories, singular, transition, anchor, anchor_age = self.candidates_for_row(row, state, cap)
        blocks = [candidates.astype(np.float32), np.log1p(source_score), np.arange(cap, dtype=np.float32), np.full(cap, cap, dtype=np.float32)]
        names = ["candidate", "candidate_source_score", "candidate_source_rank", "candidate_count"]
        role_matrices = {}
        role_anchors = []
        role_activity = []
        for role in ROLE_NAMES:
            maximum, mean, total_count, distinct = self.aggregate_records(roles[role], candidates)
            role_matrices[role] = maximum
            anchors = np.zeros(cap, dtype=np.float32)
            for record in roles[role]:
                label, _ = record.anchor()
                anchors[candidates == label] = 1.0
            blocks.extend([
                np.log1p(maximum[:, STAT_COUNT]), maximum[:, STAT_SHARE14], maximum[:, STAT_SHARE60], maximum[:, STAT_SHARE180],
                maximum[:, STAT_RANK], np.log1p(maximum[:, STAT_AGE]), anchors,
                np.full(cap, np.log1p(total_count), dtype=np.float32), np.full(cap, distinct, dtype=np.float32),
            ])
            names.extend([f"{role}_log_count", f"{role}_share14", f"{role}_share60", f"{role}_share180", f"{role}_rank", f"{role}_log_age", f"{role}_anchor", f"{role}_activity", f"{role}_distinct_labels"])
            role_anchors.append(anchors)
            role_activity.append(total_count)
        if transition is None:
            transition_stats = np.zeros((cap, 9), dtype=np.float32)
            transition_stats[:, STAT_AGE] = 3650.0
            transition_stats[:, STAT_RANK] = 128.0
        else:
            transition_stats = transition.match(candidates)
        self_probability = float(transition_stats[candidates == anchor, STAT_SHARE60][0]) if anchor in candidates else 0.0
        blocks.extend([
            np.full(cap, np.log1p(anchor_age), dtype=np.float32),
            np.full(cap, self.gap_bucket(anchor_age), dtype=np.float32),
            np.full(cap, 1.0 - self_probability, dtype=np.float32),
            (candidates == anchor).astype(np.float32),
            transition_stats[:, STAT_SHARE14], transition_stats[:, STAT_SHARE60], transition_stats[:, STAT_SHARE180], transition_stats[:, STAT_RANK],
        ])
        names.extend(["anchor_log_age", "anchor_gap_bucket", "change_hazard", "anchor_flag", "transition_share14", "transition_share60", "transition_share180", "transition_rank"])
        source_matrices = {}
        for source, records in (("product", products), ("category", categories)):
            maximum, mean, total_count, distinct = self.aggregate_records(records, candidates)
            source_matrices[source] = maximum
            blocks.extend([
                np.log1p(maximum[:, STAT_COUNT]), maximum[:, STAT_SHARE14], mean[:, STAT_SHARE14], maximum[:, STAT_SHARE60], mean[:, STAT_SHARE60], maximum[:, STAT_SHARE180], mean[:, STAT_SHARE180]
            ])
            names.extend([f"{source}_log_count", f"{source}_max_share14", f"{source}_mean_share14", f"{source}_max_share60", f"{source}_mean_share60", f"{source}_max_share180", f"{source}_mean_share180"])
        singular_matrices = {}
        for source in ("area", "doc_type", "geo", "billing", "currency"):
            if singular[source] is None:
                matrix = np.zeros((cap, 9), dtype=np.float32)
                matrix[:, STAT_AGE] = 3650.0
                matrix[:, STAT_RANK] = 128.0
            else:
                matrix = singular[source].match(candidates)
            singular_matrices[source] = matrix
            blocks.extend([matrix[:, STAT_SHARE14], matrix[:, STAT_SHARE60], matrix[:, STAT_SHARE180]])
            names.extend([f"{source}_share14", f"{source}_share60", f"{source}_share180"])
        global_matrix = state.global_record.match(candidates)
        blocks.extend([
            global_matrix[:, STAT_SHARE14], global_matrix[:, STAT_SHARE60], global_matrix[:, STAT_SHARE180],
            global_matrix[:, STAT_WINDOW30], global_matrix[:, STAT_WINDOW90], global_matrix[:, STAT_WINDOW180],
            singular_matrices["area"][:, STAT_WINDOW30], singular_matrices["area"][:, STAT_WINDOW90], singular_matrices["area"][:, STAT_WINDOW180],
        ])
        names.extend(["global_share14", "global_share60", "global_share180", "global_momentum30", "global_momentum90", "global_momentum180", "area_momentum30", "area_momentum90", "area_momentum180"])
        role_support = np.stack([role_matrices[role][:, STAT_SHARE60] for role in ROLE_NAMES], axis=0)
        role_anchor_matrix = np.stack(role_anchors, axis=0)
        support_count = (role_support > 0).sum(axis=0).astype(np.float32)
        strongest = role_support.max(axis=0)
        mean_support = role_support.mean(axis=0)
        anchor_count = role_anchor_matrix.sum(axis=0)
        raw_posterior = (
            4.0 * role_matrices["sold"][:, STAT_SHARE60]
            + 0.6 * role_matrices["payer"][:, STAT_SHARE60]
            + 0.6 * role_matrices["ship"][:, STAT_SHARE60]
            + 0.6 * role_matrices["bill"][:, STAT_SHARE60]
            + 1.2 * transition_stats[:, STAT_SHARE60]
            + 0.35 * source_matrices["product"][:, STAT_SHARE60]
            + 0.35 * source_matrices["category"][:, STAT_SHARE60]
            + 0.35 * singular_matrices["area"][:, STAT_SHARE60]
            + 0.20 * singular_matrices["geo"][:, STAT_SHARE60]
            + 0.15 * singular_matrices["doc_type"][:, STAT_SHARE60]
            + 0.15 * singular_matrices["billing"][:, STAT_SHARE60]
            + 0.10 * singular_matrices["currency"][:, STAT_SHARE60]
            + 0.20 * global_matrix[:, STAT_SHARE60]
        )
        raw_posterior = raw_posterior + 1e-8
        posterior_log = np.log(raw_posterior / raw_posterior.sum()).astype(np.float32)
        blocks.extend([support_count, strongest, mean_support, anchor_count, posterior_log])
        names.extend(["cross_role_support_count", "strongest_role_evidence", "mean_role_evidence", "cross_role_anchor_count", "hierarchical_log_posterior"])
        total_parties = sum(getattr(row, f"{role}_party_count") for role in ROLE_NAMES)
        query_values = {
            "item_count": math.log1p(float(row.item_count)),
            "distinct_products": float(row.distinct_products),
            "distinct_categories": float(row.distinct_categories),
            "total_parties": float(total_parties),
            "party_agreement": float(row.party_agreement),
            "days_into_horizon": float((row.time - state.cutoff).total_seconds() / 86400.0),
            "month": float(row.time.month),
            "day_of_week": float(row.time.dayofweek),
            "hour": float(row.time.hour),
            "sales_area_code": stable_code(row.sales_area),
            "doc_type_code": stable_code(row.doc_type),
            "sales_org_code": stable_code(row.sales_org),
            "channel_code": stable_code(row.channel),
            "division_code": stable_code(row.division),
            "billing_code": stable_code(row.billing_company),
            "currency_code": stable_code(row.currency),
            "country_code": stable_code(row.sold_country),
            "region_code": stable_code(row.sold_region),
        }
        for name, value in query_values.items():
            blocks.append(np.full(cap, value, dtype=np.float32))
            names.append(name)
        features = np.column_stack(blocks).astype(np.float32, copy=False)
        return candidates, posterior_log, features, names

    def build_pairs(self, frame, state, cap):
        query = frame.reset_index(drop=True)
        total = len(query) * cap
        features = None
        candidates = np.empty(total, dtype=np.int16)
        posterior = np.empty(total, dtype=np.float32)
        names = None
        build_started = time.time()
        for index, row in enumerate(query.itertuples(index=False)):
            row_candidates, row_posterior, row_features, row_names = self.pair_features(row, state, cap)
            if features is None:
                features = np.empty((total, row_features.shape[1]), dtype=np.float32)
                names = row_names
            elif row_names != names:
                raise RuntimeError("pair feature schema changed between rows")
            start = index * cap
            end = start + cap
            candidates[start:end] = row_candidates
            posterior[start:end] = row_posterior
            features[start:end] = row_features
        groups = np.full(len(query), cap, dtype=np.int32)
        elapsed = time.time() - build_started
        print(f"[matrix] built queries={len(query)} pairs={total} features={features.shape[1]} seconds={elapsed:.1f} queries_per_second={len(query) / max(elapsed, 1e-9):.1f}")
        return PairData(features, candidates, posterior, groups, query["document"].to_numpy(np.int64), names, query)

    def pair_recall(self, pair, targets):
        repeated = np.repeat(np.asarray(targets, dtype=np.int16), pair.groups)
        hit_rows = pair.candidates == repeated
        offsets = np.concatenate(([0], np.cumsum(pair.groups)))
        hits = np.add.reduceat(hit_rows.astype(np.int8), offsets[:-1]) > 0
        return float(hits.mean())

    def labels_for_pair(self, pair, frame):
        targets = frame["label"].to_numpy(np.int16)
        repeated = np.repeat(targets, pair.groups)
        labels = (pair.candidates == repeated).astype(np.int8)
        offsets = np.concatenate(([0], np.cumsum(pair.groups)))
        hit = np.add.reduceat(labels, offsets[:-1]) > 0
        return labels, hit

    def concatenate_training(self, pairs, frames):
        feature_parts = []
        label_parts = []
        group_parts = []
        for pair, frame in zip(pairs, frames):
            labels, hits = self.labels_for_pair(pair, frame)
            cap = int(pair.groups[0])
            if hits.all():
                feature_parts.append(pair.features)
                label_parts.append(labels)
                group_parts.append(pair.groups)
            else:
                reshaped = pair.features.reshape(len(pair.groups), cap, -1)
                label_reshaped = labels.reshape(len(pair.groups), cap)
                feature_parts.append(reshaped[hits].reshape(-1, pair.features.shape[1]))
                label_parts.append(label_reshaped[hits].reshape(-1))
                group_parts.append(pair.groups[hits])
        return np.concatenate(feature_parts), np.concatenate(label_parts), np.concatenate(group_parts)

    def parameters(self):
        return {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [1],
            "num_leaves": 127,
            "learning_rate": 0.05,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l2": 10.0,
            "verbosity": -1,
            "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
            "seed": 1337,
            "feature_fraction_seed": 1337,
            "bagging_seed": 1337,
            "data_random_seed": 1337,
            "deterministic": True,
            "force_col_wise": True,
            "max_bin": 127,
            "max_cat_to_onehot": 16,
            "cat_smooth": 20.0,
        }

    def categorical_names(self, names):
        return [name for name in names if name == "candidate"]

    def dataset(self, features, labels, groups, names, reference=None):
        return lgb.Dataset(features, label=labels, group=groups, feature_name=names, categorical_feature=self.categorical_names(names), reference=reference, free_raw_data=False)

    def select_iterations(self, pairs, frames):
        maximum = 50 if self.debug else 1200
        diagnostic_maximum = 50 if self.debug else 400
        first_x, first_y, first_g = self.concatenate_training([pairs[0]], [frames[0]])
        second_y, second_hit = self.labels_for_pair(pairs[1], frames[1])
        cap = int(pairs[1].groups[0])
        second_x = pairs[1].features.reshape(len(second_hit), cap, -1)[second_hit].reshape(-1, pairs[1].features.shape[1])
        second_labels = second_y.reshape(len(second_hit), cap)[second_hit].reshape(-1)
        second_groups = pairs[1].groups[second_hit]
        train_set = self.dataset(first_x, first_y, first_g, pairs[0].feature_names)
        valid_set = self.dataset(second_x, second_labels, second_groups, pairs[0].feature_names, train_set)
        diagnostic = lgb.train(self.parameters(), train_set, num_boost_round=diagnostic_maximum, valid_sets=[valid_set], callbacks=[lgb.early_stopping(50 if not self.debug else 10, verbose=False), lgb.log_evaluation(0)])
        first_two_x, first_two_y, first_two_g = self.concatenate_training(pairs[:2], frames[:2])
        third_y, third_hit = self.labels_for_pair(pairs[2], frames[2])
        third_x = pairs[2].features.reshape(len(third_hit), cap, -1)[third_hit].reshape(-1, pairs[2].features.shape[1])
        third_labels = third_y.reshape(len(third_hit), cap)[third_hit].reshape(-1)
        third_groups = pairs[2].groups[third_hit]
        train_latest = self.dataset(first_two_x, first_two_y, first_two_g, pairs[0].feature_names)
        valid_latest = self.dataset(third_x, third_labels, third_groups, pairs[0].feature_names, train_latest)
        selection = lgb.train(self.parameters(), train_latest, num_boost_round=maximum, valid_sets=[valid_latest], callbacks=[lgb.early_stopping(100 if not self.debug else 10, verbose=False), lgb.log_evaluation(0)])
        rounds = max(1, int(selection.best_iteration or maximum))
        print(f"[selection] diagnostic_rounds={diagnostic.best_iteration} latest_origin_rounds={rounds}")
        self.origin_predictions = {
            1: diagnostic.predict(pairs[1].features, num_iteration=diagnostic.best_iteration),
            2: selection.predict(pairs[2].features, num_iteration=rounds),
        }
        return rounds, diagnostic, selection

    def choice_accuracy(self, pair, margin, targets, blend):
        cap = int(pair.groups[0])
        rank = margin.reshape(-1, cap)
        rank_log = np.log(np.maximum(softmax_rows(rank), 1e-12))
        posterior = pair.posterior_log.reshape(-1, cap)
        combined = (1.0 - blend) * rank_log + blend * posterior
        prediction = pair.candidates.reshape(-1, cap)[np.arange(len(pair.groups)), combined.argmax(axis=1)]
        return prediction, float((prediction == targets).mean())

    def paired_gate(self, baseline_predictions, candidate_predictions, targets):
        differences = (candidate_predictions == targets).astype(np.float32) - (baseline_predictions == targets).astype(np.float32)
        mean = float(differences.mean())
        se = float(differences.std(ddof=1) / math.sqrt(max(len(differences), 1))) if len(differences) > 1 else 0.0
        return mean, se, mean >= max(0.002, 2.0 * se)

    def select_postprocessing(self, pairs, frames, states, diagnostic, selection):
        targets = {index: frames[index]["label"].to_numpy(np.int16) for index in (1, 2)}
        margins = self.origin_predictions
        blend_candidates = [0.0, 0.15, 0.30]
        predictions = {}
        accuracies = {}
        for blend in blend_candidates:
            accuracies[blend] = []
            predictions[blend] = {}
            for index in (1, 2):
                pred, accuracy = self.choice_accuracy(pairs[index], margins[index], targets[index], blend)
                predictions[blend][index] = pred
                accuracies[blend].append(accuracy)
        selected_blend = 0.15
        best = max(blend_candidates, key=lambda value: float(np.mean(accuracies[value])))
        if best != selected_blend:
            differences = []
            gains = []
            for index in (1, 2):
                difference = (predictions[best][index] == targets[index]).astype(np.float32) - (predictions[selected_blend][index] == targets[index]).astype(np.float32)
                differences.append(difference)
                gains.append(float(difference.mean()))
            combined = np.concatenate(differences)
            standard_error = float(combined.std(ddof=1) / math.sqrt(len(combined)))
            mean_gain = float(np.mean(gains))
            if sum(gain > 0.0 for gain in gains) >= 2 and mean_gain >= max(0.002, 2.0 * standard_error):
                selected_blend = best
        pool_candidates = [0.0, 0.15, 0.30, 0.50]
        pool_accuracies = {value: [] for value in pool_candidates}
        pool_predictions = {value: {} for value in pool_candidates}
        for index in (1, 2):
            dense = self.compose_scores(pairs[index], margins[index], states[index], selected_blend)
            for weight in pool_candidates:
                pooled = self.forward_pool(dense.copy(), pairs[index].query_frame, weight)
                pred = pooled.argmax(axis=1).astype(np.int16)
                pool_predictions[weight][index] = pred
                pool_accuracies[weight].append(float((pred == targets[index]).mean()))
        selected_pool = 0.0
        best_pool = max(pool_candidates, key=lambda value: float(np.mean(pool_accuracies[value])))
        if best_pool != 0.0:
            differences = []
            gains = []
            for index in (1, 2):
                difference = (pool_predictions[best_pool][index] == targets[index]).astype(np.float32) - (pool_predictions[0.0][index] == targets[index]).astype(np.float32)
                differences.append(difference)
                gains.append(float(difference.mean()))
            combined = np.concatenate(differences)
            standard_error = float(combined.std(ddof=1) / math.sqrt(len(combined)))
            mean_gain = float(np.mean(gains))
            if sum(gain > 0.0 for gain in gains) >= 2 and mean_gain >= max(0.002, 2.0 * standard_error):
                selected_pool = best_pool
        print(f"[selection] blend_accuracies={accuracies} selected_blend={selected_blend}")
        print(f"[selection] pool_accuracies={pool_accuracies} selected_pool={selected_pool}")
        self.diagnostics["internal_blend_accuracies"] = {str(key): value for key, value in accuracies.items()}
        self.diagnostics["internal_pool_accuracies"] = {str(key): value for key, value in pool_accuracies.items()}
        for index in (1, 2):
            self.report_strata(index, pairs[index], targets[index], pool_predictions[selected_pool][index], states[index])
        return selected_blend, selected_pool

    def report_strata(self, index, pair, targets, predictions, state):
        frame = pair.query_frame
        feature_index = pair.feature_names.index("anchor_log_age")
        cap = int(pair.groups[0])
        ages = np.expm1(pair.features.reshape(len(frame), cap, -1)[:, 0, feature_index])
        strata = {
            "cold": ages >= 3649,
            "age_0_30": ages <= 30,
            "age_31_90": (ages > 30) & (ages <= 90),
            "age_91_180": (ages > 90) & (ages <= 180),
            "age_181_plus": (ages > 180) & (ages < 3649),
        }
        output = {}
        for name, mask in strata.items():
            output[name] = {"count": int(mask.sum()), "accuracy": float((predictions[mask] == targets[mask]).mean()) if mask.any() else 0.0}
        self.diagnostics["origins"][str(index)]["accuracy_strata"] = output
        print(f"[strata] origin={index} values={output}")

    def refit_model(self, pairs, frames, rounds, name):
        train_x, train_y, train_g = self.concatenate_training(pairs, frames)
        train_set = self.dataset(train_x, train_y, train_g, pairs[0].feature_names)
        model = lgb.train(self.parameters(), train_set, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
        local_path = self.output / f"{name}.txt"
        cache_path = self.cache_root / f"{name}_{'debug' if self.debug else 'full'}.txt"
        model.save_model(str(local_path))
        model.save_model(str(cache_path))
        print(f"[model] name={name} rounds={rounds} queries={len(train_g)} pairs={len(train_y)}")
        return model

    def compose_scores(self, pair, margin, state, blend):
        cap = int(pair.groups[0])
        query_count = len(pair.groups)
        global_labels = state.global_record.labels.astype(np.int64)
        global_share = state.global_record.stats[:, STAT_SHARE60]
        global_prior = np.full(502, 1e-12, dtype=np.float32)
        global_prior[global_labels] = np.maximum(global_share, 1e-12)
        global_prior /= global_prior.sum()
        background = blend * np.log(global_prior) - (1.0 - blend) * 18.0
        scores = np.tile(background, (query_count, 1)).astype(np.float32)
        rank = np.asarray(margin).reshape(query_count, cap)
        rank_log = np.log(np.maximum(softmax_rows(rank), 1e-12))
        posterior = pair.posterior_log.reshape(query_count, cap)
        candidate_scores = (1.0 - blend) * rank_log + blend * posterior
        candidate_matrix = pair.candidates.reshape(query_count, cap)
        rows = np.repeat(np.arange(query_count), cap)
        scores[rows, candidate_matrix.reshape(-1)] = candidate_scores.reshape(-1)
        return scores

    def forward_pool(self, scores, frame, weight):
        if weight <= 0.0:
            return scores
        order = np.argsort(frame["time"].to_numpy(), kind="stable")
        probabilities = softmax_rows(scores)
        sums = {}
        counts = {}
        times = frame["time"].to_numpy()
        positions = 0
        while positions < len(order):
            end = positions + 1
            timestamp = times[order[positions]]
            while end < len(order) and times[order[end]] == timestamp:
                end += 1
            batch = order[positions:end]
            updates = []
            for row_index in batch:
                parties = sequence_values(frame.iloc[row_index]["sold_parties"])
                party = parties[0] if len(parties) else None
                current = probabilities[row_index]
                if party in sums:
                    prior = sums[party] / counts[party]
                    mixed = (1.0 - weight) * current + weight * prior
                    mixed /= mixed.sum()
                    probabilities[row_index] = mixed
                updates.append((party, probabilities[row_index].copy()))
            for party, value in updates:
                if party is None:
                    continue
                if party not in sums:
                    sums[party] = value
                    counts[party] = 1.0
                else:
                    sums[party] += value
                    counts[party] += 1.0
            positions = end
        return np.log(np.maximum(probabilities, 1e-12)).astype(np.float32)

    def load_validation_labels(self):
        path = self.data_root / "tasks" / os.environ["RELBENCH_TASK"] / "val.parquet"
        frame = pd.read_parquet(path, columns=["CREATIONTIMESTAMP", "SALESDOCUMENT", "SALESGROUP"]).rename(columns={"CREATIONTIMESTAMP": "time", "SALESDOCUMENT": "document", "SALESGROUP": "label"})
        frame["label"] = frame["label"].astype(np.int16)
        frame["row_index"] = np.arange(len(frame), dtype=np.int32)
        print(f"[provenance] validation labels loaded for model B only rows={len(frame)}")
        return frame

    def validate_scores(self, scores, rows, name):
        expected = (rows, 502)
        if scores.shape != expected:
            raise RuntimeError(f"{name} scores shape {scores.shape} != {expected}")
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"{name} scores contain non-finite values")

    def file_hash(self, path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    def validate_artifacts(self, val_rows, test_rows, expected_hash):
        val_path = self.output / "val_predictions.npy"
        test_path = self.output / "test_predictions.npy"
        if self.file_hash(val_path) != expected_hash:
            raise RuntimeError("final validation artifact provenance mismatch")
        validation = np.load(val_path, allow_pickle=False, mmap_mode="r")
        test = np.load(test_path, allow_pickle=False, mmap_mode="r")
        self.validate_scores(validation, val_rows, "validation artifact")
        self.validate_scores(test, test_rows, "test artifact")
        if validation.dtype != np.float32 or test.dtype != np.float32:
            raise RuntimeError("prediction artifacts must be float32")
        print(f"[contract] val_shape={validation.shape} test_shape={test.shape} dtype=float32 finite=true alignment=preserved")

    def register_artifacts(self):
        registry = self.cache / "artifacts.json"
        lock_path = self.cache / "artifacts.lock"
        entries = [
            {"name": f"{VERSION}-document-features", "path": f"{VERSION}/document_features.pkl", "description": "Permitted header, same-document item, four-role customer, and geography features", "content_key": VERSION, "rebuild_hint": "Run main.py; extraction is rebuilt when the version changes"},
            {"name": f"{VERSION}-states", "path": VERSION, "description": "Frozen causal support states, ranker models, and model-A/model-B predictions", "content_key": VERSION, "rebuild_hint": "Run main.py with the sanitized rel-salt sales-group cache"},
        ]
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if registry.exists():
                try:
                    current = json.loads(registry.read_text())
                except json.JSONDecodeError:
                    current = []
            else:
                current = []
            names = {entry.get("name") for entry in current}
            current.extend(entry for entry in entries if entry["name"] not in names)
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(current, indent=2))
            temporary.replace(registry)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
