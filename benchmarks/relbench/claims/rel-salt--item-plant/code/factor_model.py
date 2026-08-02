from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


CLASSES = 34
FLOOR = 1e-8
ORG_ALPHA = 2.0
INTERACTION_ALPHA = 16.0

FEATURE_NAMES = [
    "organization",
    "document_organization_product",
    "document_organization_ship_product",
    "document_organization_sold_product",
    "document_product",
    "document_ship_product",
    "document_sold_product",
    "document_organization_category",
    "document_category",
    "item_organization_channel",
    "item_organization_product",
    "item_organization_category",
    "item_organization_ship_product",
    "item_organization_sold_product",
    "item_product",
    "item_parties",
    "item_party_product",
    "item_countries",
    "item_regions",
    "item_country_product",
    "item_region_product",
    "item_document_type",
    "item_billing_company",
    "item_currency",
    "item_category",
    "item_position",
    "item_sibling_disagreement",
    "item_organization_document_type",
    "item_organization_billing_company",
    "item_organization_currency",
]

INITIAL_WEIGHTS = np.asarray(
    [
        6.0,
        0.5,
        0.25,
        0.25,
        0.05,
        0.1,
        0.1,
        0.25,
        0.1,
        0.2,
        1.0,
        0.5,
        0.5,
        0.5,
        0.1,
        0.02,
        0.1,
        0.02,
        0.02,
        0.05,
        0.05,
        0.1,
        0.1,
        0.05,
        0.1,
        0.05,
        0.05,
        0.2,
        0.2,
        0.1,
    ],
    dtype=np.float64,
)

KEY_COLUMNS = {
    "org": ("org",),
    "org_channel": ("org", "channel"),
    "org_product": ("org", "product"),
    "org_doctype_product": ("org", "doctype", "product"),
    "org_category": ("org", "category"),
    "org_ship_product": ("org", "ship", "product"),
    "org_sold_product": ("org", "sold", "product"),
    "product": ("product",),
    "category": ("category",),
    "position": ("position",),
    "doctype": ("doctype",),
    "billing": ("billing",),
    "currency": ("currency",),
    "org_doctype": ("org", "doctype"),
    "org_billing": ("org", "billing"),
    "org_currency": ("org", "currency"),
    "sold": ("sold",),
    "ship": ("ship",),
    "bill_party": ("bill_party",),
    "payer": ("payer",),
    "sold_product": ("sold", "product"),
    "ship_product": ("ship", "product"),
    "bill_party_product": ("bill_party", "product"),
    "payer_product": ("payer", "product"),
    "sold_country": ("sold_country",),
    "ship_country": ("ship_country",),
    "bill_country": ("bill_country",),
    "payer_country": ("payer_country",),
    "sold_region": ("sold_region",),
    "ship_region": ("ship_region",),
    "bill_region": ("bill_region",),
    "payer_region": ("payer_region",),
    "sold_country_product": ("sold_country", "product"),
    "ship_country_product": ("ship_country", "product"),
    "bill_country_product": ("bill_country", "product"),
    "payer_country_product": ("payer_country", "product"),
    "sold_region_product": ("sold_region", "product"),
    "ship_region_product": ("ship_region", "product"),
    "bill_region_product": ("bill_region", "product"),
    "payer_region_product": ("payer_region", "product"),
    "signature": ("product", "category", "sold", "ship", "bill_party", "payer"),
}

GROUP_FACTORS = {
    "parties": ("sold", "ship", "bill_party", "payer"),
    "party_product": ("sold_product", "ship_product", "bill_party_product", "payer_product"),
    "countries": ("sold_country", "ship_country", "bill_country", "payer_country"),
    "regions": ("sold_region", "ship_region", "bill_region", "payer_region"),
    "country_product": (
        "sold_country_product",
        "ship_country_product",
        "bill_country_product",
        "payer_country_product",
    ),
    "region_product": (
        "sold_region_product",
        "ship_region_product",
        "bill_region_product",
        "payer_region_product",
    ),
}

DOCUMENT_FACTORS = {
    "org_product": 1,
    "org_ship_product": 2,
    "org_sold_product": 3,
    "product": 4,
    "ship_product": 5,
    "sold_product": 6,
    "org_category": 7,
    "category": 8,
}

ITEM_FACTORS = {
    "org_channel": 9,
    "org_product": 10,
    "org_category": 11,
    "org_ship_product": 12,
    "org_sold_product": 13,
    "product": 14,
    "parties": 15,
    "party_product": 16,
    "countries": 17,
    "regions": 18,
    "country_product": 19,
    "region_product": 20,
    "doctype": 21,
    "billing": 22,
    "currency": 23,
    "category": 24,
    "position": 25,
    "org_doctype": 27,
    "org_billing": 28,
    "org_currency": 29,
}


def _codes(values) -> np.ndarray:
    result = pd.factorize(values, sort=False, use_na_sentinel=True)[0].astype(np.int32) + 1
    return result


def _integer_values(values) -> np.ndarray:
    if hasattr(values, "to_numpy"):
        result = values.to_numpy(dtype=np.int64, na_value=-1)
    else:
        result = np.asarray(values, dtype=np.int64)
    return np.where(result >= 0, result + 1, 0).astype(np.int32)


class FeatureStore:
    def __init__(self, db):
        started = time.time()
        item = db.table_dict["salesdocumentitem"].df
        header = db.table_dict["salesdocument"].df
        customer = db.table_dict["customer"].df
        address = db.table_dict["address"].df
        item_ids = item["ID"].to_numpy(dtype=np.int64)
        header_ids = header["SALESDOCUMENT"].to_numpy(dtype=np.int64)
        if not np.array_equal(item_ids, np.arange(len(item), dtype=np.int64)):
            raise RuntimeError("salesdocumentitem IDs are not positional")
        if not np.array_equal(header_ids, np.arange(len(header), dtype=np.int64)):
            raise RuntimeError("salesdocument IDs are not positional")
        doc = item["SALESDOCUMENT"].to_numpy(dtype=np.int32)
        raw_parties = {
            "sold": _integer_values(item["SOLDTOPARTY"]),
            "ship": _integer_values(item["SHIPTOPARTY"]),
            "bill_party": _integer_values(item["BILLTOPARTY"]),
            "payer": _integer_values(item["PAYERPARTY"]),
        }
        customer_ids = customer["CUSTOMER"].to_numpy(dtype=np.int64)
        customer_addresses = customer["ADDRESSID"].to_numpy(dtype=np.int64, na_value=-1)
        max_customer = int(max(customer_ids.max(), max(int(v.max()) - 1 for v in raw_parties.values())))
        address_for_customer = np.full(max_customer + 2, -1, dtype=np.int64)
        address_for_customer[customer_ids + 1] = customer_addresses
        address_ids = address["ADDRESSID"].to_numpy(dtype=np.int64)
        max_address = int(max(address_ids.max(), customer_addresses.max()))
        country_for_address = np.zeros(max_address + 1, dtype=np.int32)
        region_for_address = np.zeros(max_address + 1, dtype=np.int32)
        country_for_address[address_ids] = _codes(address["COUNTRY"])
        region_for_address[address_ids] = _codes(address["REGION"])
        arrays = {
            "doc": doc,
            "product": _codes(item["PRODUCT"]),
            "category": _codes(item["SALESDOCUMENTITEMCATEGORY"]),
            "position": _codes(item["SALESDOCUMENTITEM"]),
            "org": _codes(header["SALESORGANIZATION"])[doc],
            "channel": _codes(header["DISTRIBUTIONCHANNEL"])[doc],
            "doctype": _codes(header["SALESDOCUMENTTYPE"])[doc],
            "billing": _codes(header["BILLINGCOMPANYCODE"])[doc],
            "currency": _codes(header["TRANSACTIONCURRENCY"])[doc],
        }
        arrays.update(raw_parties)
        for role, party_values in raw_parties.items():
            addresses = address_for_customer[party_values]
            valid = (addresses >= 0) & (addresses < len(country_for_address))
            countries = np.zeros(len(item), dtype=np.int32)
            regions = np.zeros(len(item), dtype=np.int32)
            countries[valid] = country_for_address[addresses[valid]]
            regions[valid] = region_for_address[addresses[valid]]
            prefix = "bill" if role == "bill_party" else role
            arrays[f"{prefix}_country"] = countries
            arrays[f"{prefix}_region"] = regions
        self.arrays = arrays
        self.key_cache: dict[str, np.ndarray] = {}
        self.n_rows = len(item)
        self.n_org = int(arrays["org"].max()) + 1
        self.n_doctype = int(arrays["doctype"].max()) + 1
        print(f"[factor] feature store rows={self.n_rows} elapsed={time.time() - started:.2f}s")

    def key_code(self, name: str) -> np.ndarray:
        if name in self.key_cache:
            return self.key_cache[name]
        columns = KEY_COLUMNS[name]
        if len(columns) == 1:
            result = self.arrays[columns[0]]
        else:
            combined = self.arrays[columns[0]].astype(np.uint64)
            for column in columns[1:]:
                values = self.arrays[column].astype(np.uint64)
                radix = np.uint64(int(values.max()) + 1)
                combined = combined * radix + values
            result = pd.factorize(combined, sort=False, use_na_sentinel=False)[0].astype(np.int32) + 1
        self.key_cache[name] = result
        return result


@dataclass
class FactorCounts:
    weighted: np.ndarray
    raw_support: np.ndarray
    weighted_support: np.ndarray


class HistoryModel:
    def __init__(
        self,
        store: FeatureStore,
        ids: np.ndarray,
        labels: np.ndarray,
        timestamps: np.ndarray,
        cutoff: np.datetime64,
        name: str,
    ):
        self.store = store
        self.ids = np.asarray(ids, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.timestamps = np.asarray(timestamps, dtype="datetime64[ns]")
        self.cutoff = np.datetime64(cutoff, "ns")
        self.name = name
        if np.any(self.timestamps >= self.cutoff):
            raise RuntimeError(f"history {name} contains labels at or after its cutoff")
        age = np.maximum(
            0.0,
            (self.cutoff - self.timestamps) / np.timedelta64(1, "D"),
        ).astype(np.float64)
        self.recency = (
            0.5 * np.exp2(-age / 90.0) + 0.5 * np.exp2(-age / 180.0)
        ).astype(np.float32)
        global_counts = np.bincount(
            self.labels, weights=self.recency, minlength=CLASSES
        ).astype(np.float64)
        global_counts += FLOOR
        self.global_prior = (global_counts / global_counts.sum()).astype(np.float32)
        self.map_cache: dict[str, FactorCounts] = {}
        self.raw_class_cache: dict[str, np.ndarray] = {}
        print(f"[factor] history={name} rows={len(self.ids)} cutoff={self.cutoff}")

    def counts(self, name: str) -> FactorCounts:
        if name in self.map_cache:
            return self.map_cache[name]
        all_codes = self.store.key_code(name)
        codes = all_codes[self.ids].astype(np.int64)
        n_codes = int(all_codes.max()) + 1
        flat = codes * CLASSES + self.labels
        weighted = np.bincount(
            flat,
            weights=self.recency,
            minlength=n_codes * CLASSES,
        ).reshape(n_codes, CLASSES).astype(np.float32)
        raw_support = np.bincount(codes, minlength=n_codes).astype(np.float32)
        weighted_support = weighted.sum(axis=1, dtype=np.float32)
        result = FactorCounts(weighted, raw_support, weighted_support)
        self.map_cache[name] = result
        return result

    def raw_class_counts(self, name: str) -> np.ndarray:
        if name in self.raw_class_cache:
            return self.raw_class_cache[name]
        all_codes = self.store.key_code(name)
        codes = all_codes[self.ids].astype(np.int64)
        n_codes = int(all_codes.max()) + 1
        flat = codes * CLASSES + self.labels
        result = np.bincount(
            flat,
            minlength=n_codes * CLASSES,
        ).reshape(n_codes, CLASSES).astype(np.int32)
        self.raw_class_cache[name] = result
        return result

    def raw_posterior(self, name: str, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = self.counts(name)
        codes = self.store.key_code(name)[target_ids]
        values = counts.weighted[codes]
        support = counts.weighted_support[codes]
        posterior = (
            values + INTERACTION_ALPHA * self.global_prior[None, :]
        ) / (support[:, None] + INTERACTION_ALPHA)
        posterior = self._normalize(posterior)
        reliability = self._reliability(counts, codes, posterior)
        return posterior, reliability

    def organization_posterior(self, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        counts = self.counts("org")
        codes = self.store.key_code("org")[target_ids]
        values = counts.weighted[codes]
        support = counts.weighted_support[codes]
        posterior = (
            values + ORG_ALPHA * self.global_prior[None, :]
        ) / (support[:, None] + ORG_ALPHA)
        cold = counts.raw_support[codes] == 0
        if np.any(cold):
            cold_ids = target_ids[cold]
            numerator = 0.25 * np.repeat(self.global_prior[None, :], len(cold_ids), axis=0)
            denominator = np.full(len(cold_ids), 0.25, dtype=np.float32)
            for name in (
                "doctype",
                "billing",
                "currency",
                "ship_country",
                "sold_country",
                "product",
            ):
                candidate, reliability = self.raw_posterior(name, cold_ids)
                numerator += reliability[:, None] * candidate
                denominator += reliability
            posterior[cold] = numerator / denominator[:, None]
        posterior = self._normalize(posterior)
        reliability = self._reliability(counts, codes, posterior)
        return posterior, reliability

    def posterior(
        self,
        name: str,
        target_ids: np.ndarray,
        organization: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if name == "org":
            return self.organization_posterior(target_ids)
        if organization is None:
            organization = self.organization_posterior(target_ids)[0]
        counts = self.counts(name)
        codes = self.store.key_code(name)[target_ids]
        values = counts.weighted[codes]
        support = counts.weighted_support[codes]
        posterior = (
            values + INTERACTION_ALPHA * organization
        ) / (support[:, None] + INTERACTION_ALPHA)
        posterior = self._normalize(posterior)
        reliability = self._reliability(counts, codes, posterior)
        return posterior, reliability

    def family(
        self,
        name: str,
        target_ids: np.ndarray,
        organization: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        names = GROUP_FACTORS.get(name, (name,))
        logarithm = np.zeros((len(target_ids), CLASSES), dtype=np.float32)
        reliability = np.zeros(len(target_ids), dtype=np.float32)
        for factor in names:
            posterior, factor_reliability = self.posterior(factor, target_ids, organization)
            logarithm += np.log(np.maximum(posterior, FLOOR)).astype(np.float32)
            reliability += factor_reliability
        scale = float(len(names))
        return logarithm / scale, reliability / scale

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        result = np.maximum(values, FLOOR)
        return result / result.sum(axis=1, keepdims=True)

    @staticmethod
    def _reliability(
        counts: FactorCounts,
        codes: np.ndarray,
        posterior: np.ndarray,
    ) -> np.ndarray:
        raw = counts.raw_support[codes]
        weighted = counts.weighted_support[codes]
        partition = np.partition(posterior, -2, axis=1)
        margin = partition[:, -1] - partition[:, -2]
        entropy = -np.sum(posterior * np.log(np.maximum(posterior, FLOOR)), axis=1) / np.log(CLASSES)
        age = np.sqrt(np.clip(weighted / np.maximum(raw, 1.0), 0.0, 1.0))
        reliability = (
            np.minimum(3.0, np.log1p(raw))
            * np.maximum(0.05, margin)
            * (0.5 + 0.5 * (1.0 - entropy))
            * age
        )
        reliability[raw == 0] = 0.0
        return reliability.astype(np.float32)


@dataclass
class GateFeatures:
    documents: np.ndarray
    inverse: np.ndarray
    sizes: np.ndarray
    matrix: np.ndarray


class ConstantGate:
    def __init__(self, probability: float = 0.0):
        self.probability = float(probability)

    def predict_proba(self, matrix: np.ndarray) -> np.ndarray:
        positive = np.full(len(matrix), self.probability, dtype=np.float64)
        return np.column_stack([1.0 - positive, positive])


@dataclass
class GateModel:
    estimator: object
    prior: float

    def balanced_probability(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(matrix)[:, 1], dtype=np.float64)

    def probability(self, matrix: np.ndarray) -> np.ndarray:
        balanced = np.clip(self.balanced_probability(matrix), 1e-7, 1.0 - 1e-7)
        prior_odds = self.prior / max(1.0 - self.prior, 1e-12)
        odds = balanced / (1.0 - balanced) * prior_odds
        return odds / (1.0 + odds)


def document_labels(store: FeatureStore, ids: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    documents = store.arrays["doc"][ids]
    unique_documents, inverse = np.unique(documents, return_inverse=True)
    minimum = np.full(len(unique_documents), CLASSES, dtype=np.int16)
    maximum = np.full(len(unique_documents), -1, dtype=np.int16)
    np.minimum.at(minimum, inverse, labels.astype(np.int16))
    np.maximum.at(maximum, inverse, labels.astype(np.int16))
    return unique_documents, (minimum != maximum).astype(np.int8)


def select_gate_training_rows(
    store: FeatureStore,
    ids: np.ndarray,
    labels: np.ndarray,
    limit: int,
) -> np.ndarray:
    if len(ids) <= limit:
        return np.arange(len(ids), dtype=np.int64)
    documents = store.arrays["doc"][ids]
    unique_documents, multi = document_labels(store, ids, labels)
    positive_documents = unique_documents[multi == 1]
    recent = np.arange(len(ids) - limit, len(ids), dtype=np.int64)
    if len(positive_documents) == 0:
        return recent
    positive = np.flatnonzero(np.isin(documents, positive_documents))
    return np.unique(np.concatenate([recent, positive]))


def _per_document_nunique(inverse: np.ndarray, values: np.ndarray, n_documents: int) -> np.ndarray:
    radix = int(values.max()) + 1
    pairs = inverse.astype(np.int64) * radix + values.astype(np.int64)
    unique_pairs = np.unique(pairs)
    owners = unique_pairs // radix
    return np.bincount(owners, minlength=n_documents).astype(np.float32)


def build_gate_features(
    history: HistoryModel,
    target_ids: np.ndarray,
) -> GateFeatures:
    store = history.store
    documents = store.arrays["doc"][target_ids]
    unique_documents, first, inverse, sizes = np.unique(
        documents,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    n_documents = len(unique_documents)
    organization, _ = history.organization_posterior(target_ids)
    organization_prediction = organization.argmax(axis=1)
    disagreement = np.zeros(len(target_ids), dtype=np.float32)
    entropy = np.zeros(len(target_ids), dtype=np.float32)
    margin = np.zeros(len(target_ids), dtype=np.float32)
    maximum_disagreement = np.zeros(len(target_ids), dtype=np.float32)
    for name in ("org_product", "org_ship_product", "org_sold_product", "product"):
        posterior, _ = history.posterior(name, target_ids, organization)
        current = (posterior.argmax(axis=1) != organization_prediction).astype(np.float32)
        disagreement += current
        maximum_disagreement = np.maximum(maximum_disagreement, current)
        entropy += -np.sum(
            posterior * np.log(np.maximum(posterior, FLOOR)), axis=1
        ).astype(np.float32) / np.log(CLASSES)
        partition = np.partition(posterior, -2, axis=1)
        margin += partition[:, -1] - partition[:, -2]
    disagreement /= 4.0
    entropy /= 4.0
    margin /= 4.0
    mean_disagreement = np.bincount(inverse, weights=disagreement, minlength=n_documents) / sizes
    max_disagreement = np.zeros(n_documents, dtype=np.float32)
    np.maximum.at(max_disagreement, inverse, maximum_disagreement)
    mean_entropy = np.bincount(inverse, weights=entropy, minlength=n_documents) / sizes
    mean_margin = np.bincount(inverse, weights=margin, minlength=n_documents) / sizes
    unique_product = _per_document_nunique(inverse, store.arrays["product"][target_ids], n_documents)
    unique_category = _per_document_nunique(inverse, store.arrays["category"][target_ids], n_documents)
    unique_ship = _per_document_nunique(inverse, store.arrays["ship"][target_ids], n_documents)
    unique_sold = _per_document_nunique(inverse, store.arrays["sold"][target_ids], n_documents)
    signature = store.key_code("signature")[target_ids]
    unique_signature = _per_document_nunique(inverse, signature, n_documents)
    continuous = np.column_stack(
        [
            np.log1p(sizes),
            np.log1p(unique_product),
            np.log1p(unique_category),
            np.log1p(unique_ship),
            np.log1p(unique_sold),
            np.log1p(unique_signature),
            unique_product / sizes,
            (unique_ship + unique_sold) / (2.0 * sizes),
            mean_disagreement,
            max_disagreement,
            mean_entropy,
            mean_margin,
        ]
    ).astype(np.float32)
    organization_codes = store.arrays["org"][target_ids][first]
    doctype_codes = store.arrays["doctype"][target_ids][first]
    matrix = np.zeros(
        (n_documents, continuous.shape[1] + store.n_org + store.n_doctype),
        dtype=np.float32,
    )
    matrix[:, : continuous.shape[1]] = continuous
    rows = np.arange(n_documents)
    matrix[rows, continuous.shape[1] + organization_codes] = 1.0
    matrix[rows, continuous.shape[1] + store.n_org + doctype_codes] = 1.0
    return GateFeatures(unique_documents, inverse, sizes.astype(np.int32), matrix)


def fit_gate(features: GateFeatures, labels: np.ndarray, c_value: float) -> GateModel:
    prior = float(np.mean(labels))
    if labels.min(initial=0) == labels.max(initial=0):
        return GateModel(ConstantGate(prior), prior)
    estimator = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=150,
        tol=1e-6,
    )
    estimator.fit(features.matrix, labels)
    return GateModel(estimator, prior)


def gate_coherence(gate: GateModel, features: GateFeatures) -> np.ndarray:
    exception = gate.probability(features.matrix)
    document_coherence = np.clip(1.0 - exception, 0.05, 1.0)
    document_coherence[features.sizes <= 1] = 0.0
    return document_coherence[features.inverse].astype(np.float32)


def _document_aggregate(
    logarithm: np.ndarray,
    reliability: np.ndarray,
    inverse: np.ndarray,
    first_signature: np.ndarray,
    n_documents: int,
) -> np.ndarray:
    document_for_signature = inverse[first_signature]
    order = np.argsort(document_for_signature, kind="stable")
    ordered_documents = document_for_signature[order]
    ordered_values = (
        reliability[first_signature[order], None] * logarithm[first_signature[order]]
    )
    unique_documents, starts = np.unique(ordered_documents, return_index=True)
    reduced = np.add.reduceat(ordered_values, starts, axis=0)
    result = np.zeros((n_documents, CLASSES), dtype=np.float32)
    result[unique_documents] = reduced
    return result


def build_design(
    history: HistoryModel,
    target_ids: np.ndarray,
    gate: GateModel,
) -> np.ndarray:
    started = time.time()
    store = history.store
    target_ids = np.asarray(target_ids, dtype=np.int64)
    organization, _ = history.organization_posterior(target_ids)
    design = np.empty((len(target_ids), CLASSES, len(FEATURE_NAMES)), dtype=np.float32)
    design[:, :, 0] = np.log(np.maximum(organization, FLOOR)).astype(np.float32)
    gate_features = build_gate_features(history, target_ids)
    coherence = gate_coherence(gate, gate_features)
    documents = store.arrays["doc"][target_ids]
    unique_documents, document_inverse = np.unique(documents, return_inverse=True)
    signature = store.key_code("signature")[target_ids]
    radix = int(signature.max()) + 1
    pairs = documents.astype(np.int64) * radix + signature.astype(np.int64)
    _, first_signature = np.unique(pairs, return_index=True)
    factors = list(dict.fromkeys(list(DOCUMENT_FACTORS) + list(ITEM_FACTORS)))
    for name in factors:
        logarithm, reliability = history.family(name, target_ids, organization)
        item_index = ITEM_FACTORS.get(name)
        if item_index is not None:
            design[:, :, item_index] = logarithm
        document_index = DOCUMENT_FACTORS.get(name)
        if document_index is not None:
            aggregate = _document_aggregate(
                logarithm,
                reliability,
                document_inverse,
                first_signature,
                len(unique_documents),
            )
            design[:, :, document_index] = aggregate[document_inverse] * coherence[:, None]
        if name == "org_product":
            ordered_documents = document_inverse[first_signature]
            order = np.argsort(ordered_documents, kind="stable")
            ordered_documents = ordered_documents[order]
            unique_owner, starts, counts = np.unique(
                ordered_documents,
                return_index=True,
                return_counts=True,
            )
            mean = np.zeros((len(unique_documents), CLASSES), dtype=np.float32)
            mean[unique_owner] = np.add.reduceat(
                logarithm[first_signature[order]], starts, axis=0
            ) / counts[:, None]
            design[:, :, 26] = logarithm - mean[document_inverse]
    print(
        f"[factor] design history={history.name} rows={len(target_ids)} "
        f"features={design.shape[2]} elapsed={time.time() - started:.2f}s"
    )
    return design


def fit_nonnegative_weights(
    design: np.ndarray,
    labels: np.ndarray,
    row_weights: np.ndarray,
    regularization: float,
    initial: np.ndarray = INITIAL_WEIGHTS,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    rows = np.arange(len(labels))
    flat_design = design.reshape(-1, design.shape[2])

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ weights
        maximum = logits.max(axis=1, keepdims=True)
        exponential = np.exp(logits - maximum)
        probabilities = exponential / exponential.sum(axis=1, keepdims=True)
        log_partition = np.log(exponential.sum(axis=1)) + maximum[:, 0]
        loss = np.sum(row_weights * (log_partition - logits[rows, labels]))
        difference = weights - initial
        loss += 0.5 * regularization * np.dot(difference, difference)
        probabilities[rows, labels] -= 1.0
        probabilities *= row_weights[:, None]
        gradient = flat_design.T @ probabilities.reshape(-1)
        gradient += regularization * difference
        return float(loss), gradient.astype(np.float64)

    result = minimize(
        objective,
        np.asarray(initial, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * design.shape[2],
        options={"maxiter": 150, "ftol": 1e-10, "gtol": 1e-6, "maxls": 30},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError("non-finite factor weights")
    print(
        f"[factor] optimizer lambda={regularization:g} success={result.success} "
        f"iterations={result.nit} loss={result.fun:.3f}"
    )
    return np.maximum(result.x, 0.0)


def weighted_accuracy(
    design: np.ndarray,
    labels: np.ndarray,
    row_weights: np.ndarray,
    weights: np.ndarray,
) -> float:
    prediction = (design @ weights).argmax(axis=1)
    return float(np.sum(row_weights * (prediction == labels)) / np.sum(row_weights))


def stratified_sample(
    design: np.ndarray,
    labels: np.ndarray,
    maximum_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(labels) <= maximum_rows:
        return design, labels, np.ones(len(labels), dtype=np.float64)
    baseline = design[:, :, 0].argmax(axis=1)
    exceptional = np.flatnonzero(baseline != labels)
    ordinary = np.flatnonzero(baseline == labels)
    rng = np.random.default_rng(seed)
    exceptional_limit = min(len(exceptional), maximum_rows // 2)
    if len(exceptional) > exceptional_limit:
        exceptional_selected = rng.choice(exceptional, exceptional_limit, replace=False)
    else:
        exceptional_selected = exceptional
    ordinary_limit = maximum_rows - len(exceptional_selected)
    ordinary_selected = rng.choice(ordinary, ordinary_limit, replace=False)
    selected = np.concatenate([exceptional_selected, ordinary_selected])
    rng.shuffle(selected)
    row_weights = np.empty(len(selected), dtype=np.float64)
    exceptional_mask = baseline[selected] != labels[selected]
    row_weights[exceptional_mask] = len(exceptional) / max(1, exceptional_mask.sum())
    row_weights[~exceptional_mask] = len(ordinary) / max(1, (~exceptional_mask).sum())
    return design[selected], labels[selected], row_weights


def select_regularization(
    origins: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
) -> tuple[float, bool, dict]:
    candidates = (1.0, 10.0, 50.0)
    records = {}
    baseline_weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    baseline_weights[0] = 6.0
    baseline_scores = [weighted_accuracy(x, y, sw, baseline_weights) for x, y, sw, _ in origins]
    if len(origins) == 1:
        selected = 10.0
        fitted = fit_nonnegative_weights(origins[0][0], origins[0][1], origins[0][2], selected)
        score = weighted_accuracy(origins[0][0], origins[0][1], origins[0][2], fitted)
        records[str(selected)] = [score]
        enabled = score >= baseline_scores[0]
        print(
            f"[factor] origin={origins[0][3]} baseline={baseline_scores[0]:.6f} "
            f"fitted={score:.6f}"
        )
        return selected, enabled, {"baseline": baseline_scores, "candidates": records}
    best_key = None
    selected = candidates[0]
    for regularization in candidates:
        scores = []
        for held in range(len(origins)):
            train_parts = [origin for index, origin in enumerate(origins) if index != held]
            train_x = np.concatenate([part[0] for part in train_parts])
            train_y = np.concatenate([part[1] for part in train_parts])
            train_sw = np.concatenate([part[2] for part in train_parts])
            weights = fit_nonnegative_weights(
                train_x,
                train_y,
                train_sw,
                regularization,
            )
            x, y, sw, name = origins[held]
            score = weighted_accuracy(x, y, sw, weights)
            scores.append(score)
            print(f"[factor] cv lambda={regularization:g} origin={name} accuracy={score:.6f}")
        records[str(regularization)] = scores
        key = (float(np.mean(scores)), float(np.min(scores)))
        if best_key is None or key > best_key:
            best_key = key
            selected = regularization
    selected_scores = records[str(selected)]
    enabled = (
        float(np.mean(selected_scores)) >= float(np.mean(baseline_scores))
        and float(np.min(selected_scores)) >= float(np.min(baseline_scores)) - 1e-4
    )
    print(
        f"[factor] selected_lambda={selected:g} corrections_enabled={enabled} "
        f"cv_mean={np.mean(selected_scores):.6f} cv_worst={np.min(selected_scores):.6f} "
        f"baseline_mean={np.mean(baseline_scores):.6f} baseline_worst={np.min(baseline_scores):.6f}"
    )
    return selected, enabled, {"baseline": baseline_scores, "candidates": records}


def select_gate_c(
    datasets: list[tuple[GateFeatures, np.ndarray, GateFeatures, np.ndarray, str]],
) -> tuple[float, dict]:
    candidates = (0.03, 0.1, 0.3)
    records = {}
    selected = candidates[0]
    best_key = None
    for c_value in candidates:
        scores = []
        for train_features, train_labels, test_features, test_labels, name in datasets:
            gate = fit_gate(train_features, train_labels, c_value)
            prediction = gate.balanced_probability(test_features.matrix) >= 0.5
            score = float(balanced_accuracy_score(test_labels, prediction))
            scores.append(score)
            print(f"[factor] gate C={c_value:g} origin={name} balanced_accuracy={score:.6f}")
        records[str(c_value)] = scores
        key = (float(np.mean(scores)), float(np.min(scores)))
        if best_key is None or key > best_key:
            best_key = key
            selected = c_value
    print(
        f"[factor] selected_gate_C={selected:g} mean={np.mean(records[str(selected)]):.6f} "
        f"worst={np.min(records[str(selected)]):.6f}"
    )
    return selected, records


def probabilities(design: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = design @ weights
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return result.astype(np.float32)


def document_exception_classes(
    history: HistoryModel,
    target_ids: np.ndarray,
    minimum_support: int,
    minimum_purity: float,
    mode: str,
) -> np.ndarray:
    organization, _ = history.organization_posterior(target_ids)
    organization_class = organization.argmax(axis=1)
    if mode.startswith("item_org_doctype_product"):
        factor_names = ("org_doctype_product",)
    elif mode.startswith("item_org_product"):
        factor_names = ("org_product",)
    else:
        factor_names = ("org_product", "org_ship_product", "org_sold_product")
    candidates = []
    for name in factor_names:
        counts = history.counts(name)
        codes = history.store.key_code(name)[target_ids]
        values = (
            counts.weighted[codes]
            if "recency" in mode
            else history.raw_class_counts(name)[codes]
        )
        top = values.argmax(axis=1)
        support = counts.raw_support[codes]
        purity = values.max(axis=1) / np.maximum(values.sum(axis=1), 1e-12)
        supported = support >= minimum_support
        candidates.append(
            np.where(
                (purity >= minimum_purity) & supported & (top != organization_class),
                top,
                -1,
            )
        )
    candidate_matrix = np.column_stack(candidates)
    active = candidate_matrix >= 0
    vote_counts = np.zeros((len(target_ids), CLASSES), dtype=np.uint8)
    rows, columns = np.nonzero(active)
    if len(rows):
        np.add.at(vote_counts, (rows, candidate_matrix[rows, columns]), 1)
    row_active = vote_counts.max(axis=1) > 0
    row_vote = vote_counts.argmax(axis=1)
    if mode != "document_ensemble":
        result = organization_class.copy()
        result[row_active] = row_vote[row_active]
        return result.astype(np.int16)
    documents = history.store.arrays["doc"][target_ids]
    _, inverse = np.unique(documents, return_inverse=True)
    document_votes = np.zeros((int(inverse.max()) + 1, CLASSES), dtype=np.int32)
    active_rows = np.flatnonzero(row_active)
    if len(active_rows):
        np.add.at(document_votes, (inverse[active_rows], row_vote[active_rows]), 1)
    document_active = document_votes.max(axis=1) > 0
    document_class = document_votes.argmax(axis=1)
    result = organization_class.copy()
    eligible = document_active[inverse]
    result[eligible] = document_class[inverse[eligible]]
    return result.astype(np.int16)


def apply_document_exception_gate(
    prediction: np.ndarray,
    history: HistoryModel,
    target_ids: np.ndarray,
    minimum_support: int,
    minimum_purity: float,
    mode: str,
) -> np.ndarray:
    exception_class = document_exception_classes(
        history,
        target_ids,
        minimum_support,
        minimum_purity,
        mode,
    )
    current = prediction.argmax(axis=1)
    changed = np.flatnonzero(exception_class != current)
    if len(changed) == 0:
        return prediction
    logits = np.log(np.maximum(prediction, FLOOR)).astype(np.float64)
    maximum = logits[changed].max(axis=1)
    logits[changed, exception_class[changed]] = maximum + 0.1
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    print(
        f"[factor] exception_gate mode={mode} support={minimum_support} "
        f"purity={minimum_purity:g} "
        f"changed_rows={len(changed)}"
    )
    return result.astype(np.float32)
