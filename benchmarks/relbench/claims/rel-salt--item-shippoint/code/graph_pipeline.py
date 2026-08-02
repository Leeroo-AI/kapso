import gc
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import fcntl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear as GeometricLinear

from kapso_datasets.common import load_task, run_data_dir, save_predictions, shared_cache_dir


N_CLASSES = 81
HIDDEN_DIM = 192
FANOUT = (24, 12)
RELATION_NAMES = (
    "product",
    "category",
    "sold",
    "ship",
    "bill",
    "payer",
    "address",
    "organization",
    "channel",
    "division",
    "company",
    "currency",
    "document_type",
    "country",
    "region",
)


@dataclass
class GraphTopology:
    codes: dict
    cardinalities: dict
    times: np.ndarray
    ids: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_labels: np.ndarray
    val_labels: np.ndarray


def factorize(values):
    codes, uniques = pd.factorize(values, sort=False)
    if np.any(codes < 0):
        codes = codes + 1
        cardinality = len(uniques) + 1
    else:
        cardinality = len(uniques)
    return codes.astype(np.int32, copy=False), int(cardinality)


def load_topology():
    started = time.time()
    context = load_task(upto_test_timestamp=False)
    database = context.db
    item_columns = [
        "SALESDOCUMENT",
        "SALESDOCUMENTITEM",
        "SALESDOCUMENTITEMCATEGORY",
        "PRODUCT",
        "SOLDTOPARTY",
        "SHIPTOPARTY",
        "BILLTOPARTY",
        "PAYERPARTY",
        "CREATIONTIMESTAMP",
        "ID",
    ]
    document_columns = [
        "SALESDOCUMENT",
        "SALESDOCUMENTTYPE",
        "SALESORGANIZATION",
        "DISTRIBUTIONCHANNEL",
        "ORGANIZATIONDIVISION",
        "BILLINGCOMPANYCODE",
        "TRANSACTIONCURRENCY",
        "CREATIONTIMESTAMP",
    ]
    customer_columns = ["CUSTOMER", "ADDRESSID"]
    address_columns = ["ADDRESSID", "COUNTRY", "REGION"]
    items = database.table_dict["salesdocumentitem"].df[item_columns]
    documents = database.table_dict["salesdocument"].df[document_columns]
    customers = database.table_dict["customer"].df[customer_columns]
    addresses = database.table_dict["address"].df[address_columns]
    document_rows = pd.Index(documents["SALESDOCUMENT"]).get_indexer(items["SALESDOCUMENT"])
    if np.any(document_rows < 0):
        raise RuntimeError("document topology contains unmatched item edges")
    customer_index = pd.Index(customers["CUSTOMER"])
    customer_rows = {}
    for name, column in (
        ("sold", "SOLDTOPARTY"),
        ("ship", "SHIPTOPARTY"),
        ("bill", "BILLTOPARTY"),
        ("payer", "PAYERPARTY"),
    ):
        customer_rows[name] = customer_index.get_indexer(items[column]).astype(np.int32)
        if np.any(customer_rows[name] < 0):
            raise RuntimeError(f"customer topology contains unmatched {name} edges")
    referenced_address_rows = pd.Index(addresses["ADDRESSID"]).get_indexer(customers["ADDRESSID"])
    if np.any(referenced_address_rows < 0):
        raise RuntimeError("address topology contains unmatched customer edges")
    referenced_addresses = addresses.iloc[referenced_address_rows]
    country_by_customer, country_cardinality = factorize(referenced_addresses["COUNTRY"])
    region_by_customer, region_cardinality = factorize(referenced_addresses["REGION"])
    product, product_cardinality = factorize(items["PRODUCT"])
    category, category_cardinality = factorize(items["SALESDOCUMENTITEMCATEGORY"])
    item_number, item_number_cardinality = factorize(items["SALESDOCUMENTITEM"])
    document_type_values, document_type_cardinality = factorize(documents["SALESDOCUMENTTYPE"])
    organization_values, organization_cardinality = factorize(documents["SALESORGANIZATION"])
    channel_values, channel_cardinality = factorize(documents["DISTRIBUTIONCHANNEL"])
    division_values, division_cardinality = factorize(documents["ORGANIZATIONDIVISION"])
    company_values, company_cardinality = factorize(documents["BILLINGCOMPANYCODE"])
    currency_values, currency_cardinality = factorize(documents["TRANSACTIONCURRENCY"])
    codes = {
        "item": np.arange(len(items), dtype=np.int32),
        "document": document_rows.astype(np.int32),
        "product": product,
        "category": category,
        "item_number": item_number,
        "sold": customer_rows["sold"],
        "ship": customer_rows["ship"],
        "bill": customer_rows["bill"],
        "payer": customer_rows["payer"],
        "address": customer_rows["ship"],
        "country": country_by_customer[customer_rows["ship"]],
        "region": region_by_customer[customer_rows["ship"]],
        "organization": organization_values[document_rows],
        "channel": channel_values[document_rows],
        "division": division_values[document_rows],
        "company": company_values[document_rows],
        "currency": currency_values[document_rows],
        "document_type": document_type_values[document_rows],
    }
    cardinalities = {
        "item": len(items),
        "document": len(documents),
        "product": product_cardinality,
        "category": category_cardinality,
        "item_number": item_number_cardinality,
        "sold": len(customers),
        "ship": len(customers),
        "bill": len(customers),
        "payer": len(customers),
        "address": len(customers),
        "country": country_cardinality,
        "region": region_cardinality,
        "organization": organization_cardinality,
        "channel": channel_cardinality,
        "division": division_cardinality,
        "company": company_cardinality,
        "currency": currency_cardinality,
        "document_type": document_type_cardinality,
    }
    item_index = pd.Index(items["ID"])
    train_indices = item_index.get_indexer(context.train.df["ID"]).astype(np.int32)
    val_indices = item_index.get_indexer(context.val.df["ID"]).astype(np.int32)
    test_indices = item_index.get_indexer(context.test.df["ID"]).astype(np.int32)
    if np.any(train_indices < 0) or np.any(val_indices < 0) or np.any(test_indices < 0):
        raise RuntimeError("task rows do not map one-to-one to item nodes")
    topology = GraphTopology(
        codes=codes,
        cardinalities=cardinalities,
        times=items["CREATIONTIMESTAMP"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
        ids=items["ID"].to_numpy(),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        train_labels=context.train.df[context.target_col].to_numpy(dtype=np.int64),
        val_labels=context.val.df[context.target_col].to_numpy(dtype=np.int64),
    )
    print(
        f"[phase] topology items={len(items)} documents={len(documents)} "
        f"customers={len(customers)} addresses={len(referenced_addresses)} elapsed={time.time() - started:.2f}s"
    )
    return topology


class CountState:
    def __init__(self, keys, counts):
        self.keys = keys
        self.counts = counts

    @classmethod
    def build(cls, all_keys, history_indices, labels):
        history_keys = all_keys[history_indices]
        keys, inverse = np.unique(history_keys, return_inverse=True)
        flat = inverse.astype(np.int64) * N_CLASSES + labels
        counts = np.bincount(flat, minlength=len(keys) * N_CLASSES).reshape(-1, N_CLASSES)
        return cls(keys, counts.astype(np.int32, copy=False))

    def lookup(self, query_keys):
        positions = np.searchsorted(self.keys, query_keys)
        valid = positions < len(self.keys)
        valid[valid] &= self.keys[positions[valid]] == query_keys[valid]
        output = np.zeros((len(query_keys), N_CLASSES), dtype=np.float32)
        output[valid] = self.counts[positions[valid]]
        return output


class ModeState:
    def __init__(self, keys, modes, supports, purities):
        self.keys = keys
        self.modes = modes
        self.supports = supports
        self.purities = purities

    @classmethod
    def build(cls, all_keys, history_indices, labels):
        pairs = all_keys[history_indices].astype(np.int64) * N_CLASSES + labels
        unique_pairs, pair_counts = np.unique(pairs, return_counts=True)
        keys_by_pair = unique_pairs // N_CLASSES
        classes_by_pair = (unique_pairs % N_CLASSES).astype(np.int16)
        starts = np.r_[0, np.flatnonzero(np.diff(keys_by_pair)) + 1]
        keys = keys_by_pair[starts]
        supports = np.add.reduceat(pair_counts, starts)
        maxima = np.maximum.reduceat(pair_counts, starts)
        group_ids = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(pair_counts)]))
        candidates = np.flatnonzero(pair_counts == maxima[group_ids])
        candidate_groups = group_ids[candidates]
        first = np.r_[True, np.diff(candidate_groups) != 0]
        best_positions = candidates[first]
        modes = classes_by_pair[best_positions]
        purities = maxima.astype(np.float32) / supports
        return cls(keys, modes, supports.astype(np.int32), purities)

    def lookup(self, query_keys):
        positions = np.searchsorted(self.keys, query_keys)
        valid = positions < len(self.keys)
        valid[valid] &= self.keys[positions[valid]] == query_keys[valid]
        modes = np.full(len(query_keys), -1, dtype=np.int16)
        supports = np.zeros(len(query_keys), dtype=np.int32)
        purities = np.zeros(len(query_keys), dtype=np.float32)
        modes[valid] = self.modes[positions[valid]]
        supports[valid] = self.supports[positions[valid]]
        purities[valid] = self.purities[positions[valid]]
        return modes, supports, purities


class HierarchicalPosterior:
    def __init__(self, topology, history_indices, labels, cutoff=None):
        self.topology = topology
        self.prior_counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
        self.prior = self.prior_counts / self.prior_counts.sum()
        organization = topology.codes["organization"].astype(np.int64)
        document_type = topology.codes["document_type"].astype(np.int64)
        category = topology.codes["category"].astype(np.int64)
        product = topology.codes["product"].astype(np.int64)
        type_cardinality = topology.cardinalities["document_type"]
        category_cardinality = topology.cardinalities["category"]
        product_cardinality = topology.cardinalities["product"]
        self.organization_type_keys = organization * type_cardinality + document_type
        self.base_keys = self.organization_type_keys * category_cardinality + category
        self.product_keys = self.organization_type_keys * product_cardinality + product
        cutoff_ns = int(topology.times[history_indices].max() + 1) if cutoff is None else int(cutoff)
        recent = topology.times[history_indices] >= cutoff_ns - 60 * 86_400_000_000_000
        self.base_state = CountState.build(self.base_keys, history_indices[recent], labels[recent])
        self.fallback_state = CountState.build(self.organization_type_keys, history_indices, labels)
        self.product_state = ModeState.build(self.product_keys, history_indices, labels)

    def predict(self, query_indices, pool_documents=True):
        base_counts = self.base_state.lookup(self.base_keys[query_indices])
        fallback_counts = self.fallback_state.lookup(self.organization_type_keys[query_indices])
        support = base_counts.sum(axis=1)
        missing = support == 0
        base_counts[missing] = fallback_counts[missing]
        support = base_counts.sum(axis=1)
        missing = support == 0
        base_counts[missing] = self.prior_counts
        support = base_counts.sum(axis=1)
        probabilities = (base_counts + 0.25 * self.prior) / (support[:, None] + 0.25)
        confidence = probabilities.max(axis=1)
        modes, product_support, product_purity = self.product_state.lookup(self.product_keys[query_indices])
        replace = (confidence < 0.995) & (product_support > 0)
        if np.any(replace):
            probabilities[replace] = 0.005 * self.prior
            probabilities[np.flatnonzero(replace), modes[replace]] += 0.995
            support[replace] = product_support[replace]
            confidence[replace] = product_purity[replace]
        probabilities, support = pool_by_document(
            probabilities,
            support,
            self.topology.codes["document"][query_indices],
            pool_documents,
        )
        confidence = probabilities.max(axis=1)
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, None)), axis=1) / math.log(N_CLASSES)
        stats = np.column_stack((np.log1p(support), confidence, entropy)).astype(np.float32)
        return probabilities.astype(np.float32), stats


def pool_by_document(scores, support, document_codes, enabled=True):
    if not enabled or len(scores) == 0:
        return scores, support
    _, inverse = np.unique(document_codes, return_inverse=True)
    document_scores = np.zeros((int(inverse.max()) + 1, scores.shape[1]), dtype=np.float64)
    document_sizes = np.bincount(inverse)
    np.add.at(document_scores, inverse, scores)
    document_scores /= document_sizes[:, None]
    document_support = np.zeros(len(document_sizes), dtype=np.float32)
    np.maximum.at(document_support, inverse, support)
    return document_scores[inverse].astype(np.float32), document_support[inverse]


class EntityRelationState:
    def __init__(self, keys, counts, weighted_counts, last_times):
        self.keys = keys
        self.counts = counts
        self.weighted_counts = weighted_counts
        self.last_times = last_times

    @classmethod
    def build(cls, codes, history_indices, labels, times, cutoff):
        history_codes = codes[history_indices]
        _, degree_inverse, degrees = np.unique(history_codes, return_inverse=True, return_counts=True)
        strides = np.maximum(1, np.ceil(degrees / FANOUT[0]).astype(np.int32))
        hashes = (history_indices.astype(np.uint64) * np.uint64(11400714819323198485)) >> np.uint64(32)
        sampled = hashes % strides[degree_inverse].astype(np.uint64) == 0
        history_indices = history_indices[sampled]
        labels = labels[sampled]
        history_codes = history_codes[sampled]
        keys, inverse = np.unique(history_codes, return_inverse=True)
        flat = inverse.astype(np.int64) * N_CLASSES + labels
        counts = np.bincount(flat, minlength=len(keys) * N_CLASSES).reshape(-1, N_CLASSES)
        ages = np.maximum(0.0, (cutoff - times[history_indices]) / 86_400_000_000_000.0)
        weights = np.exp(-ages / 180.0)
        weighted = np.bincount(flat, weights=weights, minlength=len(keys) * N_CLASSES).reshape(-1, N_CLASSES)
        last_times = np.full(len(keys), np.iinfo(np.int64).min, dtype=np.int64)
        np.maximum.at(last_times, inverse, times[history_indices])
        return cls(
            keys.astype(np.int32, copy=False),
            counts.astype(np.int32, copy=False),
            weighted.astype(np.float32, copy=False),
            last_times,
        )

    def features(self, query_codes, prior, cutoff):
        positions = np.searchsorted(self.keys, query_codes)
        valid = positions < len(self.keys)
        valid[valid] &= self.keys[positions[valid]] == query_codes[valid]
        output = np.zeros((len(query_codes), N_CLASSES + 3), dtype=np.float32)
        output[:, :N_CLASSES] = prior
        if not np.any(valid):
            return output
        rows = positions[valid]
        counts = self.counts[rows].astype(np.float32)
        weighted = self.weighted_counts[rows]
        support = counts.sum(axis=1)
        weighted_support = weighted.sum(axis=1)
        mean_probabilities = (counts + 2.0 * prior) / (support[:, None] + 2.0)
        age_probabilities = (weighted + 2.0 * prior) / (weighted_support[:, None] + 2.0)
        probabilities = 0.5 * mean_probabilities + 0.5 * age_probabilities
        entropy = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, None)), axis=1) / math.log(N_CLASSES)
        age_days = np.maximum(0.0, (cutoff - self.last_times[rows]) / 86_400_000_000_000.0)
        output[valid, :N_CLASSES] = probabilities
        output[valid, N_CLASSES] = np.log1p(support)
        output[valid, N_CLASSES + 1] = entropy
        output[valid, N_CLASSES + 2] = np.log1p(age_days) / math.log(3661.0)
        return output


class FrozenGraphState:
    def __init__(self, topology, history_indices, labels, cutoff):
        self.topology = topology
        self.cutoff = int(cutoff)
        counts = np.bincount(labels, minlength=N_CLASSES).astype(np.float32)
        self.prior = counts / counts.sum()
        self.relations = {}
        for name in RELATION_NAMES:
            self.relations[name] = EntityRelationState.build(
                topology.codes[name], history_indices, labels, topology.times, self.cutoff
            )

    def batch_features(self, item_indices):
        features = [
            self.relations[name].features(
                self.topology.codes[name][item_indices], self.prior, self.cutoff
            )
            for name in RELATION_NAMES
        ]
        return np.stack(features, axis=1)


class RelationGraphResidual(nn.Module):
    def __init__(self, cardinalities, layers):
        super().__init__()
        self.layers = layers
        self.product_embedding = nn.Embedding(cardinalities["product"], 64)
        self.customer_embedding = nn.Embedding(cardinalities["ship"], 64)
        self.id_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(cardinalities[name], 32)
                for name in (
                    "category",
                    "address",
                    "organization",
                    "channel",
                    "division",
                    "company",
                    "currency",
                    "document_type",
                    "country",
                    "region",
                )
            }
        )
        self.relation_layer_one = nn.ModuleList(
            [GeometricLinear(N_CLASSES + 3, HIDDEN_DIM) for _ in RELATION_NAMES]
        )
        self.relation_layer_two = nn.ModuleList(
            [GeometricLinear(HIDDEN_DIM * 2, HIDDEN_DIM) for _ in RELATION_NAMES]
        )
        self.plant_indices = [RELATION_NAMES.index(name) for name in ("product", "sold", "ship", "bill", "payer", "organization", "company", "region")]
        self.shipping_indices = [RELATION_NAMES.index(name) for name in ("ship", "address", "country", "region", "document_type")]
        self.loading_indices = [RELATION_NAMES.index(name) for name in ("product", "category")]
        self.plant_tower = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Dropout(0.1))
        self.shipping_tower = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Dropout(0.1))
        self.loading_tower = nn.Sequential(nn.Linear(HIDDEN_DIM, 128), nn.ReLU(), nn.Dropout(0.1))
        item_width = 64 * 3 + 32 * 7
        self.item_phi = nn.Sequential(nn.Linear(item_width, 128), nn.ReLU(), nn.Dropout(0.1))
        self.document_rho = nn.Sequential(nn.Linear(128, 192), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(
            nn.Linear(128 * 3 + 192 + N_CLASSES + 3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, N_CLASSES),
        )

    def raw_item_features(self, codes):
        values = [
            self.product_embedding(codes["product"]),
            self.customer_embedding(codes["ship"]),
            self.customer_embedding(codes["sold"]),
        ]
        for name in ("category", "organization", "company", "document_type", "country", "region", "channel"):
            values.append(self.id_embeddings[name](codes[name]))
        return torch.cat(values, dim=1)

    def forward(self, relation_features, codes, base, stats, document_inverse):
        relation_hidden = []
        for index, layer in enumerate(self.relation_layer_one):
            relation_hidden.append(F.relu(layer(relation_features[:, index])))
        hidden = torch.stack(relation_hidden, dim=1)
        if self.layers > 1:
            relation_support = relation_features[:, :, N_CLASSES]
            sampled_relations = torch.topk(relation_support, k=FANOUT[1], dim=1).indices
            relation_mask = torch.zeros_like(relation_support)
            relation_mask.scatter_(1, sampled_relations, 1.0)
            sampled_mean = (hidden * relation_mask[:, :, None]).sum(dim=1) / relation_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            neighbor_mean = sampled_mean[:, None, :].expand_as(hidden)
            second = []
            for index, layer in enumerate(self.relation_layer_two):
                second.append(F.relu(layer(torch.cat((hidden[:, index], neighbor_mean[:, index]), dim=1))))
            hidden = torch.stack(second, dim=1)
        plant = self.plant_tower(hidden[:, self.plant_indices].mean(dim=1))
        shipping = self.shipping_tower(hidden[:, self.shipping_indices].mean(dim=1))
        loading = self.loading_tower(hidden[:, self.loading_indices].mean(dim=1))
        item = self.item_phi(self.raw_item_features(codes))
        document_count = int(document_inverse.max().item()) + 1
        document_sum = torch.zeros((document_count, item.shape[1]), device=item.device, dtype=item.dtype)
        document_sum.index_add_(0, document_inverse, item)
        document_size = torch.bincount(document_inverse, minlength=document_count).to(item.dtype).unsqueeze(1)
        document = self.document_rho(document_sum / document_size.clamp_min(1.0))[document_inverse]
        return self.head(torch.cat((plant, shipping, loading, document, base, stats), dim=1))


def complete_document_subset(topology, query_indices, limit):
    if limit is None or len(query_indices) <= limit:
        return query_indices
    document_codes = topology.codes["document"][query_indices]
    selected_documents = np.unique(document_codes[:limit])
    return query_indices[np.isin(document_codes, selected_documents)]


def document_batches(topology, query_indices, batch_size=4096):
    document_codes = topology.codes["document"][query_indices]
    order = np.argsort(document_codes, kind="stable")
    sorted_documents = document_codes[order]
    start = 0
    while start < len(order):
        end = min(len(order), start + batch_size)
        while end < len(order) and sorted_documents[end] == sorted_documents[end - 1]:
            end += 1
        positions = order[start:end]
        local_documents = document_codes[positions]
        inverse = np.r_[0, np.cumsum(local_documents[1:] != local_documents[:-1])].astype(np.int64)
        yield positions, inverse
        start = end


def tensor_codes(topology, item_indices, device):
    names = ("product", "ship", "sold", "category", "organization", "company", "document_type", "country", "region", "channel")
    return {
        name: torch.as_tensor(topology.codes[name][item_indices].astype(np.int64), device=device)
        for name in names
    }


def autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def residual_gate(stats):
    confidence = stats[:, 1]
    support = stats[:, 0]
    protected = (confidence > 0.98) & (support > math.log1p(20.0))
    return torch.where(protected, torch.full_like(confidence, 0.02), torch.ones_like(confidence))


def train_snapshot(model, optimizer, topology, query_indices, query_labels, graph_state, posterior, device, label):
    started = time.time()
    model.train()
    base, stats = posterior.predict(query_indices)
    total_loss = 0.0
    total_rows = 0
    for positions, document_inverse in document_batches(topology, query_indices):
        item_indices = query_indices[positions]
        relation_features = torch.as_tensor(graph_state.batch_features(item_indices), device=device)
        base_tensor = torch.as_tensor(base[positions], device=device)
        stats_tensor = torch.as_tensor(stats[positions], device=device)
        labels_tensor = torch.as_tensor(query_labels[positions], device=device)
        inverse_tensor = torch.as_tensor(document_inverse, device=device)
        codes = tensor_codes(topology, item_indices, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            residual = model(relation_features, codes, base_tensor, stats_tensor, inverse_tensor)
            gate = residual_gate(stats_tensor)
            logits = torch.log(base_tensor.clamp_min(1e-6)) + 0.75 * gate[:, None] * residual
            posterior_error = (base_tensor.argmax(dim=1) != labels_tensor).to(stats_tensor.dtype)
            weights = 1.0 + 20.0 * posterior_error + 4.0 * (1.0 - stats_tensor[:, 1])
            predictive_loss = (F.cross_entropy(logits, labels_tensor, reduction="none") * weights).mean()
            protect = ((stats_tensor[:, 1] > 0.98) & (stats_tensor[:, 0] > math.log1p(20.0))).to(residual.dtype)
            penalty = 0.1 * (residual.square().mean(dim=1) * protect).mean()
            loss = predictive_loss + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(positions)
        total_rows += len(positions)
    print(f"[train] snapshot={label} rows={total_rows} loss={total_loss / max(1, total_rows):.6f} elapsed={time.time() - started:.2f}s")


def predict_residual(model, topology, query_indices, graph_state, base, stats, device):
    model.eval()
    residual = np.zeros((len(query_indices), N_CLASSES), dtype=np.float32)
    with torch.inference_mode():
        for positions, document_inverse in document_batches(topology, query_indices):
            item_indices = query_indices[positions]
            relation_features = torch.as_tensor(graph_state.batch_features(item_indices), device=device)
            base_tensor = torch.as_tensor(base[positions], device=device)
            stats_tensor = torch.as_tensor(stats[positions], device=device)
            inverse_tensor = torch.as_tensor(document_inverse, device=device)
            codes = tensor_codes(topology, item_indices, device)
            with autocast_context(device):
                values = model(relation_features, codes, base_tensor, stats_tensor, inverse_tensor)
            residual[positions] = values.float().cpu().numpy()
    return residual


def logits_for_blend(base, stats, residual, blend, document_codes):
    protected = (stats[:, 1] > 0.98) & (stats[:, 0] > math.log1p(20.0))
    gate = np.where(protected, 0.02, 1.0).astype(np.float32)
    logits = np.log(np.clip(base, 1e-6, None)) + blend * gate[:, None] * residual
    pooled, _ = pool_by_document(logits, np.ones(len(logits), dtype=np.float32), document_codes, True)
    return pooled.astype(np.float32)


def snapshot_indices(topology, start, end, include_validation=False):
    source_indices = topology.train_indices
    source_labels = topology.train_labels
    if include_validation:
        source_indices = np.concatenate((topology.train_indices, topology.val_indices))
        source_labels = np.concatenate((topology.train_labels, topology.val_labels))
    times = topology.times[source_indices]
    history_mask = times < np.datetime64(start, "ns").astype(np.int64)
    query_mask = (times >= np.datetime64(start, "ns").astype(np.int64)) & (times < np.datetime64(end, "ns").astype(np.int64))
    return source_indices[history_mask], source_labels[history_mask], source_indices[query_mask], source_labels[query_mask]


def build_snapshot(topology, history_indices, history_labels, cutoff):
    started = time.time()
    cutoff_ns = np.datetime64(cutoff, "ns").astype(np.int64)
    if np.any(topology.times[history_indices] >= cutoff_ns):
        raise RuntimeError("snapshot contains labels at or after its cutoff")
    posterior = HierarchicalPosterior(topology, history_indices, history_labels, cutoff_ns)
    graph_state = FrozenGraphState(topology, history_indices, history_labels, cutoff_ns)
    rate = len(history_indices) / max(0.001, time.time() - started)
    print(f"[phase] snapshot cutoff={cutoff} history={len(history_indices)} rate={rate:.0f} rows/s elapsed={time.time() - started:.2f}s")
    return posterior, graph_state


def cache_paths():
    cache = shared_cache_dir() / "lane3_heterograph_residual_v4"
    cache.mkdir(parents=True, exist_ok=True)
    return {
        "val": cache / "val_baseline.npy",
        "val_stats": cache / "val_stats.npy",
        "test": cache / "test_baseline.npy",
        "test_stats": cache / "test_stats.npy",
    }


def register_cache(paths):
    cache = shared_cache_dir()
    registry = cache / "artifacts.json"
    lock_path = cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            entries = json.loads(registry.read_text()) if registry.exists() else []
        except json.JSONDecodeError:
            entries = []
        content_key = "rel-salt-item-shippoint-lane3-heterograph-residual-v4"
        if not any(entry.get("content_key") == content_key for entry in entries):
            entries.append(
                {
                    "name": "lane3 cutoff-safe hierarchical posterior",
                    "path": str(paths["val"].parent.relative_to(cache)),
                    "description": "Model A validation and Model B test posterior scores and frozen statistics",
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py; permitted-column topology and historical labels rebuild the arrays",
                }
            )
            temporary = registry.with_suffix(".tmp.lane3")
            temporary.write_text(json.dumps(entries, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_or_load_baselines(topology):
    paths = cache_paths()
    expected = {
        "val": (len(topology.val_indices), N_CLASSES),
        "val_stats": (len(topology.val_indices), 3),
        "test": (len(topology.test_indices), N_CLASSES),
        "test_stats": (len(topology.test_indices), 3),
    }
    valid = all(path.exists() for path in paths.values())
    arrays = {}
    if valid:
        for name, path in paths.items():
            arrays[name] = np.load(path, allow_pickle=False)
            valid = valid and arrays[name].shape == expected[name] and np.all(np.isfinite(arrays[name]))
    if valid:
        print("[phase] loaded deterministic posterior cache")
        return arrays
    started = time.time()
    posterior_a = HierarchicalPosterior(topology, topology.train_indices, topology.train_labels)
    val, val_stats = posterior_a.predict(topology.val_indices)
    combined_indices = np.concatenate((topology.train_indices, topology.val_indices))
    combined_labels = np.concatenate((topology.train_labels, topology.val_labels))
    posterior_b = HierarchicalPosterior(topology, combined_indices, combined_labels)
    test, test_stats = posterior_b.predict(topology.test_indices)
    arrays = {"val": val, "val_stats": val_stats, "test": test, "test_stats": test_stats}
    for name, path in paths.items():
        np.save(path, arrays[name])
    register_cache(paths)
    print(f"[phase] cached deterministic posterior baseline elapsed={time.time() - started:.2f}s")
    return arrays


def internal_selection(model, topology, device, debug):
    if debug:
        history_indices, history_labels, query_indices, query_labels = snapshot_indices(topology, "2020-01-01", "2020-02-01")
        history_indices = history_indices[-100_000:]
        history_labels = history_labels[-100_000:]
        query_indices = complete_document_subset(topology, query_indices, 20_000)
        query_label_map = np.full(len(topology.times), -1, dtype=np.int16)
        query_label_map[topology.train_indices] = topology.train_labels
        query_labels = query_label_map[query_indices].astype(np.int64)
        posterior, state = build_snapshot(topology, history_indices, history_labels, "2020-01-01")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
        train_snapshot(model, optimizer, topology, query_indices, query_labels, state, posterior, device, "debug-jan")
        return 0.25
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)
    for start, end in (("2019-11-01", "2019-12-01"), ("2019-12-01", "2020-01-01")):
        history_indices, history_labels, query_indices, query_labels = snapshot_indices(topology, start, end)
        posterior, state = build_snapshot(topology, history_indices, history_labels, start)
        train_snapshot(model, optimizer, topology, query_indices, query_labels, state, posterior, device, start[:7])
        del posterior, state
        gc.collect()
    history_indices, history_labels, query_indices, query_labels = snapshot_indices(topology, "2020-01-01", "2020-02-01")
    posterior, state = build_snapshot(topology, history_indices, history_labels, "2020-01-01")
    base, stats = posterior.predict(query_indices)
    residual = predict_residual(model, topology, query_indices, state, base, stats, device)
    candidates = [0.0, 0.25, 0.5, 0.75]
    accuracies = {}
    document_codes = topology.codes["document"][query_indices]
    for blend in candidates:
        logits = logits_for_blend(base, stats, residual, blend, document_codes)
        accuracies[blend] = float(np.mean(logits.argmax(axis=1) == query_labels))
    best_accuracy = max(accuracies.values())
    selected = min(value for value in candidates if accuracies[value] >= best_accuracy - 0.0003)
    predictions = logits_for_blend(base, stats, residual, selected, document_codes).argmax(axis=1)
    support = np.expm1(stats[:, 0])
    strata = {}
    for name, mask in (
        ("support_0_4", support < 5),
        ("support_5_19", (support >= 5) & (support < 20)),
        ("support_20_plus", support >= 20),
    ):
        strata[name] = {
            "count": int(mask.sum()),
            "accuracy": float(np.mean(predictions[mask] == query_labels[mask])) if np.any(mask) else None,
        }
    print(f"[select] forward_blends={json.dumps(accuracies, sort_keys=True)} selected={selected}")
    print(f"[select] forward_strata={json.dumps(strata, sort_keys=True)}")
    train_snapshot(model, optimizer, topology, query_indices, query_labels, state, posterior, device, "2020-01")
    del posterior, state, base, stats, residual
    gc.collect()
    return selected


def final_model_a(model, topology, baselines, blend, device, debug):
    cutoff = np.datetime64("2020-02-01", "ns").astype(np.int64)
    history_indices = topology.train_indices
    history_labels = topology.train_labels
    if debug:
        history_indices = history_indices[-100_000:]
        history_labels = history_labels[-100_000:]
    graph_state = FrozenGraphState(topology, history_indices, history_labels, cutoff)
    query_indices = topology.val_indices
    if debug:
        query_indices = complete_document_subset(topology, query_indices, 100_000)
    positions = pd.Index(topology.val_indices).get_indexer(query_indices)
    residual = predict_residual(
        model,
        topology,
        query_indices,
        graph_state,
        baselines["val"][positions],
        baselines["val_stats"][positions],
        device,
    )
    logits = logits_for_blend(
        baselines["val"][positions],
        baselines["val_stats"][positions],
        residual,
        blend,
        topology.codes["document"][query_indices],
    )
    output = np.log(np.clip(baselines["val"], 1e-6, None)).astype(np.float32)
    output[positions] = logits
    del graph_state, residual
    gc.collect()
    print(f"[phase] Model A validation rows graph={len(query_indices)} fallback={len(output) - len(query_indices)}")
    return output


def update_model_b(model, topology, device, debug):
    if debug:
        return
    history_indices, history_labels, query_indices, query_labels = snapshot_indices(
        topology, "2020-06-01", "2020-07-01", include_validation=True
    )
    posterior, state = build_snapshot(topology, history_indices, history_labels, "2020-06-01")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    train_snapshot(model, optimizer, topology, query_indices, query_labels, state, posterior, device, "Model-B-2020-06")
    del posterior, state
    gc.collect()


def final_model_b(model, topology, baselines, blend, device, debug):
    cutoff = np.datetime64("2020-07-01", "ns").astype(np.int64)
    history_indices = np.concatenate((topology.train_indices, topology.val_indices))
    history_labels = np.concatenate((topology.train_labels, topology.val_labels))
    if debug:
        history_indices = history_indices[-100_000:]
        history_labels = history_labels[-100_000:]
    graph_state = FrozenGraphState(topology, history_indices, history_labels, cutoff)
    query_indices = topology.test_indices
    if debug:
        query_indices = complete_document_subset(topology, query_indices, 100_000)
    positions = pd.Index(topology.test_indices).get_indexer(query_indices)
    residual = predict_residual(
        model,
        topology,
        query_indices,
        graph_state,
        baselines["test"][positions],
        baselines["test_stats"][positions],
        device,
    )
    logits = logits_for_blend(
        baselines["test"][positions],
        baselines["test_stats"][positions],
        residual,
        blend,
        topology.codes["document"][query_indices],
    )
    output = np.log(np.clip(baselines["test"], 1e-6, None)).astype(np.float32)
    output[positions] = logits
    del graph_state, residual
    gc.collect()
    print(f"[phase] Model B test rows graph={len(query_indices)} fallback={len(output) - len(query_indices)}")
    return output


def write_diagnostics(debug, blend, topology):
    diagnostics = {
        "debug": debug,
        "selected_forward_blend": blend,
        "layers": 1 if debug else 2,
        "fanout": list(FANOUT),
        "hidden_dimension": HIDDEN_DIM,
        "batch_size": 4096,
        "train_rows": len(topology.train_indices),
        "validation_rows": len(topology.val_indices),
        "test_rows": len(topology.test_indices),
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))


def run(debug=False):
    torch.manual_seed(1337)
    np.random.seed(1337)
    topology = load_topology()
    baselines = build_or_load_baselines(topology)
    save_predictions(baselines["val"], baselines["test"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelationGraphResidual(topology.cardinalities, 1 if debug else 2).to(device)
    print(f"[phase] graph model device={device} layers={model.layers} torch={torch.__version__}")
    blend = internal_selection(model, topology, device, debug)
    validation_output = final_model_a(model, topology, baselines, blend, device, debug)
    save_predictions(validation_output, baselines["test"])
    update_model_b(model, topology, device, debug)
    test_output = final_model_b(model, topology, baselines, blend, device, debug)
    save_predictions(validation_output, test_output)
    write_diagnostics(debug, blend, topology)
    if validation_output.shape != (293780, N_CLASSES) or test_output.shape != (398536, N_CLASSES):
        raise RuntimeError(f"prediction contract mismatch val={validation_output.shape} test={test_output.shape}")
    if not np.all(np.isfinite(validation_output)) or not np.all(np.isfinite(test_output)):
        raise RuntimeError("prediction contract contains non-finite scores")
