from __future__ import annotations

import os
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from relbench.base import Database, Table
from relbench.modeling.graph import AttachTargetTransform, make_pkey_fkey_graph
from relbench.modeling.nn import HeteroEncoder
from sklearn.metrics import roc_auc_score
from torch import nn
from torch_frame import stype
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, SAGEConv

from renewal import OOF_ORIGIN_INDICES, cache_root, connection, ensure_event_cache, read_origins


class ResidualHeteroSAGE(nn.Module):
    def __init__(self, node_types: list[str], edge_types: list[tuple[str, str, str]], channels: int = 128, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.convs = nn.ModuleList(
            [HeteroConv({edge: SAGEConv((channels, channels), channels, aggr="mean") for edge in edge_types}, aggr="sum") for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.ModuleDict({node_type: nn.LayerNorm(channels) for node_type in node_types}) for _ in range(layers)])
        self.dropout = dropout

    def forward(self, x_dict: dict[str, torch.Tensor], edge_index_dict: dict) -> dict[str, torch.Tensor]:
        for conv, norms in zip(self.convs, self.norms):
            updated = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.dropout(F.relu(norms[node_type](value + x_dict[node_type][: value.size(0)])), self.dropout, self.training)
                for node_type, value in updated.items()
            }
        return x_dict


class SmokeModel(nn.Module):
    def __init__(self, data, stats):
        super().__init__()
        self.encoder = HeteroEncoder(
            128,
            {node_type: data[node_type].tf.col_names_dict for node_type in data.node_types},
            stats,
            torch_frame_model_kwargs={"channels": 128, "num_layers": 2},
        )
        self.gnn = ResidualHeteroSAGE(list(data.node_types), list(data.edge_types), 128, 2, 0.2)
        self.head = nn.Sequential(nn.Linear(128 + 24, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, batch, rfm: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch.tf_dict)
        hidden = self.gnn(encoded, batch.edge_index_dict)
        seed = hidden["customer"][: batch["customer"].batch_size]
        return self.head(torch.cat([seed, rfm], dim=1)).squeeze(1)


def graph_smoke_test() -> dict:
    started = time.time()
    events, _ = ensure_event_cache(debug=True)
    con = connection()
    review = con.execute(
        f"""
        SELECT
            review_time, customer_id, product_id, rating, verified,
            text_length, summary_length, price,
            CAST(category_hash AS VARCHAR) AS category_code,
            CAST(brand_hash AS VARCHAR) AS brand_code,
            title_length, description_length, category_missing, description_missing
        FROM read_parquet('{events}')
        ORDER BY review_time LIMIT 4096
        """
    ).fetch_df()
    customer_ids = np.sort(review["customer_id"].unique())
    product_ids = np.sort(review["product_id"].unique())
    customer_map = {value: index for index, value in enumerate(customer_ids)}
    product_map = {value: index for index, value in enumerate(product_ids)}
    review["customer_id"] = review["customer_id"].map(customer_map)
    review["product_id"] = review["product_id"].map(product_map)
    customer = pd.DataFrame({"customer_id": np.arange(len(customer_ids)), "name_length": np.ones(len(customer_ids), dtype=np.float32)})
    product = review.sort_values("review_time").drop_duplicates("product_id", keep="last")[
        ["product_id", "price", "category_code", "brand_code", "title_length", "description_length", "category_missing", "description_missing"]
    ].sort_values("product_id").reset_index(drop=True)
    review_table = review[["review_time", "customer_id", "product_id", "rating", "verified", "text_length", "summary_length"]].copy()
    database = Database(
        {
            "customer": Table(customer, {}, "customer_id", None),
            "product": Table(product, {}, "product_id", None),
            "review": Table(review_table, {"customer_id": "customer", "product_id": "product"}, None, "review_time"),
        }
    )
    column_types = {
        "customer": {"name_length": stype.numerical},
        "product": {
            "price": stype.numerical,
            "category_code": stype.categorical,
            "brand_code": stype.categorical,
            "title_length": stype.numerical,
            "description_length": stype.numerical,
            "category_missing": stype.numerical,
            "description_missing": stype.numerical,
        },
        "review": {
            "review_time": stype.timestamp,
            "rating": stype.numerical,
            "verified": stype.numerical,
            "text_length": stype.numerical,
            "summary_length": stype.numerical,
        },
    }
    data, stats = make_pkey_fkey_graph(database, column_types)
    batch_size = min(16, len(customer))
    seeds = torch.arange(batch_size, dtype=torch.long)
    cutoff = int(review_table["review_time"].astype("int64").max() / 1_000_000_000) - 86400
    seed_time = torch.full((batch_size,), cutoff, dtype=torch.long)
    targets = torch.arange(batch_size).remainder(2).float()
    loader = NeighborLoader(
        data,
        num_neighbors=[8, 4],
        input_nodes=("customer", seeds),
        input_time=seed_time,
        time_attr="time",
        temporal_strategy="last",
        batch_size=batch_size,
        shuffle=False,
        transform=AttachTargetTransform("customer", targets),
    )
    batch = next(iter(loader))
    if "batch" in batch["review"] and len(batch["review"].time):
        permitted = batch["customer"].seed_time[batch["review"].batch]
        violations = int((batch["review"].time > permitted).sum())
    else:
        violations = 0
    if violations:
        raise RuntimeError(f"temporal sampler returned {violations} post-seed reviews")
    model = SmokeModel(data, stats)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    rfm = torch.zeros((batch_size, 24), dtype=torch.float32)
    model.train()
    logits = model(batch, rfm)
    loss = F.binary_cross_entropy_with_logits(logits, batch["customer"].y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(batch, rfm))
    edge_count = int(sum(store.edge_index.size(1) for store in batch.edge_stores))
    return {
        "seconds": time.time() - started,
        "batch_size": batch_size,
        "nodes": int(sum(store.num_nodes for store in batch.node_stores)),
        "edges": edge_count,
        "loss": float(loss.detach()),
        "finite": bool(torch.isfinite(probabilities).all()),
        "post_seed_violations": violations,
        "fanouts": [8, 4],
    }


class EventStore:
    def __init__(self, debug: bool = False):
        path, _ = ensure_event_cache(debug=debug)
        limit = "LIMIT 250000" if debug else ""
        frame = connection().execute(
            f"""
            SELECT
                customer_id,
                CAST(epoch(review_time) / 86400 AS INTEGER) AS event_day,
                rating, verified, text_length, summary_length,
                price, category_hash, brand_hash, title_length,
                description_length, category_missing, description_missing
            FROM read_parquet('{path}')
            ORDER BY customer_id, review_time
            {limit}
            """
        ).fetch_df()
        self.customer = frame.pop("customer_id").to_numpy(np.int32)
        self.event_day = frame.pop("event_day").to_numpy(np.int32)
        self.values = frame.to_numpy(np.float32)
        max_customer = max(1_850_193, int(self.customer.max()) + 1)
        counts = np.bincount(self.customer, minlength=max_customer)
        self.pointer = np.empty(max_customer + 1, dtype=np.int64)
        self.pointer[0] = 0
        np.cumsum(counts, out=self.pointer[1:])

    def batch(self, customers: np.ndarray, seed_days: np.ndarray, fanout: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        batch_size = len(customers)
        review = np.zeros((batch_size, fanout, 7), dtype=np.float32)
        product = np.zeros((batch_size, fanout, 10), dtype=np.float32)
        mask = np.zeros((batch_size, fanout), dtype=np.float32)
        for row, (customer, seed_day) in enumerate(zip(customers, seed_days)):
            left = self.pointer[customer]
            right = self.pointer[customer + 1]
            end = left + np.searchsorted(self.event_day[left:right], seed_day, side="right")
            start = max(left, end - fanout)
            count = end - start
            if count <= 0:
                continue
            values = np.nan_to_num(self.values[start:end], nan=0.0, posinf=0.0, neginf=0.0)
            days = self.event_day[start:end]
            offset = fanout - count
            ages = np.clip(seed_day - days, 0, 730) / 91.0
            review[row, offset:, 0] = ages
            review[row, offset:, 1] = values[:, 0] / 5.0
            review[row, offset:, 2] = values[:, 1]
            review[row, offset:, 3] = np.log1p(values[:, 2]) / 8.0
            review[row, offset:, 4] = np.log1p(np.nan_to_num(values[:, 3])) / 5.0
            review[row, offset:, 5] = np.sin(2 * np.pi * (days % 365) / 365.0)
            review[row, offset:, 6] = np.cos(2 * np.pi * (days % 365) / 365.0)
            product[row, offset:, 0] = np.log1p(np.clip(values[:, 4], 0, 1000)) / 7.0
            product[row, offset:, 1] = np.sin(2 * np.pi * values[:, 5] / 65521.0)
            product[row, offset:, 2] = np.cos(2 * np.pi * values[:, 5] / 65521.0)
            product[row, offset:, 3] = np.sin(2 * np.pi * values[:, 6] / 65521.0)
            product[row, offset:, 4] = np.cos(2 * np.pi * values[:, 6] / 65521.0)
            product[row, offset:, 5] = np.log1p(values[:, 7]) / 5.0
            product[row, offset:, 6] = np.log1p(values[:, 8]) / 9.0
            product[row, offset:, 7] = values[:, 9]
            product[row, offset:, 8] = values[:, 10]
            product[row, offset:, 9] = np.minimum(ages, 1.0)
            mask[row, offset:] = 1.0
        return review, product, mask


def compact_rfm(frame: pd.DataFrame) -> np.ndarray:
    names = [
        "recency_days", "tenure_days", "n_7", "n_30", "n_91", "n_365", "n_all",
        "products_91", "products_365", "days_active_91", "gap_1", "gap_2",
        "rating_91", "rating_std_365", "verified_91", "text_length_91",
        "price_91", "share_30_91", "share_91_365", "fast_slow_30_365",
        "product_diversity_91", "recency_percentile", "activity_91_percentile",
        "activity_365_percentile", "tenure_percentile", "customer_target",
        "cohort_target", "history_label_n", "historical_return_share", "season_sin", "season_cos",
    ]
    data = frame[names].astype(np.float32).to_numpy()
    data = np.nan_to_num(data, nan=0.0, posinf=10.0, neginf=-10.0)
    data[:, 0] /= 91.0
    data[:, 1] = np.log1p(np.clip(data[:, 1], 0, None)) / 9.0
    for column in range(2, 10):
        data[:, column] = np.log1p(np.clip(data[:, column], 0, None)) / 7.0
    data[:, 10:12] = np.clip(data[:, 10:12], 0, 730) / 91.0
    data[:, 12] /= 5.0
    data[:, 13] /= 2.5
    data[:, 15] = np.log1p(np.clip(data[:, 15], 0, None)) / 8.0
    data[:, 16] = np.log1p(np.clip(data[:, 16], 0, None)) / 7.0
    data[:, 17:25] = np.clip(data[:, 17:25], 0, 10)
    data[:, 27] = np.log1p(np.clip(data[:, 27], 0, None)) / 4.0
    return data.astype(np.float32)


class GraphSeedNet(nn.Module):
    def __init__(self, rfm_channels: int, hidden: int = 128, dropout: float = 0.2):
        super().__init__()
        self.product_projection = nn.Linear(10, hidden)
        self.review_projection = nn.Linear(7, hidden)
        self.customer_projection = nn.Linear(rfm_channels, hidden)
        self.product_to_review = nn.Linear(hidden * 2, hidden)
        self.review_norm = nn.LayerNorm(hidden)
        self.review_to_customer = nn.Linear(hidden * 2, hidden)
        self.customer_norm = nn.LayerNorm(hidden)
        self.dropout = dropout
        self.head = nn.Sequential(nn.Linear(hidden + rfm_channels, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, review: torch.Tensor, product: torch.Tensor, mask: torch.Tensor, rfm: torch.Tensor) -> torch.Tensor:
        product_hidden = F.relu(self.product_projection(product))
        review_residual = self.review_projection(review)
        review_hidden = self.product_to_review(torch.cat([review_residual, product_hidden], dim=-1))
        review_hidden = F.dropout(F.relu(self.review_norm(review_hidden + review_residual)), self.dropout, self.training)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        message = (review_hidden * mask.unsqueeze(-1)).sum(dim=1) / denominator
        customer_residual = self.customer_projection(rfm)
        customer_hidden = self.review_to_customer(torch.cat([customer_residual, message], dim=-1))
        customer_hidden = F.dropout(F.relu(self.customer_norm(customer_hidden + customer_residual)), self.dropout, self.training)
        return self.head(torch.cat([customer_hidden, rfm], dim=1)).squeeze(1)


def origin_balanced_indices(origins: np.ndarray, maximum_origin: int, size: int, rng: np.random.Generator) -> np.ndarray:
    eligible_origins = np.unique(origins[origins < maximum_origin])
    if len(eligible_origins) == 0:
        return np.empty(0, dtype=np.int64)
    per_origin = max(1, size // len(eligible_origins))
    selected = []
    for origin in eligible_origins:
        candidates = np.flatnonzero(origins == origin)
        selected.append(rng.choice(candidates, size=per_origin, replace=len(candidates) < per_origin))
    result = np.concatenate(selected)
    rng.shuffle(result)
    return result[:size]


def seed_days(origin_indices: np.ndarray, origins: list[pd.Timestamp]) -> np.ndarray:
    days = np.array([int(value.timestamp() // 86400) for value in origins], dtype=np.int32)
    return days[origin_indices]


def graph_predict(model: GraphSeedNet, store: EventStore, customers: np.ndarray, days: np.ndarray, rfm: np.ndarray, device: torch.device, batch_size: int = 512, fanout: int = 16) -> np.ndarray:
    model.eval()
    result = np.empty(len(customers), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(customers), batch_size):
            stop = min(start + batch_size, len(customers))
            review, product, mask = store.batch(customers[start:stop], days[start:stop], fanout)
            logits = model(
                torch.from_numpy(review).to(device),
                torch.from_numpy(product).to(device),
                torch.from_numpy(mask).to(device),
                torch.from_numpy(rfm[start:stop]).to(device),
            )
            result[start:stop] = torch.sigmoid(logits).cpu().numpy()
    return result


def expanding_graph_oof(frame: pd.DataFrame, labels: pd.DataFrame, debug: bool = False) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rejection = cache_root() / "graph_referee_rejected_v1.json"
    if rejection.exists() and not debug:
        payload = json.loads(rejection.read_text())
        print(f"[lane3] graph_referee_cache=rejected median_delta={payload['median_delta']:.6f}", flush=True)
        predictions = np.full(len(frame), np.nan, dtype=np.float32)
        return predictions, np.zeros(len(frame), dtype=bool), payload["graph_folds"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    origins = frame["origin_index"].to_numpy(np.int16)
    customers = frame["customer_id"].to_numpy(np.int32)
    all_origins = read_origins("train")
    days = seed_days(origins, all_origins)
    rfm = compact_rfm(frame)
    label_vector = labels.set_index("row_id")["churn"]
    y = label_vector.loc[frame["row_id"]].to_numpy(np.float32)
    store = EventStore(debug=debug)
    model = GraphSeedNet(rfm.shape[1], 128, 0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    predictions = np.full(len(frame), np.nan, dtype=np.float32)
    fold_rows = []
    folds = (1,) if debug else OOF_ORIGIN_INDICES
    samples = 20000 if debug else 500000
    fanout = 8 if debug else 16
    batch_size = 512
    for fold in folds:
        for _ in range(2):
            selected = origin_balanced_indices(origins, fold, samples, rng)
            model.train()
            for start in range(0, len(selected), batch_size):
                index = selected[start : start + batch_size]
                review, product, mask = store.batch(customers[index], days[index], fanout)
                logits = model(
                    torch.from_numpy(review).to(device),
                    torch.from_numpy(product).to(device),
                    torch.from_numpy(mask).to(device),
                    torch.from_numpy(rfm[index]).to(device),
                )
                target = torch.from_numpy(y[index]).to(device)
                loss = F.binary_cross_entropy_with_logits(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print(f"[lane3] graph_pseudo_epoch fold={fold} seeds={len(selected)}", flush=True)
        hold = np.flatnonzero(origins == fold)
        if len(hold):
            pred = graph_predict(model, store, customers[hold], days[hold], rfm[hold], device, batch_size, fanout)
            predictions[hold] = pred
            auc = float(roc_auc_score(y[hold], pred))
            fold_rows.append({"fold": int(fold), "hold_n": int(len(hold)), "auc": auc})
            print(f"[lane3] graph_fold={fold} hold={len(hold)} auc={auc:.6f}", flush=True)
    return predictions, np.isfinite(predictions), fold_rows


def fit_graph_final(frame: pd.DataFrame, labels: np.ndarray, prediction_frame: pd.DataFrame, debug: bool = False) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    origins = frame["origin_index"].to_numpy(np.int16)
    train_origins = read_origins("train")
    if int(origins.max()) >= len(train_origins):
        train_origins = train_origins + [pd.Timestamp("2015-10-01 00:00:00")]
    train_days = seed_days(origins, train_origins)
    train_rfm = compact_rfm(frame)
    store = EventStore(debug=debug)
    model = GraphSeedNet(train_rfm.shape[1], 128, 0.2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    customers = frame["customer_id"].to_numpy(np.int32)
    samples = 20000 if debug else 500000
    fanout = 8 if debug else 16
    batch_size = 512
    maximum = int(origins.max()) + 1
    for _ in range(6):
        selected = origin_balanced_indices(origins, maximum, samples, rng)
        model.train()
        for start in range(0, len(selected), batch_size):
            index = selected[start : start + batch_size]
            review, product, mask = store.batch(customers[index], train_days[index], fanout)
            logits = model(
                torch.from_numpy(review).to(device),
                torch.from_numpy(product).to(device),
                torch.from_numpy(mask).to(device),
                torch.from_numpy(train_rfm[index]).to(device),
            )
            target = torch.from_numpy(labels[index].astype(np.float32)).to(device)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    prediction_rfm = compact_rfm(prediction_frame)
    prediction_customers = prediction_frame["customer_id"].to_numpy(np.int32)
    prediction_day = np.full(len(prediction_frame), int(pd.Timestamp(prediction_frame.attrs["timestamp"]).timestamp() // 86400), dtype=np.int32)
    return graph_predict(model, store, prediction_customers, prediction_day, prediction_rfm, device, batch_size, fanout)
