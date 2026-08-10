from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import HeteroConv, LayerNorm, SAGEConv

from relbench.modeling.nn import HeteroTemporalEncoder


# Configuration

@dataclass
class GraphConfig:
    hidden: int = 128
    layers: int = 2
    fanout: tuple[int, int] = (64, 32)
    batch_size: int = 1024
    dropout: float = 0.20
    learning_rate: float = 5e-3
    weight_decay: float = 1e-5
    epochs: int = 8
    seeds: tuple[int, ...] = (2026, 3407)
    num_workers: int = 0


# Batch transforms

class AttachSupervision:
    def __init__(self, entity: str, target: Tensor, weight: Tensor):
        self.entity = entity
        self.target = target
        self.weight = weight

    def __call__(self, batch: HeteroData) -> HeteroData:
        index = batch[self.entity].input_id
        batch[self.entity].y = self.target[index]
        batch[self.entity].weight = self.weight[index]
        return batch


# Model

class NodeEncoder(torch.nn.Module):
    def __init__(
        self,
        numerical_dim: int,
        text_dim: int,
        cardinalities: list[int],
        hidden: int,
    ):
        super().__init__()
        self.numerical = torch.nn.Linear(numerical_dim, hidden)
        self.text = torch.nn.Linear(text_dim, hidden, bias=False) if text_dim else None
        self.categories = torch.nn.ModuleList(
            [torch.nn.Embedding(cardinality, hidden) for cardinality in cardinalities]
        )
        self.norm = torch.nn.LayerNorm(hidden)

    def forward(self, store) -> Tensor:
        value = self.numerical(store.x_num.float())
        if self.text is not None:
            value = value + self.text(store.x_text.float())
        if len(self.categories):
            for column, encoder in enumerate(self.categories):
                value = value + encoder(store.cat[:, column])
        return self.norm(F.gelu(value))


class HeterogeneousGraphSAGE(torch.nn.Module):
    def __init__(
        self,
        data: HeteroData,
        feature_meta: dict[str, dict[str, object]],
        hidden: int,
        layers: int,
        dropout: float,
    ):
        super().__init__()
        self.encoders = torch.nn.ModuleDict(
            {
                node_type: NodeEncoder(
                    int(feature_meta[node_type]["numerical_dim"]),
                    int(feature_meta[node_type]["text_dim"]),
                    list(feature_meta[node_type]["cardinalities"]),
                    hidden,
                )
                for node_type in data.node_types
            }
        )
        timed_nodes = [node_type for node_type in data.node_types if "time" in data[node_type]]
        self.temporal = HeteroTemporalEncoder(timed_nodes, hidden)
        self.convolutions = torch.nn.ModuleList()
        self.normalizations = torch.nn.ModuleList()
        for _ in range(layers):
            self.convolutions.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv((hidden, hidden), hidden, aggr="mean")
                        for edge_type in data.edge_types
                    },
                    aggr="sum",
                )
            )
            self.normalizations.append(
                torch.nn.ModuleDict(
                    {node_type: LayerNorm(hidden, mode="node") for node_type in data.node_types}
                )
            )
        self.dropout = float(dropout)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, batch: HeteroData, entity: str) -> Tensor:
        seed_time = batch[entity].seed_time
        encoded = {node_type: self.encoders[node_type](batch[node_type]) for node_type in batch.node_types}
        relative = self.temporal(seed_time, batch.time_dict, batch.batch_dict)
        for node_type, value in relative.items():
            encoded[node_type] = encoded[node_type] + value
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            encoded = convolution(encoded, batch.edge_index_dict)
            encoded = {
                node_type: F.dropout(
                    F.relu(normalization[node_type](value)),
                    p=self.dropout,
                    training=self.training,
                )
                for node_type, value in encoded.items()
            }
        return self.head(encoded[entity][: seed_time.numel()]).view(-1)


# Loader construction

def effective_fanout(data: HeteroData, fanout: tuple[int, int]) -> dict:
    incoming: dict[str, int] = {node_type: 0 for node_type in data.node_types}
    for _, _, destination in data.edge_types:
        incoming[destination] += 1
    return {
        edge_type: [
            max(2, int(math.ceil(fanout[layer] / max(1, incoming[edge_type[2]]))))
            for layer in range(len(fanout))
        ]
        for edge_type in data.edge_types
    }


def make_loader(
    data: HeteroData,
    entity: str,
    node_ids: np.ndarray,
    timestamps: np.ndarray,
    config: GraphConfig,
    shuffle: bool,
    targets: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> NeighborLoader:
    node_tensor = torch.from_numpy(np.asarray(node_ids, dtype=np.int64))
    time_tensor = torch.from_numpy(np.asarray(timestamps, dtype=np.int64))
    transform = None
    if targets is not None:
        transform = AttachSupervision(
            entity,
            torch.from_numpy(np.asarray(targets, dtype=np.float32)),
            torch.from_numpy(np.asarray(weights, dtype=np.float32)),
        )
    return NeighborLoader(
        data,
        num_neighbors=effective_fanout(data, config.fanout),
        time_attr="time",
        input_nodes=(entity, node_tensor),
        input_time=time_tensor,
        transform=transform,
        batch_size=config.batch_size,
        temporal_strategy="uniform",
        shuffle=shuffle,
        num_workers=config.num_workers,
        persistent_workers=config.num_workers > 0,
    )


def assert_temporal_batch(batch: HeteroData, entity: str) -> None:
    seed_time = batch[entity].seed_time
    for node_type, times in batch.time_dict.items():
        if not times.numel():
            continue
        assignment = batch[node_type].batch
        if torch.any(times > seed_time[assignment]):
            excess = int((times > seed_time[assignment]).sum())
            raise RuntimeError(f"temporal sampler exposed {excess} future {node_type} nodes")


# Training and inference

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_graph_model(
    data: HeteroData,
    feature_meta: dict[str, dict[str, object]],
    entity: str,
    node_ids: np.ndarray,
    timestamps: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    config: GraphConfig,
    seed: int,
    device: torch.device,
    epoch_callback: Callable[[int, HeterogeneousGraphSAGE, float, float], bool] | None = None,
) -> tuple[HeterogeneousGraphSAGE, list[float]]:
    seed_all(seed)
    loader = make_loader(
        data,
        entity,
        node_ids,
        timestamps,
        config,
        True,
        targets,
        weights,
    )
    model = HeterogeneousGraphSAGE(
        data,
        feature_meta,
        config.hidden,
        config.layers,
        config.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    epoch_times: list[float] = []
    asserted = False
    for epoch in range(1, config.epochs + 1):
        model.train()
        started = time.time()
        loss_total = 0.0
        weight_total = 0.0
        for batch in loader:
            if not asserted:
                assert_temporal_batch(batch, entity)
                asserted = True
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch, entity)
            loss_rows = F.binary_cross_entropy_with_logits(
                logits,
                batch[entity].y.float(),
                reduction="none",
            )
            row_weights = batch[entity].weight.float()
            loss = (loss_rows * row_weights).sum() / row_weights.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += float((loss_rows.detach() * row_weights).sum().cpu())
            weight_total += float(row_weights.sum().cpu())
        elapsed = time.time() - started
        epoch_times.append(elapsed)
        mean_loss = loss_total / max(weight_total, 1e-8)
        print(f"[gnn] seed={seed} epoch={epoch}/{config.epochs} loss={mean_loss:.6f} elapsed={elapsed:.1f}s")
        if epoch_callback is not None and not epoch_callback(epoch, model, mean_loss, elapsed):
            break
    return model, epoch_times


@torch.no_grad()
def predict_graph_model(
    model: HeterogeneousGraphSAGE,
    data: HeteroData,
    entity: str,
    node_ids: np.ndarray,
    timestamps: np.ndarray,
    config: GraphConfig,
    device: torch.device,
) -> np.ndarray:
    if not len(node_ids):
        return np.empty(0, dtype=np.float64)
    loader = make_loader(data, entity, node_ids, timestamps, config, False)
    model.eval()
    predictions: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        predictions.append(torch.sigmoid(model(batch, entity)).cpu().numpy())
    return np.concatenate(predictions).astype(np.float64, copy=False)


def origin_auc(labels: np.ndarray, predictions: np.ndarray, origins: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for origin in np.unique(origins):
        mask = origins == origin
        result[str(origin)] = float(roc_auc_score(labels[mask], predictions[mask]))
    return result
