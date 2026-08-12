from __future__ import annotations

import copy
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from temporal_pipeline import DAY, NODE_COUNTS, TemporalData, sample_relation_routes, seconds


def padded_mapping(values: np.ndarray) -> torch.Tensor:
    result = np.zeros(len(values) + 1, dtype=np.int64)
    raw = np.asarray(values, dtype=np.int64)
    result[1:] = np.where(raw >= 0, raw + 1, 0)
    return torch.from_numpy(result)


def masked_mean(values: torch.Tensor, weights: torch.Tensor, dimension: int) -> torch.Tensor:
    expanded = weights.unsqueeze(-1)
    total = (values * expanded).sum(dim=dimension)
    denominator = expanded.sum(dim=dimension).clamp_min(1e-6)
    return total / denominator


class TemporalHeteroGraphSAGE(nn.Module):
    def __init__(self, data: TemporalData, hidden: int = 96, input_channels: int = 64, dropout: float = 0.2):
        super().__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(count + 1, input_channels, padding_idx=0) for name, count in NODE_COUNTS.items()}
        )
        self.projections = nn.ModuleDict({name: nn.Linear(input_channels, input_channels) for name in NODE_COUNTS})
        self.register_buffer("beer_style", padded_mapping(data.beer_style), persistent=False)
        self.register_buffer("beer_brewer", padded_mapping(data.beer_brewer), persistent=False)
        self.register_buffer("brewer_state", padded_mapping(data.brewer_state), persistent=False)
        self.register_buffer("brewer_country", padded_mapping(data.brewer_country), persistent=False)
        self.register_buffer("brewer_type", padded_mapping(data.brewer_type), persistent=False)
        self.register_buffer("place_state", padded_mapping(data.place_state), persistent=False)
        self.register_buffer("place_country", padded_mapping(data.place_country), persistent=False)
        self.register_buffer("place_type", padded_mapping(data.place_type), persistent=False)
        self.register_buffer("state_country", padded_mapping(data.state_country), persistent=False)
        self.beer_static = nn.Linear(input_channels, input_channels)
        self.brewer_static = nn.Linear(input_channels, input_channels)
        self.place_static = nn.Linear(input_channels, input_channels)
        self.state_static = nn.Linear(input_channels, input_channels)
        self.beer_root = nn.Linear(input_channels, hidden)
        self.beer_neighbor = nn.Linear(input_channels, hidden)
        self.place_root = nn.Linear(input_channels, hidden)
        self.place_neighbor = nn.Linear(input_channels, hidden)
        self.user_root = nn.Linear(input_channels, hidden)
        self.rating_message = nn.Linear(hidden, hidden)
        self.place_message = nn.Linear(hidden, hidden)
        self.favorite_message = nn.Linear(hidden, hidden)
        self.first_norm = nn.LayerNorm(hidden)
        self.output_norm = nn.LayerNorm(hidden)
        self.link_user = nn.Linear(hidden, hidden)
        self.link_beer = nn.Linear(input_channels, hidden)
        self.link_place = nn.Linear(input_channels, hidden)
        self.event_type = nn.Linear(hidden, 3)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()

    def base(self, node_type: str, identifiers: torch.Tensor) -> torch.Tensor:
        mask = identifiers.ne(0).unsqueeze(-1)
        return self.projections[node_type](self.embeddings[node_type](identifiers)) * mask

    def state_base(self, identifiers: torch.Tensor) -> torch.Tensor:
        own = self.base("state", identifiers)
        country = self.base("country", self.state_country[identifiers])
        return own + self.state_static(country)

    def brewer_base(self, identifiers: torch.Tensor) -> torch.Tensor:
        own = self.base("brewer", identifiers)
        state = self.state_base(self.brewer_state[identifiers])
        country = self.base("country", self.brewer_country[identifiers])
        place_type = self.base("place_type", self.brewer_type[identifiers])
        return own + self.brewer_static(state + country + place_type)

    def beer_base(self, identifiers: torch.Tensor) -> torch.Tensor:
        own = self.base("beer", identifiers)
        style = self.base("style", self.beer_style[identifiers])
        brewer = self.brewer_base(self.beer_brewer[identifiers])
        return own + self.beer_static(style + brewer)

    def place_base(self, identifiers: torch.Tensor) -> torch.Tensor:
        own = self.base("place", identifiers)
        state = self.state_base(self.place_state[identifiers])
        country = self.base("country", self.place_country[identifiers])
        place_type = self.base("place_type", self.place_type[identifiers])
        return own + self.place_static(state + country + place_type)

    def relation_hidden(
        self,
        entities: torch.Tensor,
        neighbors: torch.Tensor,
        first_weight: torch.Tensor,
        second_weight: torch.Tensor,
        node_type: str,
    ) -> torch.Tensor:
        neighbor_base = self.base("user", neighbors)
        neighbor_mean = masked_mean(neighbor_base, second_weight, 2)
        if node_type == "beer":
            root = self.beer_base(entities)
            hidden = self.beer_root(root) + self.beer_neighbor(neighbor_mean)
        else:
            root = self.place_base(entities)
            hidden = self.place_root(root) + self.place_neighbor(neighbor_mean)
        hidden = F.relu(self.first_norm(hidden))
        return masked_mean(hidden, first_weight, 1)

    def encode(
        self,
        users: torch.Tensor,
        rating_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        place_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        favorite_route: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        rating = self.relation_hidden(*rating_route, "beer")
        place = self.relation_hidden(*place_route, "place")
        favorite = self.relation_hidden(*favorite_route, "beer")
        root = self.base("user", users)
        hidden = self.user_root(root)
        hidden = hidden + self.rating_message(rating) + self.place_message(place) + self.favorite_message(favorite)
        hidden = self.output_norm(hidden)
        return F.dropout(F.relu(hidden), p=self.dropout, training=self.training)

    def destination(self, node_type: str, identifiers: torch.Tensor) -> torch.Tensor:
        if node_type == "beer":
            return self.link_beer(self.beer_base(identifiers))
        return self.link_place(self.place_base(identifiers))


def route_tensors(data: TemporalData, users, event_time, fanout, inclusive, device):
    routes = []
    event_time = np.asarray(event_time, dtype=np.int64)
    for relation in [data.beer, data.place, data.favorite]:
        entities, neighbors, first_times, second_times = sample_relation_routes(
            relation, users, event_time, fanout, inclusive
        )
        entity_ids = torch.from_numpy(np.maximum(entities + 1, 0)).to(device)
        neighbor_ids = torch.from_numpy(np.maximum(neighbors + 1, 0)).to(device)
        first_age = np.maximum(event_time[:, None] - first_times, 0) / (90.0 * DAY)
        second_age = np.maximum(event_time[:, None, None] - second_times, 0) / (90.0 * DAY)
        first_weight = np.where(first_times >= 0, np.exp(-first_age), 0).astype(np.float32)
        second_weight = np.where(second_times >= 0, np.exp(-second_age), 0).astype(np.float32)
        routes.append(
            (
                entity_ids,
                neighbor_ids,
                torch.from_numpy(first_weight).to(device),
                torch.from_numpy(second_weight).to(device),
            )
        )
    return routes


def existing_order(first_time: np.ndarray):
    valid = np.flatnonzero(first_time < np.iinfo(np.int64).max)
    order = valid[np.argsort(np.asarray(first_time[valid]))]
    return order.astype(np.int32), np.asarray(first_time[order], dtype=np.int64)


def sample_negatives(order, ordered_time, event_time, positive, rng):
    limits = np.searchsorted(ordered_time, event_time, side="right")
    limits = np.maximum(limits, 1)
    choices = (rng.random(len(event_time)) * limits).astype(np.int64)
    result = order[choices]
    collision = result == positive
    attempts = 0
    while collision.any() and attempts < 5:
        choices[collision] = (rng.random(collision.sum()) * limits[collision]).astype(np.int64)
        result[collision] = order[choices[collision]]
        collision = result == positive
        attempts += 1
    return result.astype(np.int32)


def auxiliary_events(data: TemporalData, cutoff: int, count: int, rng: np.random.Generator):
    relation_counts = [int(count * 0.60), int(count * 0.25)]
    relation_counts.append(count - sum(relation_counts))
    pieces = []
    for event_type, (relation, size) in enumerate(zip([data.beer, data.place, data.favorite], relation_counts)):
        users, entities, event_time = relation.sample_events(cutoff, size, rng)
        pieces.append(
            (
                users,
                entities,
                event_time,
                np.full(size, event_type, dtype=np.int64),
            )
        )
    users = np.concatenate([piece[0] for piece in pieces])
    entities = np.concatenate([piece[1] for piece in pieces])
    event_time = np.concatenate([piece[2] for piece in pieces])
    event_type = np.concatenate([piece[3] for piece in pieces])
    order = rng.permutation(len(users))
    return users[order], entities[order], event_time[order], event_type[order]


def pretrain_encoder(
    data: TemporalData,
    cutoff: int,
    checkpoint: Path,
    fanout: tuple[int, int],
    debug: bool,
    seed: int,
    initial_state=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TemporalHeteroGraphSAGE(data).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    if checkpoint.exists() and not debug:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        print(f"[graph] loaded checkpoint {checkpoint.name}")
        return model
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    beer_first = np.minimum(np.asarray(data.beer.entity_first), np.asarray(data.beer_created))
    beer_order, beer_ordered_time = existing_order(beer_first)
    place_order, place_ordered_time = existing_order(data.place.entity_first)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    epochs = 2
    sample_count = 51200 if debug else 300000
    batch_size = 256 if debug else 512
    maximum_steps = 200 if debug else None
    completed_steps = 0
    start = time.time()
    for epoch in range(epochs):
        users, entities, event_time, event_type = auxiliary_events(data, cutoff, sample_count, rng)
        order = rng.permutation(len(users))
        losses = []
        model.train()
        for offset in range(0, len(order), batch_size):
            if maximum_steps is not None and completed_steps >= maximum_steps:
                break
            indices = order[offset : offset + batch_size]
            batch_users = users[indices]
            batch_entities = entities[indices]
            batch_time = event_time[indices]
            batch_type = event_type[indices]
            routes = route_tensors(data, batch_users, batch_time, fanout, False, device)
            user_tensor = torch.from_numpy(batch_users.astype(np.int64) + 1).to(device)
            entity_tensor = torch.from_numpy(batch_entities.astype(np.int64) + 1).to(device)
            type_tensor = torch.from_numpy(batch_type).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                embedding = model.encode(user_tensor, routes[0], routes[1], routes[2])
                type_loss = F.cross_entropy(model.event_type(embedding), type_tensor)
                link_losses = []
                beer_mask_numpy = batch_type != 1
                place_mask_numpy = batch_type == 1
                if beer_mask_numpy.any():
                    mask = torch.from_numpy(beer_mask_numpy).to(device)
                    negative = sample_negatives(
                        beer_order,
                        beer_ordered_time,
                        batch_time[beer_mask_numpy],
                        batch_entities[beer_mask_numpy],
                        rng,
                    )
                    negative_tensor = torch.from_numpy(negative.astype(np.int64) + 1).to(device)
                    source = model.link_user(embedding[mask])
                    positive_score = (source * model.destination("beer", entity_tensor[mask])).sum(-1) / math.sqrt(model.hidden)
                    negative_score = (source * model.destination("beer", negative_tensor)).sum(-1) / math.sqrt(model.hidden)
                    link_losses.append(F.softplus(-positive_score).mean() + F.softplus(negative_score).mean())
                if place_mask_numpy.any():
                    mask = torch.from_numpy(place_mask_numpy).to(device)
                    negative = sample_negatives(
                        place_order,
                        place_ordered_time,
                        batch_time[place_mask_numpy],
                        batch_entities[place_mask_numpy],
                        rng,
                    )
                    negative_tensor = torch.from_numpy(negative.astype(np.int64) + 1).to(device)
                    source = model.link_user(embedding[mask])
                    positive_score = (source * model.destination("place", entity_tensor[mask])).sum(-1) / math.sqrt(model.hidden)
                    negative_score = (source * model.destination("place", negative_tensor)).sum(-1) / math.sqrt(model.hidden)
                    link_losses.append(F.softplus(-positive_score).mean() + F.softplus(negative_score).mean())
                loss = type_loss + 0.5 * torch.stack(link_losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            completed_steps += 1
        print(f"[graph] epoch={epoch + 1} steps={completed_steps} loss={np.mean(losses):.6f} elapsed={time.time() - start:.1f}s")
        if maximum_steps is not None and completed_steps >= maximum_steps:
            break
    if not debug:
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(model.state_dict(), temporary)
        os.replace(temporary, checkpoint)
    return model


def extract_embeddings(model, data: TemporalData, users, timestamps, fanout, batch_size=2048):
    device = next(model.parameters()).device
    users = np.asarray(users, dtype=np.int32)
    event_time = seconds(timestamps)
    result = np.empty((len(users), model.hidden), dtype=np.float32)
    model.eval()
    start = time.time()
    with torch.no_grad():
        for offset in range(0, len(users), batch_size):
            limit = min(offset + batch_size, len(users))
            batch_users = users[offset:limit]
            batch_time = event_time[offset:limit]
            routes = route_tensors(data, batch_users, batch_time, fanout, True, device)
            user_tensor = torch.from_numpy(batch_users.astype(np.int64) + 1).to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                embedding = model.encode(user_tensor, routes[0], routes[1], routes[2])
            result[offset:limit] = embedding.float().cpu().numpy()
    elapsed = time.time() - start
    rate = len(users) / max(elapsed, 1e-6)
    print(f"[graph] embedded rows={len(users)} seconds={elapsed:.1f} rate={rate:.1f}/s")
    return result, rate


class ChurnHead(nn.Module):
    def __init__(self, width: int = 96):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(width, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)


def fit_churn_head(embedding, labels, epochs, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    model = ChurnHead(embedding.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    embedding = np.asarray(embedding, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(labels))
        for offset in range(0, len(order), 4096):
            indices = order[offset : offset + 4096]
            values = torch.from_numpy(embedding[indices]).to(device)
            target = torch.from_numpy(labels[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            loss.backward()
            optimizer.step()
    return model


def predict_churn_head(model, embedding):
    device = next(model.parameters()).device
    embedding = np.asarray(embedding, dtype=np.float32)
    result = np.empty(len(embedding), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(embedding), 8192):
            values = torch.from_numpy(embedding[offset : offset + 8192]).to(device)
            result[offset : offset + len(values)] = torch.sigmoid(model(values)).cpu().numpy()
    return result


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
