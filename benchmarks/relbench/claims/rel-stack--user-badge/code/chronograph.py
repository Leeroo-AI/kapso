from __future__ import annotations

import math
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from numpy.lib.format import open_memmap
from torch import nn
from torch.nn import functional as F

from temporal_features import cache_root, database_root, register_artifact

try:
    from torch_geometric.utils import scatter as pyg_scatter

    GRAPH_BACKEND = "pyg"
except Exception:
    pyg_scatter = None
    GRAPH_BACKEND = "torch"


# Events

def build_graph_events(user_ids: np.ndarray) -> tuple[dict[str, np.ndarray], int]:
    root = cache_root() / "graph_events"
    keys = ("time", "source", "destination", "event_type", "category", "relation", "numeric")
    paths = {name: root / f"{name}.npy" for name in keys}
    post_path = root / "post_ids.npy"
    if all(path.exists() for path in paths.values()) and post_path.exists():
        arrays = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
        return arrays, len(np.load(post_path, mmap_mode="r"))
    root.mkdir(parents=True, exist_ok=True)
    db = database_root()
    post_ids = duckdb.sql(f"select Id from read_parquet('{db / 'posts.parquet'}') order by Id").fetchnumpy()["Id"].astype(np.int64)
    sql = f"""
    with posts as materialized (select * from read_parquet('{db / 'posts.parquet'}'))
    select CreationDate ts,Id src,0 src_type,Id dst,0 dst_type,0 event_type,0 category,0 relation,
           length(coalesce(DisplayName,''))+length(coalesce(AboutMe,'')) numeric1,
           length(coalesce(Location,''))+length(coalesce(WebsiteUrl,'')) numeric2,
           0 table_id,Id pk,0 sub from read_parquet('{db / 'users.parquet'}')
    union all
    select p.CreationDate,p.OwnerUserId,0,p.Id,1,
           case when p.PostTypeId=1 then 1 else 2 end,p.PostTypeId,1,
           length(coalesce(p.Body,''))+length(coalesce(p.Tags,'')),length(coalesce(p.Title,'')),
           1,p.Id,0 from posts p where p.OwnerUserId is not null
    union all
    select p.CreationDate,p.Id,1,p.Id,1,
           case when p.PostTypeId=1 then 1 else 2 end,p.PostTypeId,0,
           length(coalesce(p.Body,''))+length(coalesce(p.Tags,'')),length(coalesce(p.Title,'')),
           1,p.Id,0 from posts p where p.OwnerUserId is null
    union all
    select p.CreationDate,p.Id,1,p.ParentId,1,3,p.PostTypeId,2,
           length(coalesce(p.Body,'')),length(coalesce(p.Title,'')),1,p.Id,1
           from posts p join posts parent on p.ParentId=parent.Id
    union all
    select c.CreationDate,c.UserId,0,c.PostId,1,4,0,
           case when c.UserId=p.OwnerUserId then 1 else 2 end,
           length(coalesce(c.Text,'')),0,2,c.Id,0
           from read_parquet('{db / 'comments.parquet'}') c join posts p on c.PostId=p.Id
           where c.UserId is not null
    union all
    select h.CreationDate,h.UserId,0,h.PostId,1,5,h.PostHistoryTypeId,
           case when h.UserId=p.OwnerUserId then 1 else 2 end,
           length(coalesce(h.Text,'')),length(coalesce(h.Comment,'')),3,h.Id,0
           from read_parquet('{db / 'postHistory.parquet'}') h join posts p on h.PostId=p.Id
           where h.UserId is not null
    union all
    select v.CreationDate,v.PostId,1,coalesce(p.OwnerUserId,v.PostId),case when p.OwnerUserId is null then 1 else 0 end,
           6,v.VoteTypeId,3,1,0,4,v.Id,0
           from read_parquet('{db / 'votes.parquet'}') v join posts p on v.PostId=p.Id
    union all
    select l.CreationDate,l.PostId,1,l.RelatedPostId,1,7,l.LinkTypeId,4,1,0,5,l.Id,0
           from read_parquet('{db / 'postLinks.parquet'}') l join posts p on l.PostId=p.Id
           join posts r on l.RelatedPostId=r.Id
    union all
    select b.Date,b.UserId,0,b.UserId,0,8,abs(hash(b.Name))%2048,5,b.Class,length(coalesce(b.Name,'')),6,b.Id,0
           from read_parquet('{db / 'badges.parquet'}') b
    order by ts,table_id,pk,sub
    """
    frame = duckdb.sql(sql).df()
    source = np.empty(len(frame), dtype=np.int32)
    destination = np.empty(len(frame), dtype=np.int32)
    src_id = frame["src"].to_numpy(dtype=np.int64)
    dst_id = frame["dst"].to_numpy(dtype=np.int64)
    src_user = frame["src_type"].to_numpy(dtype=np.int8) == 0
    dst_user = frame["dst_type"].to_numpy(dtype=np.int8) == 0
    source[src_user] = np.searchsorted(user_ids, src_id[src_user]).astype(np.int32)
    source[~src_user] = len(user_ids) + np.searchsorted(post_ids, src_id[~src_user]).astype(np.int32)
    destination[dst_user] = np.searchsorted(user_ids, dst_id[dst_user]).astype(np.int32)
    destination[~dst_user] = len(user_ids) + np.searchsorted(post_ids, dst_id[~dst_user]).astype(np.int32)
    arrays = {
        "time": frame["ts"].to_numpy(dtype="datetime64[s]").astype(np.int64),
        "source": source,
        "destination": destination,
        "event_type": frame["event_type"].to_numpy(dtype=np.uint8),
        "category": frame["category"].to_numpy(dtype=np.uint16),
        "relation": frame["relation"].to_numpy(dtype=np.uint8),
        "numeric": np.log1p(frame[["numeric1", "numeric2"]].fillna(0).to_numpy(dtype=np.float32)),
    }
    del frame
    for name, array in arrays.items():
        np.save(paths[name], array)
    np.save(post_path, post_ids)
    register_artifact(
        "lane3 chronological heterogeneous graph events",
        root,
        "Deterministically ordered user/post events with typed endpoints and compact attributes.",
        "rel-stack-user-badge-lane3-graph-events-v1",
    )
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}, len(post_ids)


# Network

def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if pyg_scatter is not None:
        return pyg_scatter(values, index, dim=0, dim_size=size, reduce="mean")
    output = torch.zeros((size, values.shape[1]), device=values.device, dtype=values.dtype)
    output.index_add_(0, index, values)
    count = torch.zeros(size, device=values.device, dtype=values.dtype)
    count.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return output / count.clamp_min(1).unsqueeze(1)


def scatter_max(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    if pyg_scatter is not None:
        return pyg_scatter(values, index, dim=0, dim_size=size, reduce="max")
    output = torch.full((size,), torch.iinfo(values.dtype).min, device=values.device, dtype=values.dtype)
    output.scatter_reduce_(0, index, values, reduce="amax", include_self=True)
    return output


class TimeEncoding(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        frequencies = torch.exp(torch.linspace(math.log(1e-4), math.log(1.0), dimension))
        self.frequency = nn.Parameter(frequencies)
        self.phase = nn.Parameter(torch.zeros(dimension))

    def forward(self, delta_days: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(delta_days.clamp_min(0)).unsqueeze(-1)
        return torch.cos(scaled * self.frequency.float() + self.phase.float()).to(self.frequency.dtype)


class GraphState:
    def __init__(self, nodes: int, dimension: int, neighbors: int, device: torch.device, dtype: torch.dtype) -> None:
        self.memory = torch.zeros((nodes, dimension), device=device, dtype=dtype)
        self.last_time = torch.full((nodes,), -1.0, device=device, dtype=torch.float32)
        self.neighbors = torch.full((nodes, neighbors), -1, device=device, dtype=torch.int64)
        self.neighbor_time = torch.full((nodes, neighbors), -1.0, device=device, dtype=torch.float32)


class TemporalGraphMemory(nn.Module):
    def __init__(self, dimension: int = 96, message_dimension: int = 96, time_dimension: int = 32, heads: int = 2, neighbors: int = 15) -> None:
        super().__init__()
        self.dimension = dimension
        self.neighbor_count = neighbors
        self.heads = heads
        self.time_encoding = TimeEncoding(time_dimension)
        self.event_embedding = nn.Embedding(9, 16)
        self.category_embedding = nn.Embedding(2048, 16)
        self.relation_embedding = nn.Embedding(8, 8)
        self.numeric_network = nn.Sequential(nn.Linear(2, 16), nn.SiLU(), nn.Linear(16, 16))
        context_dimension = 16 + 16 + 8 + 16 + 2 * time_dimension
        self.message_network = nn.Sequential(
            nn.Linear(2 * dimension + context_dimension, message_dimension),
            nn.SiLU(),
            nn.Linear(message_dimension, message_dimension),
        )
        self.reverse_network = nn.Linear(message_dimension, message_dimension)
        self.query = nn.Linear(dimension, dimension)
        self.key = nn.Linear(dimension, dimension)
        self.value = nn.Linear(dimension, dimension)
        self.time_projection = nn.Linear(time_dimension, dimension)
        self.attention_output = nn.Linear(dimension, dimension)
        self.update = nn.GRUCell(message_dimension + dimension, dimension)
        self.event_head = nn.Sequential(nn.Linear(2 * dimension, dimension), nn.SiLU(), nn.Linear(dimension, 9))
        self.gap_head = nn.Sequential(nn.Linear(2 * dimension, dimension), nn.SiLU(), nn.Linear(dimension, 1))

    def recent_attention(self, nodes: torch.Tensor, node_time: torch.Tensor, state: GraphState) -> torch.Tensor:
        neighbor_ids = state.neighbors[nodes]
        valid = neighbor_ids >= 0
        safe = neighbor_ids.clamp_min(0)
        neighbor_memory = state.memory[safe]
        delta = (node_time.unsqueeze(1) - state.neighbor_time[nodes]).clamp_min(0) / 86400.0
        key = self.key(neighbor_memory) + self.time_projection(self.time_encoding(delta))
        value = self.value(neighbor_memory)
        query = self.query(state.memory[nodes])
        head_dimension = self.dimension // self.heads
        query = query.view(len(nodes), self.heads, head_dimension)
        key = key.view(len(nodes), self.neighbor_count, self.heads, head_dimension).transpose(1, 2)
        value = value.view(len(nodes), self.neighbor_count, self.heads, head_dimension).transpose(1, 2)
        scores = (query.unsqueeze(2) * key).sum(-1) / math.sqrt(head_dimension)
        scores = scores.masked_fill(~valid.unsqueeze(1), -1e4)
        weights = torch.softmax(scores.float(), dim=-1).to(value.dtype)
        weights = weights * valid.unsqueeze(1)
        attended = (weights.unsqueeze(-1) * value).sum(2).reshape(len(nodes), self.dimension)
        return self.attention_output(attended)

    def process(self, batch: dict[str, torch.Tensor], state: GraphState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source = batch["source"]
        destination = batch["destination"]
        timestamp = batch["time"].float()
        old_source = state.memory[source]
        old_destination = state.memory[destination]
        source_gap = torch.where(state.last_time[source] >= 0, (timestamp - state.last_time[source]).clamp_min(0) / 86400.0, torch.zeros_like(timestamp))
        destination_gap = torch.where(state.last_time[destination] >= 0, (timestamp - state.last_time[destination]).clamp_min(0) / 86400.0, torch.zeros_like(timestamp))
        context = torch.cat(
            [
                self.event_embedding(batch["event_type"]),
                self.category_embedding(batch["category"] % 2048),
                self.relation_embedding(batch["relation"]),
                self.numeric_network(batch["numeric"]),
                self.time_encoding(source_gap),
                self.time_encoding(destination_gap),
            ],
            dim=1,
        ).to(old_source.dtype)
        message = self.message_network(torch.cat([old_source, old_destination, context], dim=1))
        nodes, inverse = torch.unique(torch.cat([source, destination]), sorted=True, return_inverse=True)
        source_inverse = inverse[: len(source)]
        destination_inverse = inverse[len(source) :]
        combined_message = torch.cat([message, self.reverse_network(message)], dim=0)
        aggregated = scatter_mean(combined_message, inverse, len(nodes))
        combined_time = torch.cat([timestamp, timestamp])
        node_time = scatter_max(combined_time, inverse, len(nodes))
        attended = self.recent_attention(nodes, node_time, state)
        updated = self.update(torch.cat([aggregated, attended], dim=1), state.memory[nodes])
        updated_source = updated[source_inverse]
        updated_destination = updated[destination_inverse]
        event_logits = self.event_head(torch.cat([updated_source, updated_destination], dim=1))
        gap_prediction = self.gap_head(torch.cat([updated_source, updated_destination], dim=1)).squeeze(1)
        positive = (updated_source.float() * updated_destination.float()).sum(1) / math.sqrt(self.dimension)
        negative = (updated_source.float() * updated_destination.roll(1, 0).float()).sum(1) / math.sqrt(self.dimension)
        endpoint_loss = F.binary_cross_entropy_with_logits(torch.cat([positive, negative]), torch.cat([torch.ones_like(positive), torch.zeros_like(negative)]))
        gap_loss = F.smooth_l1_loss(gap_prediction.float(), torch.log1p(source_gap).float())
        positions = torch.arange(2 * len(source), device=source.device, dtype=torch.int64)
        last_positions = scatter_max(positions, inverse, len(nodes))
        other = torch.cat([destination, source])
        recent_neighbor = other[last_positions]
        recent_time = combined_time[last_positions]
        with torch.no_grad():
            state.memory[nodes] = updated.detach()
            state.last_time[nodes] = node_time
            state.neighbors[nodes, 1:] = state.neighbors[nodes, :-1].clone()
            state.neighbor_time[nodes, 1:] = state.neighbor_time[nodes, :-1].clone()
            state.neighbors[nodes, 0] = recent_neighbor
            state.neighbor_time[nodes, 0] = recent_time
        return event_logits, endpoint_loss, gap_loss


# Training and replay

def device_and_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda:0"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def tensor_batch(events: dict[str, np.ndarray], start: int, end: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "time": torch.as_tensor(np.asarray(events["time"][start:end]), device=device, dtype=torch.float32),
        "source": torch.as_tensor(np.asarray(events["source"][start:end]), device=device, dtype=torch.int64),
        "destination": torch.as_tensor(np.asarray(events["destination"][start:end]), device=device, dtype=torch.int64),
        "event_type": torch.as_tensor(np.asarray(events["event_type"][start:end]), device=device, dtype=torch.int64),
        "category": torch.as_tensor(np.asarray(events["category"][start:end]), device=device, dtype=torch.int64),
        "relation": torch.as_tensor(np.asarray(events["relation"][start:end]), device=device, dtype=torch.int64),
        "numeric": torch.as_tensor(np.asarray(events["numeric"][start:end]), device=device, dtype=dtype),
    }


def train_memory(events: dict[str, np.ndarray], node_count: int, cutoff: int, debug: bool) -> TemporalGraphMemory:
    device, dtype = device_and_dtype()
    dimension = 32 if debug else 96
    model = TemporalGraphMemory(dimension=dimension, message_dimension=dimension, time_dimension=32, heads=2, neighbors=15).to(device=device, dtype=dtype)
    checkpoint = cache_root() / ("tgn_debug_v1.pt" if debug else "tgn_full_v1.pt")
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        print(f"[graph] loaded checkpoint backend={GRAPH_BACKEND} dimension={dimension}", flush=True)
        return model
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    end = int(np.searchsorted(events["time"], cutoff, side="right"))
    begin = max(0, end - 100000) if debug else 0
    batch_size = 4096
    epochs = 1 if debug else 2
    for epoch in range(epochs):
        state = GraphState(node_count, dimension, 15, device, dtype)
        total_loss = 0.0
        batches = 0
        started = time.time()
        for start in range(begin, end, batch_size):
            stop = min(end, start + batch_size)
            batch = tensor_batch(events, start, stop, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            event_logits, endpoint_loss, gap_loss = model.process(batch, state)
            event_loss = F.cross_entropy(event_logits.float(), batch["event_type"])
            loss = 0.15 * event_loss + 0.10 * endpoint_loss + 0.05 * gap_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        rate = (end - begin) / max(1e-6, time.time() - started)
        print(f"[graph] pretrain_epoch={epoch + 1}/{epochs} loss={total_loss / max(1, batches):.6f} events_per_s={rate:.0f}", flush=True)
    torch.save(model.state_dict(), checkpoint)
    register_artifact(
        "lane3 temporal graph checkpoint",
        checkpoint,
        "Two-epoch self-supervised chronological TGN-style encoder checkpoint.",
        "rel-stack-user-badge-lane3-tgn-debug-v1" if debug else "rel-stack-user-badge-lane3-tgn-full-v1",
    )
    return model


def replay_memories(model: TemporalGraphMemory, events: dict[str, np.ndarray], node_count: int, frames: list[pd.DataFrame], mapped_users: np.ndarray, debug: bool) -> np.memmap:
    dimension = model.dimension
    path = cache_root() / ("memory_rows_debug_v1.npy" if debug else "memory_rows_full_v1.npy")
    total_rows = sum(len(frame) for frame in frames)
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == (total_rows, dimension):
            return matrix
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=(total_rows, dimension))
    combined_time = np.concatenate([frame["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64) for frame in frames])
    device, dtype = device_and_dtype()
    model.eval()
    state = GraphState(node_count, dimension, 15, device, dtype)
    batch_size = 4096
    event_start = max(0, len(events["time"]) - 100000) if debug else 0
    with torch.no_grad():
        for cutoff in np.unique(combined_time):
            event_end = int(np.searchsorted(events["time"], cutoff, side="right"))
            for start in range(event_start, event_end, batch_size):
                stop = min(event_end, start + batch_size)
                model.process(tensor_batch(events, start, stop, device, dtype), state)
            event_start = max(event_start, event_end)
            rows = np.flatnonzero(combined_time == cutoff)
            memory = state.memory[torch.as_tensor(mapped_users[rows], device=device, dtype=torch.int64)].float().cpu().numpy().astype(np.float16)
            matrix[rows] = memory
            matrix.flush()
            print(f"[graph] snapshot={pd.to_datetime(cutoff, unit='s').date()} rows={len(rows)} replayed={event_end}", flush=True)
    register_artifact(
        "lane3 temporal graph memory rows",
        path,
        "Leak-free user memory snapshot after every task origin.",
        "rel-stack-user-badge-lane3-memory-debug-v1" if debug else "rel-stack-user-badge-lane3-memory-full-v1",
    )
    return np.load(path, mmap_mode="r")
