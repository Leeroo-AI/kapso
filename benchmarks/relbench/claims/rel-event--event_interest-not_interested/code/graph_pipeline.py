from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD


@dataclass
class GraphBundle:
    user_ids: np.ndarray
    directed: sp.csr_matrix
    undirected: sp.csr_matrix
    normalized: sp.csr_matrix
    full_degree: np.ndarray
    out_degree: np.ndarray
    in_degree: np.ndarray
    und_degree: np.ndarray
    svd: np.ndarray
    deepwalk: np.ndarray
    component: np.ndarray
    component_size: np.ndarray
    community: np.ndarray
    community_size: np.ndarray
    pagerank: np.ndarray
    clustering: np.ndarray
    twohop_reach: np.ndarray


def map_ids(values: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    raw = np.asarray(values)
    missing = pd.isna(raw)
    clean = np.zeros(len(raw), dtype=np.int64)
    clean[~missing] = raw[~missing].astype(np.int64)
    positions = np.searchsorted(sorted_ids, clean)
    valid = (~missing) & (positions < len(sorted_ids))
    valid &= sorted_ids[np.minimum(positions, len(sorted_ids) - 1)] == clean
    positions[~valid] = -1
    return positions.astype(np.int32)


def build_seed_frame(task, cutoff_db, full_db) -> pd.DataFrame:
    parts = []
    cutoff_interest = cutoff_db.table_dict["event_interest"].df[
        ["primary_key", "timestamp", "user", "event", "invited"]
    ].set_index("primary_key", drop=False)
    full_interest = full_db.table_dict["event_interest"].df[
        ["primary_key", "timestamp", "user", "event", "invited"]
    ].set_index("primary_key", drop=False)
    for split in ["train", "val", "test"]:
        table = task.get_table(split).df.reset_index(drop=True)
        source = full_interest if split == "test" else cutoff_interest
        entity = source.loc[table["primary_key"].to_numpy()].reset_index(drop=True)
        expected = table["timestamp"].to_numpy(dtype="datetime64[ns]")
        observed = entity["timestamp"].to_numpy(dtype="datetime64[ns]")
        if not np.array_equal(expected, observed):
            raise RuntimeError(f"timestamp mapping failed for {split}")
        frame = entity[["timestamp", "user", "event", "invited", "primary_key"]].copy()
        frame["split"] = split
        frame["split_index"] = np.arange(len(frame), dtype=np.int32)
        if split == "test":
            frame["label"] = np.nan
        else:
            frame["label"] = table[task.target_col].to_numpy(dtype=np.float32)
        parts.append(frame)
    seeds = pd.concat(parts, ignore_index=True)
    seeds["row_id"] = np.arange(len(seeds), dtype=np.int32)
    if seeds[["user", "timestamp"]].isna().any().any():
        raise RuntimeError("mapped seed entities contain null user identifiers or timestamps")
    seeds["event_missing"] = seeds["event"].isna().astype(np.float32)
    seeds["event_key"] = seeds["event"].fillna(-(seeds["row_id"] + 1)).astype(np.int64)
    return seeds


def _deepwalk_embedding(adjacency: sp.csr_matrix, seed: int = 17) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    started = time.time()
    rng = np.random.default_rng(seed)
    n = adjacency.shape[0]
    walk_length = 20
    walks_per_node = 5
    context = 10
    epochs = 5
    walks = np.empty((n * walks_per_node, walk_length), dtype=np.int32)
    degrees = np.diff(adjacency.indptr)
    for walk_index in range(walks_per_node):
        current = rng.permutation(n).astype(np.int32)
        block = slice(walk_index * n, (walk_index + 1) * n)
        walks[block, 0] = current
        for step in range(1, walk_length):
            degree = degrees[current]
            movable = degree > 0
            following = current.copy()
            offsets = (rng.random(movable.sum()) * degree[movable]).astype(np.int64)
            following[movable] = adjacency.indices[
                adjacency.indptr[current[movable]] + offsets
            ]
            current = following
            walks[block, step] = current
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    center_embedding = torch.nn.Embedding(n, 32, device=device)
    context_embedding = torch.nn.Embedding(n, 32, device=device)
    bound = 0.5 / 32
    torch.nn.init.uniform_(center_embedding.weight, -bound, bound)
    torch.nn.init.zeros_(context_embedding.weight)
    optimizer = torch.optim.Adam(
        list(center_embedding.parameters()) + list(context_embedding.parameters()),
        lr=0.02,
    )
    flattened = walks.reshape(-1)
    total = len(flattened)
    row_base = np.arange(total, dtype=np.int64)
    batch_size = 131072
    for epoch in range(epochs):
        context_positions = np.empty(total, dtype=np.int64)
        for position in range(walk_length):
            indices = np.arange(position, total, walk_length, dtype=np.int64)
            choices = np.concatenate(
                [
                    np.arange(-min(context, position), 0, dtype=np.int64),
                    np.arange(1, min(context, walk_length - 1 - position) + 1, dtype=np.int64),
                ]
            )
            offsets = rng.choice(choices, size=len(indices))
            context_positions[indices] = indices + offsets
        for begin in range(0, total, batch_size):
            end = min(total, begin + batch_size)
            centers = torch.as_tensor(flattened[begin:end], dtype=torch.long, device=device)
            contexts = torch.as_tensor(
                flattened[context_positions[begin:end]], dtype=torch.long, device=device
            )
            negatives = torch.randint(0, n, (end - begin, 3), device=device)
            center_vector = center_embedding(centers)
            positive = (center_vector * context_embedding(contexts)).sum(dim=1)
            negative = torch.einsum("bd,bkd->bk", center_vector, context_embedding(negatives))
            loss = -functional.logsigmoid(positive).mean() - functional.logsigmoid(-negative).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    embedding = (
        0.5 * (center_embedding.weight.detach() + context_embedding.weight.detach())
    ).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding /= np.maximum(norms, 1e-6)
    print(f"[phase] deepwalk users={n} seconds={time.time() - started:.1f}")
    return embedding


def _pagerank(directed: sp.csr_matrix) -> np.ndarray:
    n = directed.shape[0]
    degree = np.asarray(directed.sum(axis=1)).ravel()
    inverse = np.zeros(n, dtype=np.float64)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    transition = sp.diags(inverse) @ directed
    rank = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(50):
        dangling = rank[degree == 0].sum() / n
        updated = 0.15 / n + 0.85 * (transition.T @ rank + dangling)
        if np.abs(updated - rank).sum() < 1e-10:
            rank = updated
            break
        rank = updated
    return rank.astype(np.float32)


def _register_artifact(cache_dir: Path, relative_path: str) -> None:
    registry = cache_dir / "artifacts.json"
    lock_path = cache_dir / ".artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        content_key = "lane3-social-graph-v1-seed17"
        if not any(record.get("content_key") == content_key for record in records):
            records.append(
                {
                    "name": "lane3 social graph structural bundle",
                    "path": relative_path,
                    "description": "Resolved friendship CSR, adjacency SVD, DeepWalk, components, PageRank and topology statistics",
                    "content_key": content_key,
                    "rebuild_hint": "Run the lane3 full candidate on the sanitized rel-event cache",
                }
            )
            temporary = registry.with_name(f"artifacts.{os.getpid()}.tmp")
            temporary.write_text(json.dumps(records, indent=2))
            temporary.replace(registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def build_friendship_graph(users: pd.DataFrame, friends: pd.DataFrame, cache_dir: Path, debug: bool) -> GraphBundle:
    started = time.time()
    root = cache_dir / "lane3_social_graph_v1_s17"
    root.mkdir(parents=True, exist_ok=True)
    adjacency_path = root / "adjacency.npz"
    directed_path = root / "directed.npz"
    base_path = root / "base.npz"
    static_path = root / "static_full.npz"
    user_ids = np.sort(users["user_id"].dropna().astype(np.int64).unique())
    n = len(user_ids)
    if adjacency_path.exists() and directed_path.exists() and base_path.exists():
        undirected = sp.load_npz(adjacency_path).tocsr()
        directed = sp.load_npz(directed_path).tocsr()
        base = np.load(base_path, allow_pickle=False)
        if not np.array_equal(base["user_ids"], user_ids):
            raise RuntimeError("cached graph user mapping does not match database")
        full_degree = base["full_degree"]
    else:
        full_counts = friends["user"].value_counts(sort=False)
        full_degree = np.zeros(n, dtype=np.float32)
        full_positions = map_ids(full_counts.index.to_numpy(), user_ids)
        valid_counts = full_positions >= 0
        full_degree[full_positions[valid_counts]] = full_counts.to_numpy(dtype=np.float32)[valid_counts]
        resolved = friends.loc[friends["friend"].notna(), ["user", "friend"]]
        source = map_ids(resolved["user"].to_numpy(), user_ids)
        target = map_ids(resolved["friend"].to_numpy(), user_ids)
        valid = (source >= 0) & (target >= 0) & (source != target)
        directed = sp.csr_matrix(
            (np.ones(valid.sum(), dtype=np.float32), (source[valid], target[valid])),
            shape=(n, n),
        )
        directed.sum_duplicates()
        directed.data[:] = 1.0
        directed.eliminate_zeros()
        undirected = directed.maximum(directed.T).tocsr()
        undirected.setdiag(0)
        undirected.eliminate_zeros()
        sp.save_npz(adjacency_path, undirected, compressed=True)
        sp.save_npz(directed_path, directed, compressed=True)
        np.savez_compressed(base_path, user_ids=user_ids, full_degree=full_degree)
    out_degree = np.diff(directed.indptr).astype(np.float32)
    in_degree = np.asarray(directed.sum(axis=0)).ravel().astype(np.float32)
    und_degree = np.diff(undirected.indptr).astype(np.float32)
    inverse_sqrt = np.zeros(n, dtype=np.float32)
    inverse_sqrt[und_degree > 0] = 1.0 / np.sqrt(und_degree[und_degree > 0])
    normalized = (sp.diags(inverse_sqrt) @ undirected @ sp.diags(inverse_sqrt)).tocsr()
    if static_path.exists() and not debug:
        static = np.load(static_path, allow_pickle=False)
        svd_embedding = static["svd"]
        deepwalk_embedding = static["deepwalk"]
        component = static["component"]
        component_size = static["component_size"]
        community = static["community"]
        community_size = static["community_size"]
        pagerank = static["pagerank"]
        clustering = static["clustering"]
        twohop_reach = static["twohop_reach"]
        source = "cache"
    else:
        svd_embedding = TruncatedSVD(
            n_components=32, algorithm="randomized", n_iter=7, random_state=17
        ).fit_transform(undirected).astype(np.float32)
        deepwalk_embedding = np.zeros((n, 32), dtype=np.float32) if debug else _deepwalk_embedding(undirected)
        _, component = connected_components(undirected, directed=False)
        component = component.astype(np.int32)
        component_counts = np.bincount(component)
        component_size = component_counts[component].astype(np.float32)
        community_input = np.concatenate([svd_embedding[:, :16], deepwalk_embedding[:, :16]], axis=1)
        community = MiniBatchKMeans(
            n_clusters=256, batch_size=4096, n_init=3, random_state=17
        ).fit_predict(community_input).astype(np.int32)
        community_counts = np.bincount(community, minlength=256)
        community_size = community_counts[community].astype(np.float32)
        pagerank = _pagerank(directed)
        squared = (undirected @ undirected).tocsr()
        squared.setdiag(0)
        squared.eliminate_zeros()
        twohop_reach = np.diff(squared.indptr).astype(np.float32)
        common = undirected.multiply(squared)
        common_sum = np.asarray(common.sum(axis=1)).ravel()
        denominator = und_degree * np.maximum(und_degree - 1, 1)
        clustering = np.divide(
            common_sum,
            denominator,
            out=np.zeros(n, dtype=np.float32),
            where=denominator > 0,
        ).astype(np.float32)
        source = "computed"
        if not debug:
            np.savez_compressed(
                static_path,
                svd=svd_embedding,
                deepwalk=deepwalk_embedding,
                component=component,
                component_size=component_size,
                community=community,
                community_size=community_size,
                pagerank=pagerank,
                clustering=clustering,
                twohop_reach=twohop_reach,
            )
            _register_artifact(cache_dir, str(root.relative_to(cache_dir)))
    print(
        f"[phase] friendship_graph source={source} users={n} directed_edges={directed.nnz} "
        f"undirected_edges={undirected.nnz} seconds={time.time() - started:.1f}"
    )
    return GraphBundle(
        user_ids=user_ids,
        directed=directed,
        undirected=undirected,
        normalized=normalized,
        full_degree=full_degree.astype(np.float32),
        out_degree=out_degree,
        in_degree=in_degree,
        und_degree=und_degree,
        svd=svd_embedding,
        deepwalk=deepwalk_embedding,
        component=component,
        component_size=component_size,
        community=community,
        community_size=community_size,
        pagerank=pagerank,
        clustering=clustering,
        twohop_reach=twohop_reach,
    )
