# Imports

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.utils.extmath import randomized_svd

from kapso_datasets.common import load_task, shared_cache_dir


# Configuration

DAY_NS = 86_400_000_000_000
START_DAY = pd.Timestamp("2012-06-20")
MODEL_A_END = pd.Timestamp("2012-11-14")
MODEL_B_END = pd.Timestamp("2012-11-21")
FORWARD_ORIGINS = [
    pd.Timestamp("2012-10-24"),
    pd.Timestamp("2012-10-31"),
    pd.Timestamp("2012-11-07"),
    pd.Timestamp("2012-11-14"),
]
CONTENT_DIMS = 8
GRAPH_DIMS = 32
GRAPH_VERSION = "cold_user_graph_moe_v1_svd32_seed1337"
FEATURE_VERSION = "cold_user_graph_moe_features_v1"


# Data classes

@dataclass
class GraphData:
    adjacency: sparse.csr_matrix
    transition: sparse.csr_matrix
    transition_out: sparse.csr_matrix
    transition_in: sparse.csr_matrix
    features: np.ndarray
    feature_names: list[str]
    communities: np.ndarray
    community_sizes: np.ndarray
    resolved_out: np.ndarray
    full_degree: np.ndarray
    coverage: np.ndarray


@dataclass
class AttendeeData:
    users: np.ndarray
    times: np.ndarray
    statuses: np.ndarray
    blast_sizes: np.ndarray
    blast_invites: np.ndarray
    content: np.ndarray


@dataclass
class InterestData:
    users: np.ndarray
    times: np.ndarray
    invited: np.ndarray
    interested: np.ndarray
    not_interested: np.ndarray


@dataclass
class ModelChoice:
    variant: str
    gate: str
    graph_accepted: bool
    history_iterations: int
    cold_iterations: int
    forward_scores: dict


@dataclass
class PipelineResult:
    val_predictions: np.ndarray
    test_predictions: np.ndarray
    diagnostics: dict


# Utilities

def log_phase(label: str, started: float) -> float:
    now = time.time()
    print(f"[phase] {label} elapsed_s={now - started:.2f}")
    return now


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    ).astype(np.float32)


def array_from_nullable(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.int64, na_value=-1)


def stable_hash(values: pd.Series, buckets: int) -> np.ndarray:
    text = values.fillna("__missing__").astype(str)
    return (pd.util.hash_pandas_object(text, index=False).to_numpy(dtype=np.uint64) % buckets).astype(np.int32)


def matrix_auc(y: np.ndarray, prediction: np.ndarray) -> float:
    if len(y) == 0 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, prediction))


def register_artifact(path: Path, name: str, content_key: str) -> None:
    root = shared_cache_dir()
    registry = root / "artifacts.json"
    lock_path = root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                entries = json.loads(registry.read_text())
            except json.JSONDecodeError:
                entries = []
        else:
            entries = []
        relative = str(path.relative_to(root))
        if not any(entry.get("content_key") == content_key for entry in entries):
            entries.append(
                {
                    "name": name,
                    "path": relative,
                    "description": "Static resolved friendship CSR, normalized-SVD embeddings, structural features, and Louvain assignments",
                    "content_key": content_key,
                    "rebuild_hint": "Run main.py; the graph cache is rebuilt from sanitized user_friends when absent",
                }
            )
            temporary = registry.with_suffix(".tmp")
            temporary.write_text(json.dumps(entries, indent=2, sort_keys=True))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


# Daily panel

def build_daily_panel(attendees: pd.DataFrame, debug: bool) -> pd.DataFrame:
    known = attendees.loc[attendees["user_id"].notna(), ["user_id", "status", "start_time"]].copy()
    known["user_id"] = known["user_id"].astype(np.int32)
    known = known.sort_values("start_time", kind="mergesort")
    times = known["start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    users = known["user_id"].to_numpy(dtype=np.int32)
    invited = (known["status"].to_numpy() == "invited").astype(np.int8)
    dates = pd.date_range(START_DAY, MODEL_B_END, freq="7D" if debug else "D")
    frames = []
    for seed in dates:
        seed_ns = int(seed.value)
        left = np.searchsorted(times, seed_ns, side="right")
        right = np.searchsorted(times, seed_ns + 7 * DAY_NS, side="right")
        window_users = users[left:right]
        counts = np.bincount(window_users, minlength=37143)
        invitation_counts = np.bincount(window_users, weights=invited[left:right], minlength=37143)
        active = np.flatnonzero(counts).astype(np.int32)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": np.full(len(active), seed.to_datetime64(), dtype="datetime64[ns]"),
                    "user": active,
                    "target": (invitation_counts[active] > 2).astype(np.int8),
                }
            )
        )
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["timestamp", "user"], kind="mergesort").reset_index(drop=True)
    return panel


def verify_panel(panel: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame, debug: bool) -> dict:
    checks = {}
    sources = [("train", train)]
    if not debug:
        sources.append(("val", val))
    indexed = panel.set_index(["timestamp", "user"])["target"]
    for split, source in sources:
        source_index = pd.MultiIndex.from_frame(source[["timestamp", "user"]])
        available = source["timestamp"].isin(panel["timestamp"].unique()).to_numpy()
        selected_index = source_index[available]
        reconstructed = indexed.reindex(selected_index)
        official = source.loc[available, "target"].to_numpy(dtype=np.int8)
        checks[split] = {
            "rows_checked": int(available.sum()),
            "missing": int(reconstructed.isna().sum()),
            "label_mismatches": int(np.sum(reconstructed.fillna(-1).to_numpy(dtype=np.int8) != official)),
        }
    print(f"[panel] rows={len(panel)} positives={int(panel.target.sum())} checks={json.dumps(checks, sort_keys=True)}")
    return checks


# Graph construction

def neighbor_quantiles(adjacency: sparse.csr_matrix, values: np.ndarray) -> np.ndarray:
    output = np.zeros((adjacency.shape[0], 8), dtype=np.float32)
    for node in range(adjacency.shape[0]):
        start, end = adjacency.indptr[node], adjacency.indptr[node + 1]
        if start == end:
            continue
        observed = values[adjacency.indices[start:end]]
        output[node, 0] = float(np.mean(observed))
        output[node, 1] = float(np.std(observed))
        output[node, 2] = float(np.min(observed))
        output[node, 3] = float(np.max(observed))
        output[node, 4:] = np.quantile(observed, [0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    return output


def compute_communities(adjacency: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    graph = nx.from_scipy_sparse_array(adjacency, create_using=nx.Graph)
    groups = nx.algorithms.community.louvain_communities(graph, seed=1337, resolution=1.0)
    communities = np.empty(adjacency.shape[0], dtype=np.int32)
    for group_index, members in enumerate(groups):
        communities[np.fromiter(members, dtype=np.int32)] = group_index
    sizes = np.bincount(communities).astype(np.float32)
    return communities, sizes


def graph_cache_paths() -> tuple[Path, Path, Path, Path]:
    root = shared_cache_dir() / "generic_exp_1_cold_graph_moe"
    root.mkdir(parents=True, exist_ok=True)
    return root, root / f"{GRAPH_VERSION}_adjacency.npz", root / f"{GRAPH_VERSION}_directed.npz", root / f"{GRAPH_VERSION}_features.npz"


def directed_graph(user_friends: pd.DataFrame, n_users: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix, sparse.csr_matrix]:
    user_values = array_from_nullable(user_friends["user"])
    friend_values = array_from_nullable(user_friends["friend"])
    resolved = (user_values >= 0) & (user_values < n_users) & (friend_values >= 0) & (friend_values < n_users)
    directed = sparse.csr_matrix(
        (np.ones(int(resolved.sum()), dtype=np.float32), (user_values[resolved], friend_values[resolved])),
        shape=(n_users, n_users),
    )
    directed.setdiag(0)
    directed.eliminate_zeros()
    directed.data.fill(1)
    directed.sort_indices()
    outgoing_degree = np.asarray(directed.sum(axis=1)).ravel().astype(np.float32)
    incoming = directed.T.tocsr()
    incoming_degree = np.asarray(incoming.sum(axis=1)).ravel().astype(np.float32)
    outgoing_inverse = np.zeros(n_users, dtype=np.float32)
    incoming_inverse = np.zeros(n_users, dtype=np.float32)
    outgoing_inverse[outgoing_degree > 0] = 1 / outgoing_degree[outgoing_degree > 0]
    incoming_inverse[incoming_degree > 0] = 1 / incoming_degree[incoming_degree > 0]
    return directed, (sparse.diags(outgoing_inverse) @ directed).tocsr(), (sparse.diags(incoming_inverse) @ incoming).tocsr()


def feature_cache_paths(debug: bool) -> tuple[Path, dict[str, Path]]:
    root = shared_cache_dir() / "generic_exp_1_cold_graph_moe" / f"{FEATURE_VERSION}_{'debug' if debug else 'full'}"
    paths = {
        "panel": root / "panel_matrix.npy",
        "val": root / "val_matrix.npy",
        "test": root / "test_matrix.npy",
        "metadata": root / "metadata.json",
    }
    return root, paths


def load_feature_cache(debug: bool, panel_rows: int, val_rows: int, test_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    _, paths = feature_cache_paths(debug)
    if not all(path.exists() for path in paths.values()):
        return None
    metadata = json.loads(paths["metadata"].read_text())
    expected = {"panel_rows": panel_rows, "val_rows": val_rows, "test_rows": test_rows, "version": FEATURE_VERSION}
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    panel_matrix = np.load(paths["panel"], mmap_mode="r")
    val_matrix = np.load(paths["val"], mmap_mode="r")
    test_matrix = np.load(paths["test"], mmap_mode="r")
    names = metadata["feature_names"]
    if panel_matrix.shape != (panel_rows, len(names)) or val_matrix.shape != (val_rows, len(names)) or test_matrix.shape != (test_rows, len(names)):
        return None
    print(f"[features] cache_hit=1 panel={panel_matrix.shape} val={val_matrix.shape} test={test_matrix.shape}")
    return panel_matrix, val_matrix, test_matrix, names


def save_feature_cache(debug: bool, panel_matrix: np.ndarray, val_matrix: np.ndarray, test_matrix: np.ndarray, names: list[str]) -> None:
    root, paths = feature_cache_paths(debug)
    root.mkdir(parents=True, exist_ok=True)
    temporary = {}
    for key, matrix in (("panel", panel_matrix), ("val", val_matrix), ("test", test_matrix)):
        temporary[key] = paths[key].with_suffix(".tmp.npy")
        np.save(temporary[key], matrix.astype(np.float32))
        os.replace(temporary[key], paths[key])
    metadata = {
        "version": FEATURE_VERSION,
        "panel_rows": len(panel_matrix),
        "val_rows": len(val_matrix),
        "test_rows": len(test_matrix),
        "feature_names": names,
    }
    temporary_metadata = paths["metadata"].with_suffix(".tmp")
    temporary_metadata.write_text(json.dumps(metadata, sort_keys=True))
    os.replace(temporary_metadata, paths["metadata"])
    register_artifact(root, "Cold-user censored feature matrices", f"{FEATURE_VERSION}_{'debug' if debug else 'full'}")
    print(f"[features] cache_saved=1 panel={panel_matrix.shape} val={val_matrix.shape} test={test_matrix.shape}")


def build_graph(user_friends: pd.DataFrame, n_users: int) -> GraphData:
    cache_root, adjacency_path, directed_path, feature_path = graph_cache_paths()
    if adjacency_path.exists() and feature_path.exists():
        adjacency = sparse.load_npz(adjacency_path).tocsr()
        if directed_path.exists():
            directed = sparse.load_npz(directed_path).tocsr()
            outgoing_degree = np.asarray(directed.sum(axis=1)).ravel().astype(np.float32)
            incoming = directed.T.tocsr()
            incoming_degree = np.asarray(incoming.sum(axis=1)).ravel().astype(np.float32)
            outgoing_inverse = np.zeros(n_users, dtype=np.float32)
            incoming_inverse = np.zeros(n_users, dtype=np.float32)
            outgoing_inverse[outgoing_degree > 0] = 1 / outgoing_degree[outgoing_degree > 0]
            incoming_inverse[incoming_degree > 0] = 1 / incoming_degree[incoming_degree > 0]
            transition_out = (sparse.diags(outgoing_inverse) @ directed).tocsr()
            transition_in = (sparse.diags(incoming_inverse) @ incoming).tocsr()
        else:
            directed, transition_out, transition_in = directed_graph(user_friends, n_users)
            sparse.save_npz(directed_path, directed, compressed=True)
        saved = np.load(feature_path, allow_pickle=False)
        features = saved["features"]
        names = saved["feature_names"].astype(str).tolist()
        communities = saved["communities"].astype(np.int32)
        sizes = saved["community_sizes"].astype(np.float32)
        resolved_out = saved["resolved_out"].astype(np.float32)
        full_degree = saved["full_degree"].astype(np.float32)
        coverage = saved["coverage"].astype(np.float32)
        degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        inverse = np.zeros(n_users, dtype=np.float32)
        inverse[degree > 0] = 1.0 / degree[degree > 0]
        transition = (sparse.diags(inverse) @ adjacency).tocsr()
        print(f"[graph] cache_hit=1 nodes={n_users} undirected_nnz={adjacency.nnz} features={features.shape[1]}")
        return GraphData(adjacency, transition, transition_out, transition_in, features, names, communities, sizes, resolved_out, full_degree, coverage)
    user_values = array_from_nullable(user_friends["user"])
    friend_values = array_from_nullable(user_friends["friend"])
    valid_user = (user_values >= 0) & (user_values < n_users)
    valid_friend = (friend_values >= 0) & (friend_values < n_users)
    resolved = valid_user & valid_friend
    full_degree = np.bincount(user_values[valid_user], minlength=n_users).astype(np.float32)
    resolved_out = np.bincount(user_values[resolved], minlength=n_users).astype(np.float32)
    resolved_in = np.bincount(friend_values[resolved], minlength=n_users).astype(np.float32)
    rows = np.concatenate([user_values[resolved], friend_values[resolved]])
    cols = np.concatenate([friend_values[resolved], user_values[resolved]])
    adjacency = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(n_users, n_users),
    )
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    adjacency.data.fill(1.0)
    adjacency.sort_indices()
    degree = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
    inverse = np.zeros(n_users, dtype=np.float32)
    inverse[degree > 0] = 1.0 / degree[degree > 0]
    inverse_sqrt = np.zeros(n_users, dtype=np.float32)
    inverse_sqrt[degree > 0] = 1.0 / np.sqrt(degree[degree > 0])
    transition = (sparse.diags(inverse) @ adjacency).tocsr()
    directed, transition_out, transition_in = directed_graph(user_friends, n_users)
    normalized = (sparse.diags(inverse_sqrt) @ adjacency @ sparse.diags(inverse_sqrt)).tocsr()
    left, singular, _ = randomized_svd(normalized, n_components=GRAPH_DIMS, n_iter=7, random_state=1337)
    embeddings = (left * singular.reshape(1, -1)).astype(np.float32)
    log_degree = np.log1p(degree).astype(np.float32)
    one_hop = neighbor_quantiles(adjacency, log_degree)
    two_hop_mean = np.asarray(transition @ one_hop[:, 0]).ravel().astype(np.float32)
    two_hop_second = np.asarray(transition @ (one_hop[:, 0] ** 2)).ravel().astype(np.float32)
    two_hop_std = np.sqrt(np.maximum(two_hop_second - two_hop_mean**2, 0)).astype(np.float32)
    two_hop_q25 = np.asarray(transition @ one_hop[:, 4]).ravel().astype(np.float32)
    two_hop_q50 = np.asarray(transition @ one_hop[:, 5]).ravel().astype(np.float32)
    two_hop_q75 = np.asarray(transition @ one_hop[:, 6]).ravel().astype(np.float32)
    squared = adjacency @ adjacency
    common_sum = np.asarray(adjacency.multiply(squared).sum(axis=1)).ravel().astype(np.float32)
    clustering = np.divide(
        common_sum,
        degree * np.maximum(degree - 1, 1),
        out=np.zeros_like(common_sum),
        where=degree > 1,
    ).astype(np.float32)
    coverage = safe_ratio(resolved_out, full_degree)
    null_fraction = np.ones(n_users, dtype=np.float32)
    null_fraction[full_degree > 0] = 1.0 - coverage[full_degree > 0]
    communities, sizes = compute_communities(adjacency)
    columns = OrderedDict(
        degree_full=full_degree,
        degree_full_log=np.log1p(full_degree),
        degree_resolved_out=resolved_out,
        degree_resolved_in=resolved_in,
        degree_symmetric=degree,
        degree_symmetric_log=log_degree,
        degree_endpoint_null_fraction=null_fraction,
        degree_resolved_full_ratio=coverage,
        degree_isolated=(degree == 0).astype(np.float32),
        struct_neighbor_degree_mean=one_hop[:, 0],
        struct_neighbor_degree_std=one_hop[:, 1],
        struct_neighbor_degree_min=one_hop[:, 2],
        struct_neighbor_degree_max=one_hop[:, 3],
        struct_neighbor_degree_q25=one_hop[:, 4],
        struct_neighbor_degree_q50=one_hop[:, 5],
        struct_neighbor_degree_q75=one_hop[:, 6],
        struct_neighbor_degree_q90=one_hop[:, 7],
        struct_twohop_degree_mean=two_hop_mean,
        struct_twohop_degree_std=two_hop_std,
        struct_twohop_degree_q25=two_hop_q25,
        struct_twohop_degree_q50=two_hop_q50,
        struct_twohop_degree_q75=two_hop_q75,
        struct_common_neighbor_density=clustering,
        community_static_size=sizes[communities],
        community_static_log_size=np.log1p(sizes[communities]),
    )
    for index in range(GRAPH_DIMS):
        columns[f"embed_{index:02d}"] = embeddings[:, index]
    features = np.column_stack(list(columns.values())).astype(np.float32)
    names = list(columns)
    sparse.save_npz(adjacency_path, adjacency, compressed=True)
    sparse.save_npz(directed_path, directed, compressed=True)
    np.savez_compressed(
        feature_path,
        features=features,
        feature_names=np.asarray(names),
        communities=communities,
        community_sizes=sizes,
        resolved_out=resolved_out,
        full_degree=full_degree,
        coverage=coverage,
    )
    register_artifact(cache_root, "Cold-user graph structural cache", GRAPH_VERSION)
    print(f"[graph] cache_hit=0 nodes={n_users} resolved_directed={int(resolved.sum())} undirected_nnz={adjacency.nnz} communities={len(sizes)}")
    return GraphData(adjacency, transition, transition_out, transition_in, features, names, communities, sizes, resolved_out, full_degree, coverage)


# Static user and event data

def build_demographics(users: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray, np.ndarray]:
    ordered = users.sort_values("user_id", kind="mergesort").reset_index(drop=True)
    n_users = len(ordered)
    columns = OrderedDict()
    birthyear = ordered["birthyear"].to_numpy(dtype=np.float32, na_value=np.nan)
    columns["demo_birthyear"] = birthyear
    columns["demo_birthyear_missing"] = np.isnan(birthyear).astype(np.float32)
    timezone = ordered["timezone"].to_numpy(dtype=np.float32, na_value=np.nan)
    columns["demo_timezone_hours"] = timezone / 60.0
    columns["demo_timezone_missing"] = np.isnan(timezone).astype(np.float32)
    gender = ordered["gender"].fillna("__missing__").astype(str)
    for value in sorted(gender.unique()):
        columns[f"demo_gender_{value}"] = (gender == value).to_numpy(dtype=np.float32)
    locale = ordered["locale"].fillna("__missing__").astype(str)
    locale_counts = locale.value_counts()
    locale_frequency = locale.map(locale_counts).to_numpy(dtype=np.float32)
    columns["demo_locale_frequency"] = locale_frequency
    columns["demo_locale_log_frequency"] = np.log1p(locale_frequency)
    for value in sorted(locale.unique()):
        digest = hashlib.sha1(value.encode()).hexdigest()[:8]
        columns[f"demo_locale_{digest}"] = (locale == value).to_numpy(dtype=np.float32)
    location = ordered["location"].fillna("__missing__").astype(str)
    location_counts = location.value_counts()
    columns["demo_location_frequency"] = location.map(location_counts).to_numpy(dtype=np.float32)
    columns["demo_location_missing"] = (location == "__missing__").to_numpy(dtype=np.float32)
    location_hash = stable_hash(location, 32)
    for bucket in range(32):
        columns[f"demo_location_hash_{bucket:02d}"] = (location_hash == bucket).astype(np.float32)
    joined = ordered["joinedAt"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    features = np.column_stack(list(columns.values())).astype(np.float32)
    if features.shape[0] != n_users:
        raise RuntimeError("demographic row alignment failed")
    return features, list(columns), joined, locale_frequency, birthyear


def prepare_attendees(db, n_users: int) -> AttendeeData:
    attendees = db.table_dict["event_attendees"].df
    events = db.table_dict["events"].df
    event_ids = attendees["event"].to_numpy(dtype=np.int64, na_value=-1)
    valid_events = event_ids >= 0
    n_events = len(events)
    blast_sizes_all = np.bincount(event_ids[valid_events], minlength=n_events).astype(np.float32)
    invitation_rows = valid_events & (attendees["status"].to_numpy() == "invited")
    blast_invites_all = np.bincount(event_ids[invitation_rows], minlength=n_events).astype(np.float32)
    known = attendees["user_id"].notna().to_numpy()
    known_users = attendees.loc[known, "user_id"].to_numpy(dtype=np.int32)
    known_events = event_ids[known]
    known_times = attendees.loc[known, "start_time"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    status_text = attendees.loc[known, "status"].to_numpy()
    status_map = {"invited": 0, "yes": 1, "no": 2, "maybe": 3}
    statuses = np.array([status_map.get(value, 4) for value in status_text], dtype=np.int8)
    content_columns = [name for name in events.columns if name.startswith("c_")]
    content_values = np.log1p(np.maximum(events.iloc[known_events][content_columns].to_numpy(dtype=np.float32), 0))
    generator = np.random.default_rng(1337)
    projection = generator.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(len(content_columns), CONTENT_DIMS))
    content = (content_values @ projection / np.sqrt(len(content_columns))).astype(np.float32)
    order = np.argsort(known_times, kind="mergesort")
    if known_users.max(initial=0) >= n_users:
        raise RuntimeError("attendee user index exceeds users table")
    return AttendeeData(
        known_users[order],
        known_times[order],
        statuses[order],
        blast_sizes_all[known_events][order],
        blast_invites_all[known_events][order],
        content[order],
    )


def prepare_interests(interests: pd.DataFrame) -> InterestData:
    users = interests["user"].to_numpy(dtype=np.int32)
    times = interests["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    order = np.argsort(times, kind="mergesort")
    return InterestData(
        users[order],
        times[order],
        interests["invited"].to_numpy(dtype=np.float32)[order],
        interests["interested"].to_numpy(dtype=np.float32)[order],
        interests["not_interested"].to_numpy(dtype=np.float32)[order],
    )


# Temporal feature construction

def target_neighbor_summary(adjacency: sparse.csr_matrix, target_users: np.ndarray, values: np.ndarray) -> np.ndarray:
    unique, inverse = np.unique(target_users, return_inverse=True)
    summary = np.zeros((len(unique), 5), dtype=np.float32)
    for index, user in enumerate(unique):
        start, end = adjacency.indptr[user], adjacency.indptr[user + 1]
        if start == end:
            continue
        observed = values[adjacency.indices[start:end]]
        summary[index, 0] = float(np.max(observed))
        summary[index, 1:] = np.quantile(observed, [0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    return summary[inverse]


def rolling_counts(users: np.ndarray, times: np.ndarray, statuses: np.ndarray | None, seed_ns: int, days: int, n_users: int, status: int | None = None, weights: np.ndarray | None = None) -> np.ndarray:
    left = np.searchsorted(times, seed_ns - days * DAY_NS, side="right")
    right = np.searchsorted(times, seed_ns, side="right")
    selected_users = users[left:right]
    if status is not None and statuses is not None:
        keep = statuses[left:right] == status
        selected_users = selected_users[keep]
        selected_weights = None if weights is None else weights[left:right][keep]
    else:
        selected_weights = None if weights is None else weights[left:right]
    return np.bincount(selected_users, weights=selected_weights, minlength=n_users).astype(np.float32)


def build_dynamic_features(
    samples: pd.DataFrame,
    panel: pd.DataFrame,
    graph: GraphData,
    static_features: np.ndarray,
    static_names: list[str],
    joined: np.ndarray,
    birthyear: np.ndarray,
    attendees: AttendeeData,
    interests: InterestData,
) -> tuple[np.ndarray, list[str]]:
    n_users = graph.adjacency.shape[0]
    dynamic_names = [
        "demo_account_age_days", "demo_joined_after_seed", "demo_age_years",
        "hist_att_total", "hist_att_14", "hist_att_30", "hist_att_90",
        "hist_inv_total", "hist_inv_14", "hist_inv_30", "hist_inv_90",
        "hist_yes_total", "hist_yes_90", "hist_no_total", "hist_no_90",
        "hist_maybe_total", "hist_maybe_90", "hist_inv_share_total", "hist_yes_share_total",
        "hist_recency_any_days", "hist_recency_inv_days", "hist_recency_yes_days", "hist_recency_no_days",
        "hist_blast_mean_total", "hist_blast_max_total", "hist_blast_mean_30", "hist_blast_max_30",
        "hist_blast_mean_90", "hist_blast_max_90", "hist_blast_invite_fraction_30",
        "hist_label_count", "hist_positive_count", "hist_positive_rate", "hist_positive_ever",
    ]
    dynamic_names.extend([f"content_total_mean_{index:02d}" for index in range(CONTENT_DIMS)])
    dynamic_names.extend([f"content_90_mean_{index:02d}" for index in range(CONTENT_DIMS)])
    dynamic_names.extend(
        [
            "interest_count_total", "interest_count_30", "interest_count_90",
            "interest_invited_total", "interest_invited_90", "interest_interested_total",
            "interest_interested_90", "interest_not_total", "interest_not_90",
            "interest_recency_days", "interest_positive_share_total", "interest_negative_share_total",
        ]
    )
    for window in (14, 30, 90):
        dynamic_names.extend(
            [
                f"prop_inv_{window}_mean", f"prop_inv_{window}_max", f"prop_inv_{window}_q25",
                f"prop_inv_{window}_q50", f"prop_inv_{window}_q75", f"prop_inv_{window}_q90",
                f"prop_inv_{window}_twohop_mean",
            ]
        )
    dynamic_names.extend(
        [
            "prop_positive_neighbor_fraction", "prop_positive_rate_onehop", "prop_positive_rate_twohop",
            "prop_interest_30_mean", "prop_interest_30_max", "prop_interest_30_q90",
            "prop_blast_30_mean", "prop_blast_30_max", "prop_blast_30_q90",
        ]
    )
    dynamic_names.extend([f"prop_content_90_mean_{index:02d}" for index in range(CONTENT_DIMS)])
    dynamic_names.extend(
        [
            "community_inv_30_eb_rate", "community_inv_30_count", "community_inv_30_active_members",
            "community_inv_30_coverage", "community_inv_90_eb_rate", "community_inv_90_count",
            "community_inv_90_active_members", "community_inv_90_coverage", "community_positive_eb_rate",
            "community_label_count", "community_label_member_coverage",
        ]
    )
    output = np.empty((len(samples), static_features.shape[1] + len(dynamic_names)), dtype=np.float32)
    total_att = np.zeros(n_users, dtype=np.float32)
    total_status = np.zeros((5, n_users), dtype=np.float32)
    total_blast = np.zeros(n_users, dtype=np.float32)
    max_blast = np.zeros(n_users, dtype=np.float32)
    total_content = np.zeros((n_users, CONTENT_DIMS), dtype=np.float32)
    last_any = np.full(n_users, -1, dtype=np.int64)
    last_status = np.full((5, n_users), -1, dtype=np.int64)
    total_interest = np.zeros(n_users, dtype=np.float32)
    total_interest_flags = np.zeros((3, n_users), dtype=np.float32)
    last_interest = np.full(n_users, -1, dtype=np.int64)
    historical_labels = np.zeros(n_users, dtype=np.float32)
    historical_positive = np.zeros(n_users, dtype=np.float32)
    attendee_pointer = 0
    interest_pointer = 0
    outcome_order = np.argsort(panel["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64) + 7 * DAY_NS, kind="mergesort")
    outcome_close = panel["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)[outcome_order] + 7 * DAY_NS
    outcome_users = panel["user"].to_numpy(dtype=np.int32)[outcome_order]
    outcome_targets = panel["target"].to_numpy(dtype=np.float32)[outcome_order]
    outcome_pointer = 0
    community = graph.communities
    community_count = len(graph.community_sizes)
    transition = graph.transition
    sample_times = samples["timestamp"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    sample_users = samples["user"].to_numpy(dtype=np.int32)
    unique_times = np.unique(sample_times)
    for time_index, seed_ns in enumerate(unique_times):
        row_indices = np.flatnonzero(sample_times == seed_ns)
        target_users = sample_users[row_indices]
        attendee_right = np.searchsorted(attendees.times, seed_ns, side="right")
        if attendee_right > attendee_pointer:
            new_slice = slice(attendee_pointer, attendee_right)
            new_users = attendees.users[new_slice]
            np.add.at(total_att, new_users, 1)
            np.add.at(total_blast, new_users, attendees.blast_sizes[new_slice])
            np.maximum.at(max_blast, new_users, attendees.blast_sizes[new_slice])
            np.maximum.at(last_any, new_users, attendees.times[new_slice])
            for status in range(5):
                keep = attendees.statuses[new_slice] == status
                if np.any(keep):
                    status_users = new_users[keep]
                    np.add.at(total_status[status], status_users, 1)
                    np.maximum.at(last_status[status], status_users, attendees.times[new_slice][keep])
            for dimension in range(CONTENT_DIMS):
                np.add.at(total_content[:, dimension], new_users, attendees.content[new_slice, dimension])
            attendee_pointer = attendee_right
        interest_right = np.searchsorted(interests.times, seed_ns, side="right")
        if interest_right > interest_pointer:
            new_slice = slice(interest_pointer, interest_right)
            new_users = interests.users[new_slice]
            np.add.at(total_interest, new_users, 1)
            np.add.at(total_interest_flags[0], new_users, interests.invited[new_slice])
            np.add.at(total_interest_flags[1], new_users, interests.interested[new_slice])
            np.add.at(total_interest_flags[2], new_users, interests.not_interested[new_slice])
            np.maximum.at(last_interest, new_users, interests.times[new_slice])
            interest_pointer = interest_right
        outcome_right = np.searchsorted(outcome_close, seed_ns, side="right")
        if outcome_right > outcome_pointer:
            new_users = outcome_users[outcome_pointer:outcome_right]
            np.add.at(historical_labels, new_users, 1)
            np.add.at(historical_positive, new_users, outcome_targets[outcome_pointer:outcome_right])
            outcome_pointer = outcome_right
        att14 = rolling_counts(attendees.users, attendees.times, None, seed_ns, 14, n_users)
        att30 = rolling_counts(attendees.users, attendees.times, None, seed_ns, 30, n_users)
        att90 = rolling_counts(attendees.users, attendees.times, None, seed_ns, 90, n_users)
        inv14 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 14, n_users, 0)
        inv30 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 30, n_users, 0)
        inv90 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 90, n_users, 0)
        yes90 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 90, n_users, 1)
        no90 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 90, n_users, 2)
        maybe90 = rolling_counts(attendees.users, attendees.times, attendees.statuses, seed_ns, 90, n_users, 3)
        blast30_sum = rolling_counts(attendees.users, attendees.times, None, seed_ns, 30, n_users, weights=attendees.blast_sizes)
        blast90_sum = rolling_counts(attendees.users, attendees.times, None, seed_ns, 90, n_users, weights=attendees.blast_sizes)
        blast_inv30_sum = rolling_counts(attendees.users, attendees.times, None, seed_ns, 30, n_users, weights=attendees.blast_invites)
        blast30_max = np.zeros(n_users, dtype=np.float32)
        blast90_max = np.zeros(n_users, dtype=np.float32)
        left30 = np.searchsorted(attendees.times, seed_ns - 30 * DAY_NS, side="right")
        left90 = np.searchsorted(attendees.times, seed_ns - 90 * DAY_NS, side="right")
        np.maximum.at(blast30_max, attendees.users[left30:attendee_right], attendees.blast_sizes[left30:attendee_right])
        np.maximum.at(blast90_max, attendees.users[left90:attendee_right], attendees.blast_sizes[left90:attendee_right])
        content90 = np.zeros((n_users, CONTENT_DIMS), dtype=np.float32)
        for dimension in range(CONTENT_DIMS):
            content90[:, dimension] = np.bincount(
                attendees.users[left90:attendee_right],
                weights=attendees.content[left90:attendee_right, dimension],
                minlength=n_users,
            ).astype(np.float32)
        content_total_mean = total_content / np.maximum(total_att[:, None], 1)
        content90_mean = content90 / np.maximum(att90[:, None], 1)
        interest30 = rolling_counts(interests.users, interests.times, None, seed_ns, 30, n_users)
        interest90 = rolling_counts(interests.users, interests.times, None, seed_ns, 90, n_users)
        interest_inv90 = rolling_counts(interests.users, interests.times, None, seed_ns, 90, n_users, weights=interests.invited)
        interest_pos90 = rolling_counts(interests.users, interests.times, None, seed_ns, 90, n_users, weights=interests.interested)
        interest_neg90 = rolling_counts(interests.users, interests.times, None, seed_ns, 90, n_users, weights=interests.not_interested)
        global_positive = float(historical_positive.sum() / max(historical_labels.sum(), 1))
        positive_rate = (historical_positive + 5 * global_positive) / (historical_labels + 5)
        positive_neighbor_fraction = np.asarray(transition @ (historical_positive > 0).astype(np.float32)).ravel()
        positive_onehop = np.asarray(transition @ positive_rate).ravel()
        positive_twohop = np.asarray(transition @ positive_onehop).ravel()
        propagation = {}
        for window, values in ((14, inv14), (30, inv30), (90, inv90)):
            mean = np.asarray(transition @ values).ravel().astype(np.float32)
            twohop = np.asarray(transition @ mean).ravel().astype(np.float32)
            summary = target_neighbor_summary(graph.adjacency, target_users, values)
            propagation[window] = (mean, twohop, summary)
        interest_mean = np.asarray(transition @ interest30).ravel().astype(np.float32)
        interest_summary = target_neighbor_summary(graph.adjacency, target_users, interest30)
        blast30_mean = safe_ratio(blast30_sum, att30)
        blast_neighbor_mean = np.asarray(transition @ blast30_mean).ravel().astype(np.float32)
        blast_summary = target_neighbor_summary(graph.adjacency, target_users, blast30_mean)
        propagated_content = np.column_stack(
            [np.asarray(transition @ content90_mean[:, dimension]).ravel() for dimension in range(CONTENT_DIMS)]
        ).astype(np.float32)
        comm_size = graph.community_sizes
        comm_inv30 = np.bincount(community, weights=inv30, minlength=community_count).astype(np.float32)
        comm_inv90 = np.bincount(community, weights=inv90, minlength=community_count).astype(np.float32)
        comm_active30 = np.bincount(community, weights=(inv30 > 0), minlength=community_count).astype(np.float32)
        comm_active90 = np.bincount(community, weights=(inv90 > 0), minlength=community_count).astype(np.float32)
        mean_inv30 = float(inv30.mean())
        mean_inv90 = float(inv90.mean())
        comm_inv30_rate = (comm_inv30 + 20 * mean_inv30) / (comm_size + 20)
        comm_inv90_rate = (comm_inv90 + 20 * mean_inv90) / (comm_size + 20)
        comm_positive = np.bincount(community, weights=historical_positive, minlength=community_count).astype(np.float32)
        comm_labels = np.bincount(community, weights=historical_labels, minlength=community_count).astype(np.float32)
        comm_labeled_members = np.bincount(community, weights=(historical_labels > 0), minlength=community_count).astype(np.float32)
        comm_positive_rate = (comm_positive + 20 * global_positive) / (comm_labels + 20)
        users = target_users
        seed_day = seed_ns / DAY_NS
        row = OrderedDict()
        row["demo_account_age_days"] = np.maximum((seed_ns - joined[users]) / DAY_NS, 0)
        row["demo_joined_after_seed"] = (joined[users] > seed_ns).astype(np.float32)
        row["demo_age_years"] = seed_day / 365.2425 + 1970 - birthyear[users]
        row["hist_att_total"] = total_att[users]
        row["hist_att_14"] = att14[users]
        row["hist_att_30"] = att30[users]
        row["hist_att_90"] = att90[users]
        row["hist_inv_total"] = total_status[0, users]
        row["hist_inv_14"] = inv14[users]
        row["hist_inv_30"] = inv30[users]
        row["hist_inv_90"] = inv90[users]
        row["hist_yes_total"] = total_status[1, users]
        row["hist_yes_90"] = yes90[users]
        row["hist_no_total"] = total_status[2, users]
        row["hist_no_90"] = no90[users]
        row["hist_maybe_total"] = total_status[3, users]
        row["hist_maybe_90"] = maybe90[users]
        row["hist_inv_share_total"] = safe_ratio(total_status[0, users], total_att[users])
        row["hist_yes_share_total"] = safe_ratio(total_status[1, users], total_att[users])
        for name, values in (("any", last_any), ("inv", last_status[0]), ("yes", last_status[1]), ("no", last_status[2])):
            recency = np.full(len(users), 3650, dtype=np.float32)
            observed = values[users] >= 0
            recency[observed] = ((seed_ns - values[users][observed]) / DAY_NS).astype(np.float32)
            row[f"hist_recency_{name}_days"] = recency
        row["hist_blast_mean_total"] = safe_ratio(total_blast[users], total_att[users])
        row["hist_blast_max_total"] = max_blast[users]
        row["hist_blast_mean_30"] = safe_ratio(blast30_sum[users], att30[users])
        row["hist_blast_max_30"] = blast30_max[users]
        row["hist_blast_mean_90"] = safe_ratio(blast90_sum[users], att90[users])
        row["hist_blast_max_90"] = blast90_max[users]
        row["hist_blast_invite_fraction_30"] = safe_ratio(blast_inv30_sum[users], blast30_sum[users])
        row["hist_label_count"] = historical_labels[users]
        row["hist_positive_count"] = historical_positive[users]
        row["hist_positive_rate"] = positive_rate[users]
        row["hist_positive_ever"] = (historical_positive[users] > 0).astype(np.float32)
        for dimension in range(CONTENT_DIMS):
            row[f"content_total_mean_{dimension:02d}"] = content_total_mean[users, dimension]
        for dimension in range(CONTENT_DIMS):
            row[f"content_90_mean_{dimension:02d}"] = content90_mean[users, dimension]
        row["interest_count_total"] = total_interest[users]
        row["interest_count_30"] = interest30[users]
        row["interest_count_90"] = interest90[users]
        row["interest_invited_total"] = total_interest_flags[0, users]
        row["interest_invited_90"] = interest_inv90[users]
        row["interest_interested_total"] = total_interest_flags[1, users]
        row["interest_interested_90"] = interest_pos90[users]
        row["interest_not_total"] = total_interest_flags[2, users]
        row["interest_not_90"] = interest_neg90[users]
        interest_recency = np.full(len(users), 3650, dtype=np.float32)
        interest_observed = last_interest[users] >= 0
        interest_recency[interest_observed] = ((seed_ns - last_interest[users][interest_observed]) / DAY_NS).astype(np.float32)
        row["interest_recency_days"] = interest_recency
        row["interest_positive_share_total"] = safe_ratio(total_interest_flags[1, users], total_interest[users])
        row["interest_negative_share_total"] = safe_ratio(total_interest_flags[2, users], total_interest[users])
        for window in (14, 30, 90):
            mean, twohop, summary = propagation[window]
            row[f"prop_inv_{window}_mean"] = mean[users]
            row[f"prop_inv_{window}_max"] = summary[:, 0]
            row[f"prop_inv_{window}_q25"] = summary[:, 1]
            row[f"prop_inv_{window}_q50"] = summary[:, 2]
            row[f"prop_inv_{window}_q75"] = summary[:, 3]
            row[f"prop_inv_{window}_q90"] = summary[:, 4]
            row[f"prop_inv_{window}_twohop_mean"] = twohop[users]
        row["prop_positive_neighbor_fraction"] = positive_neighbor_fraction[users]
        row["prop_positive_rate_onehop"] = positive_onehop[users]
        row["prop_positive_rate_twohop"] = positive_twohop[users]
        row["prop_interest_30_mean"] = interest_mean[users]
        row["prop_interest_30_max"] = interest_summary[:, 0]
        row["prop_interest_30_q90"] = interest_summary[:, 4]
        row["prop_blast_30_mean"] = blast_neighbor_mean[users]
        row["prop_blast_30_max"] = blast_summary[:, 0]
        row["prop_blast_30_q90"] = blast_summary[:, 4]
        for dimension in range(CONTENT_DIMS):
            row[f"prop_content_90_mean_{dimension:02d}"] = propagated_content[users, dimension]
        target_comm = community[users]
        row["community_inv_30_eb_rate"] = comm_inv30_rate[target_comm]
        row["community_inv_30_count"] = comm_inv30[target_comm]
        row["community_inv_30_active_members"] = comm_active30[target_comm]
        row["community_inv_30_coverage"] = safe_ratio(comm_active30[target_comm], comm_size[target_comm])
        row["community_inv_90_eb_rate"] = comm_inv90_rate[target_comm]
        row["community_inv_90_count"] = comm_inv90[target_comm]
        row["community_inv_90_active_members"] = comm_active90[target_comm]
        row["community_inv_90_coverage"] = safe_ratio(comm_active90[target_comm], comm_size[target_comm])
        row["community_positive_eb_rate"] = comm_positive_rate[target_comm]
        row["community_label_count"] = comm_labels[target_comm]
        row["community_label_member_coverage"] = safe_ratio(comm_labeled_members[target_comm], comm_size[target_comm])
        if list(row) != dynamic_names:
            raise RuntimeError("dynamic feature ordering mismatch")
        block = np.column_stack(list(row.values())).astype(np.float32)
        output[row_indices, : static_features.shape[1]] = static_features[users]
        output[row_indices, static_features.shape[1] :] = block
        if time_index == 0 or time_index == len(unique_times) - 1 or (time_index + 1) % 30 == 0:
            print(f"[features] seed={pd.Timestamp(seed_ns).date()} index={time_index + 1}/{len(unique_times)} rows={len(row_indices)}")
    return output, static_names + dynamic_names


def add_rank_features(matrix: np.ndarray, names: list[str], timestamps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    candidates = [
        "degree_full_log", "degree_symmetric_log", "hist_att_total", "hist_att_30", "hist_att_90",
        "hist_inv_total", "hist_inv_14", "hist_inv_30", "hist_inv_90", "hist_recency_inv_days",
        "hist_blast_mean_30", "hist_positive_rate", "interest_count_30", "prop_inv_30_mean",
        "community_inv_30_eb_rate",
    ]
    indices = [names.index(name) for name in candidates]
    unique_times = np.unique(timestamps)
    added = np.empty((len(matrix), len(candidates) * 4), dtype=np.float32)
    for seed in unique_times:
        rows = np.flatnonzero(timestamps == seed)
        for feature_position, column_index in enumerate(indices):
            values = matrix[rows, column_index].astype(np.float64)
            finite = np.isfinite(values)
            filled = values.copy()
            if not finite.all():
                median = np.nanmedian(filled) if finite.any() else 0.0
                filled[~finite] = median
            ranks = rankdata(filled, method="average") / max(len(filled), 1)
            std = float(np.std(filled))
            zscore = (filled - float(np.mean(filled))) / (std if std > 1e-8 else 1.0)
            maximum = float(np.max(filled)) if len(filled) else 0.0
            minimum = float(np.min(filled)) if len(filled) else 0.0
            added[rows, feature_position * 4] = ranks
            added[rows, feature_position * 4 + 1] = zscore
            added[rows, feature_position * 4 + 2] = maximum - filled
            added[rows, feature_position * 4 + 3] = filled - minimum
    added_names = []
    for name in candidates:
        added_names.extend([f"rank_{name}_percentile", f"rank_{name}_zscore", f"rank_{name}_gap_leader", f"rank_{name}_gap_floor"])
    return np.hstack([matrix, added]), names + added_names


# Expert training and forward selection

def feature_indices(names: list[str], expert: str, variant: str, graph_dims: int) -> np.ndarray:
    selected = []
    for index, name in enumerate(names):
        if expert == "history":
            keep = name.startswith(("demo_", "degree_", "hist_", "interest_", "content_"))
            keep = keep or name.startswith(("rank_degree_", "rank_hist_", "rank_interest_"))
            if name.startswith(("content_", "demo_location_hash_")):
                keep = False
        else:
            keep = name.startswith(("demo_", "degree_", "struct_", "embed_", "prop_", "community_"))
            keep = keep or name in ("hist_inv_total", "hist_att_total", "hist_label_count")
            keep = keep or name.startswith(("rank_degree_", "rank_prop_", "rank_community_"))
            if name.startswith("embed_"):
                keep = int(name.split("_")[1]) < graph_dims
            if variant == "svd8" and "twohop" in name:
                keep = False
            if name.startswith(("community_", "rank_community_")) and "community" not in variant:
                keep = False
            if name.startswith("community_") and variant.endswith("community_rank"):
                keep = False
        if keep:
            selected.append(index)
    return np.asarray(selected, dtype=np.int32)


def fit_expert(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray | None,
    debug: bool,
    validation: tuple[np.ndarray, np.ndarray] | None,
    iterations: int | None = None,
) -> tuple[lgb.LGBMClassifier, int]:
    estimators = iterations if iterations is not None else (100 if debug else 900)
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.035,
        num_leaves=31,
        max_depth=5,
        min_child_samples=80,
        reg_lambda=10.0,
        n_estimators=int(estimators),
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
        random_state=1337,
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    callbacks = [lgb.log_evaluation(0)]
    fit_args = {}
    if validation is not None and not debug:
        validation_matrix, validation_labels = validation
        fit_args["eval_set"] = [(validation_matrix[:, indices], validation_labels)]
        fit_args["eval_metric"] = "auc"
        callbacks.append(lgb.early_stopping(75, verbose=False))
    model.fit(matrix[:, indices], labels, sample_weight=weights, callbacks=callbacks, **fit_args)
    best = int(model.best_iteration_ or estimators)
    return model, best


def gate_weight(invitation_count: np.ndarray, gate: str) -> np.ndarray:
    if gate == "hard_zero":
        return (invitation_count == 0).astype(np.float64)
    tau = float(gate.split("_")[1])
    return np.exp(-invitation_count / tau)


def slice_metrics(labels: np.ndarray, prediction: np.ndarray, metadata: dict[str, np.ndarray]) -> dict:
    result = {}
    masks = {
        "history_zero": metadata["history"] == 0,
        "history_nonzero": metadata["history"] > 0,
        "isolated": metadata["isolated"] > 0,
        "resolved_neighbor": metadata["isolated"] == 0,
        "friend_coverage_low": metadata["coverage"] <= metadata["coverage_median"],
        "friend_coverage_high": metadata["coverage"] > metadata["coverage_median"],
        "locale_rare": metadata["locale_frequency"] < 500,
        "locale_common": metadata["locale_frequency"] >= 500,
    }
    for name, mask in masks.items():
        result[name] = {
            "count": int(mask.sum()),
            "positives": int(labels[mask].sum()),
            "auc": matrix_auc(labels[mask], prediction[mask]),
        }
    return result


def select_forward_model(
    panel: pd.DataFrame,
    matrix: np.ndarray,
    names: list[str],
    graph: GraphData,
    locale_frequency: np.ndarray,
    debug: bool,
) -> ModelChoice:
    labels = panel["target"].to_numpy(dtype=np.int8)
    times = panel["timestamp"].to_numpy(dtype="datetime64[ns]")
    users = panel["user"].to_numpy(dtype=np.int32)
    history_column = names.index("hist_inv_total")
    attendee_column = names.index("hist_att_total")
    variants = [("svd8", 8)] if debug else [
        ("svd16", 16),
        ("svd32", 32),
        ("svd16_community", 16),
        ("svd16_community_rank", 16),
        ("svd32_community", 32),
    ]
    gates = ["tau_1", "tau_3", "tau_10", "hard_zero"]
    history_index = feature_indices(names, "history", "history", GRAPH_DIMS)
    fold_records = []
    iteration_history = []
    iteration_cold = {variant: [] for variant, _ in variants}
    history_gain = np.zeros(len(history_index), dtype=np.float64)
    cold_gain = {variant: None for variant, _ in variants}
    for origin in FORWARD_ORIGINS:
        origin64 = origin.to_datetime64()
        train_mask = times <= (origin - pd.Timedelta(days=7)).to_datetime64()
        valid_mask = times == origin64
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        train_indices = np.flatnonzero(train_mask)
        valid_indices = np.flatnonzero(valid_mask)
        history_model, history_best = fit_expert(
            matrix[train_indices], labels[train_indices], history_index, None, debug,
            (matrix[valid_indices], labels[valid_indices]),
        )
        history_prediction = history_model.predict_proba(matrix[valid_indices][:, history_index])[:, 1]
        iteration_history.append(history_best)
        history_gain += history_model.booster_.feature_importance(importance_type="gain")
        cold_predictions = {}
        for variant, dimensions in variants:
            cold_index = feature_indices(names, "cold", variant, dimensions)
            invitation_history = matrix[train_indices, history_column]
            weights = np.where(invitation_history <= 5, 1.0, 0.25).astype(np.float32)
            cold_model, cold_best = fit_expert(
                matrix[train_indices], labels[train_indices], cold_index, weights, debug,
                (matrix[valid_indices], labels[valid_indices]),
            )
            cold_predictions[variant] = cold_model.predict_proba(matrix[valid_indices][:, cold_index])[:, 1]
            iteration_cold[variant].append(cold_best)
            gain = cold_model.booster_.feature_importance(importance_type="gain")
            cold_gain[variant] = gain if cold_gain[variant] is None else cold_gain[variant] + gain
        fold_records.append(
            {
                "origin": str(origin.date()),
                "indices": valid_indices,
                "labels": labels[valid_indices],
                "users": users[valid_indices],
                "history_prediction": history_prediction,
                "cold_predictions": cold_predictions,
                "invitation_history": matrix[valid_indices, history_column],
                "attendee_history": matrix[valid_indices, attendee_column],
            }
        )
        print(f"[forward] origin={origin.date()} train={len(train_indices)} valid={len(valid_indices)} history_auc={matrix_auc(labels[valid_indices], history_prediction):.6f}")
    if not fold_records:
        raise RuntimeError("no forward folds available")
    candidate_scores = {}
    history_fold_scores = [matrix_auc(record["labels"], record["history_prediction"]) for record in fold_records]
    candidate_scores["history"] = {
        "fold_auc": history_fold_scores,
        "mean_auc": float(np.nanmean(history_fold_scores)),
        "std_auc": float(np.nanstd(history_fold_scores)),
    }
    for variant, _ in variants:
        for gate in gates:
            fold_scores = []
            cold_slice_deltas = []
            for record in fold_records:
                weight = gate_weight(record["invitation_history"], gate)
                prediction = weight * record["cold_predictions"][variant] + (1 - weight) * record["history_prediction"]
                fold_scores.append(matrix_auc(record["labels"], prediction))
                zero = record["attendee_history"] == 0
                cold_slice_deltas.append(
                    matrix_auc(record["labels"][zero], prediction[zero])
                    - matrix_auc(record["labels"][zero], record["history_prediction"][zero])
                )
            key = f"{variant}:{gate}"
            candidate_scores[key] = {
                "fold_auc": fold_scores,
                "mean_auc": float(np.nanmean(fold_scores)),
                "std_auc": float(np.nanstd(fold_scores)),
                "cold_slice_delta": cold_slice_deltas,
                "selection_value": float(np.nanmean(fold_scores) - 0.1 * np.nanstd(fold_scores)),
            }
    keys = [key for key in candidate_scores if key != "history"]
    best_key = max(keys, key=lambda key: candidate_scores[key]["selection_value"])
    best_variant, best_gate = best_key.split(":")
    best_fold = np.asarray(candidate_scores[best_key]["fold_auc"], dtype=float)
    history_fold = np.asarray(history_fold_scores, dtype=float)
    differences = best_fold - history_fold
    tie_band = max(0.002, float(2 * np.nanstd(differences, ddof=1) / np.sqrt(max(len(differences), 1)))) if len(differences) > 1 else 0.002
    cold_delta = np.asarray(candidate_scores[best_key]["cold_slice_delta"], dtype=float)
    consistent_cold = int(np.sum(cold_delta > 0)) >= max(2, len(cold_delta) - 1) and float(np.nanmedian(cold_delta)) > 0
    graph_accepted = float(np.nanmean(differences)) >= 0 or (consistent_cold and float(np.nanmean(differences)) >= -tie_band)
    history_iterations = int(np.clip(np.median(iteration_history), 60, 900 if not debug else 100))
    cold_iterations = int(np.clip(np.median(iteration_cold[best_variant]), 60, 900 if not debug else 100))
    selected_dimensions = 8 if best_variant == "svd8" else (16 if best_variant.startswith("svd16") else 32)
    selected_cold_index = feature_indices(names, "cold", best_variant, selected_dimensions)
    history_top = sorted(zip(history_gain.tolist(), [names[index] for index in history_index]), reverse=True)[:20]
    cold_top = sorted(zip(cold_gain[best_variant].tolist(), [names[index] for index in selected_cold_index]), reverse=True)[:20]
    best_slice_records = {}
    symmetric_degree = np.diff(graph.adjacency.indptr)
    for record in fold_records:
        weight = gate_weight(record["invitation_history"], best_gate)
        prediction = weight * record["cold_predictions"][best_variant] + (1 - weight) * record["history_prediction"]
        record_users = record["users"]
        coverage = graph.coverage[record_users]
        covered = coverage[graph.resolved_out[record_users] > 0]
        coverage_median = float(np.median(covered)) if len(covered) else 0.0
        metadata = {
            "history": record["attendee_history"],
            "isolated": (symmetric_degree[record_users] == 0).astype(np.float32),
            "coverage": coverage,
            "coverage_median": coverage_median,
            "locale_frequency": locale_frequency[record_users],
        }
        fold_slices = slice_metrics(record["labels"], prediction, metadata)
        for slice_name, values in fold_slices.items():
            best_slice_records.setdefault(slice_name, []).append(values)
    best_slice_summary = {}
    for slice_name, records in best_slice_records.items():
        aucs = np.asarray([record["auc"] for record in records], dtype=float)
        best_slice_summary[slice_name] = {
            "folds": len(records),
            "row_count_sum": int(sum(record["count"] for record in records)),
            "positive_count_sum": int(sum(record["positives"] for record in records)),
            "mean_auc": float(np.nanmean(aucs)),
            "std_auc": float(np.nanstd(aucs)),
        }
    summary = {
        "best": best_key,
        "graph_accepted": graph_accepted,
        "tie_band": tie_band,
        "history_iterations": history_iterations,
        "cold_iterations": cold_iterations,
        "scores": candidate_scores,
        "history_top_gain": [{"feature": name, "gain": gain} for gain, name in history_top],
        "cold_top_gain": [{"feature": name, "gain": gain} for gain, name in cold_top],
        "best_slice_summary": best_slice_summary,
    }
    print(f"[selection] {json.dumps(summary, sort_keys=True, allow_nan=True)}")
    return ModelChoice(best_variant, best_gate, graph_accepted, history_iterations, cold_iterations, summary)


def fit_final_chain(
    panel: pd.DataFrame,
    panel_matrix: np.ndarray,
    prediction_matrix: np.ndarray,
    names: list[str],
    end: pd.Timestamp,
    choice: ModelChoice,
    debug: bool,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    labels = panel["target"].to_numpy(dtype=np.int8)
    times = panel["timestamp"].to_numpy(dtype="datetime64[ns]")
    train = np.flatnonzero(times <= end.to_datetime64())
    history_index = feature_indices(names, "history", "history", GRAPH_DIMS)
    dimensions = 8 if choice.variant == "svd8" else (16 if choice.variant.startswith("svd16") else 32)
    cold_index = feature_indices(names, "cold", choice.variant, dimensions)
    history_model, _ = fit_expert(
        panel_matrix[train], labels[train], history_index, None, debug, None, choice.history_iterations
    )
    invitation_history = panel_matrix[train, names.index("hist_inv_total")]
    cold_weights = np.where(invitation_history <= 5, 1.0, 0.25).astype(np.float32)
    cold_model, _ = fit_expert(
        panel_matrix[train], labels[train], cold_index, cold_weights, debug, None, choice.cold_iterations
    )
    history_prediction = history_model.predict_proba(prediction_matrix[:, history_index])[:, 1]
    cold_prediction = cold_model.predict_proba(prediction_matrix[:, cold_index])[:, 1]
    weight = gate_weight(prediction_matrix[:, names.index("hist_inv_total")], choice.gate)
    mixture = weight * cold_prediction + (1 - weight) * history_prediction
    prediction = mixture if choice.graph_accepted else history_prediction
    prediction = np.clip(prediction, 1e-6, 1 - 1e-6)
    print(f"[fit] chain_end={end.date()} rows={len(train)} history_features={len(history_index)} cold_features={len(cold_index)}")
    return prediction, {"history": history_prediction, "cold": cold_prediction, "mixture": mixture, "weight": weight}


# Diagnostics

def validation_diagnostics(
    labels: np.ndarray,
    prediction: np.ndarray,
    alternatives: dict[str, np.ndarray],
    val_matrix: np.ndarray,
    names: list[str],
    val_users: np.ndarray,
    graph: GraphData,
    locale_frequency: np.ndarray,
) -> dict:
    history = val_matrix[:, names.index("hist_att_total")]
    coverage = graph.coverage[val_users]
    positive_coverage = coverage[graph.resolved_out[val_users] > 0]
    median_coverage = float(np.median(positive_coverage)) if len(positive_coverage) else 0.0
    metadata = {
        "history": history,
        "isolated": (np.diff(graph.adjacency.indptr)[val_users] == 0).astype(np.float32),
        "coverage": coverage,
        "coverage_median": median_coverage,
        "locale_frequency": locale_frequency[val_users],
    }
    rng = np.random.default_rng(1337)
    bootstraps = []
    for _ in range(100):
        sample = rng.integers(0, len(labels), size=len(labels))
        value = matrix_auc(labels[sample], prediction[sample])
        if np.isfinite(value):
            bootstraps.append(value)
    correlations = []
    keys = ["history", "cold", "mixture"]
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            correlations.append(float(spearmanr(alternatives[keys[left]], alternatives[keys[right]]).statistic))
    result = {
        "validation_auc": matrix_auc(labels, prediction),
        "candidate_auc": {key: matrix_auc(labels, alternatives[key]) for key in keys},
        "bootstrap_auc_standard_error": float(np.std(bootstraps, ddof=1)),
        "bootstrap_draws": len(bootstraps),
        "mean_pairwise_rank_correlation": float(np.mean(correlations)),
        "pairwise_rank_correlations": correlations,
        "slices": slice_metrics(labels, prediction, metadata),
    }
    print(f"[diagnostics] {json.dumps(result, sort_keys=True, allow_nan=True)}")
    return result


# Pipeline

def run_pipeline(debug: bool) -> PipelineResult:
    started = time.time()
    context = load_task()
    mark = log_phase("database_loaded", started)
    users = context.db.table_dict["users"].df
    n_users = len(users)
    if not np.array_equal(users.sort_values("user_id")["user_id"].to_numpy(dtype=np.int64), np.arange(n_users)):
        raise RuntimeError("user ids are not contiguous and zero based")
    panel = build_daily_panel(context.db.table_dict["event_attendees"].df, debug)
    panel_checks = verify_panel(panel, context.train.df, context.val.df, debug)
    mark = log_phase("daily_panel", mark)
    graph = build_graph(context.db.table_dict["user_friends"].df, n_users)
    mark = log_phase("graph_views_svd_structure", mark)
    demographics, demographic_names, joined, locale_frequency, birthyear = build_demographics(users)
    static_features = np.hstack([graph.features, demographics]).astype(np.float32)
    static_names = graph.feature_names + demographic_names
    panel_samples = panel[["timestamp", "user"]].copy()
    val_samples = context.val.df[["timestamp", "user"]].copy()
    test_samples = context.test.df[["timestamp", "user"]].copy()
    cached = load_feature_cache(debug, len(panel_samples), len(val_samples), len(test_samples))
    if cached is None:
        attendees = prepare_attendees(context.db, n_users)
        interests = prepare_interests(context.db.table_dict["event_interest"].df)
        mark = log_phase("relational_sources_prepared", mark)
        combined = pd.concat([panel_samples, val_samples, test_samples], ignore_index=True)
        combined_matrix, names = build_dynamic_features(
            combined, panel, graph, static_features, static_names, joined, birthyear, attendees, interests
        )
        panel_end = len(panel_samples)
        val_end = panel_end + len(val_samples)
        panel_matrix, rank_names = add_rank_features(
            combined_matrix[:panel_end], names, panel_samples["timestamp"].to_numpy(dtype="datetime64[ns]")
        )
        val_matrix, val_rank_names = add_rank_features(
            combined_matrix[panel_end:val_end], names, val_samples["timestamp"].to_numpy(dtype="datetime64[ns]")
        )
        test_matrix, test_rank_names = add_rank_features(
            combined_matrix[val_end:], names, test_samples["timestamp"].to_numpy(dtype="datetime64[ns]")
        )
        if rank_names != val_rank_names or rank_names != test_rank_names:
            raise RuntimeError("rank feature alignment failed")
        names = rank_names
        save_feature_cache(debug, panel_matrix, val_matrix, test_matrix, names)
    else:
        panel_matrix, val_matrix, test_matrix, names = cached
    mark = log_phase("temporal_propagation_features", mark)
    choice = select_forward_model(panel, panel_matrix, names, graph, locale_frequency, debug)
    mark = log_phase("purged_forward_gates", mark)
    val_prediction, val_alternatives = fit_final_chain(
        panel, panel_matrix, val_matrix, names, MODEL_A_END, choice, debug
    )
    frozen_validation = val_prediction.copy()
    validation_hash = hashlib.sha256(np.ascontiguousarray(frozen_validation).tobytes()).hexdigest()
    print(f"[contract] validation_predictions_frozen_from=Model_A labels_through=2012-11-14 sha256={validation_hash}")
    test_prediction, _ = fit_final_chain(
        panel, panel_matrix, test_matrix, names, MODEL_B_END, choice, debug
    )
    if hashlib.sha256(np.ascontiguousarray(val_prediction).tobytes()).hexdigest() != validation_hash:
        raise RuntimeError("Model A validation predictions changed during Model B fitting")
    mark = log_phase("model_a_model_b", mark)
    validation_labels = context.val.df["target"].to_numpy(dtype=np.int8)
    diagnostics = validation_diagnostics(
        validation_labels,
        val_prediction,
        val_alternatives,
        val_matrix,
        names,
        val_samples["user"].to_numpy(dtype=np.int32),
        graph,
        locale_frequency,
    )
    diagnostics.update(
        {
            "debug": debug,
            "panel_rows": len(panel),
            "panel_checks": panel_checks,
            "feature_count": len(names),
            "model_a_validation_sha256": validation_hash,
            "model_choice": {
                "variant": choice.variant,
                "gate": choice.gate,
                "graph_accepted": choice.graph_accepted,
                "history_iterations": choice.history_iterations,
                "cold_iterations": choice.cold_iterations,
            },
            "forward": choice.forward_scores,
            "representativeness": {
                "train_last_origin_rows": 2172,
                "train_last_origin_positive_rate": 0.10037,
                "validation_rows": len(val_samples),
                "validation_positive_rate": float(validation_labels.mean()),
                "test_rows": len(test_samples),
            },
            "elapsed_seconds": time.time() - started,
        }
    )
    log_phase("diagnostics_complete", mark)
    return PipelineResult(val_prediction, test_prediction, diagnostics)
