from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.utils import sort_edge_index


# Text sources

class TextSource:
    def __init__(self, size: int, getter: Callable[[np.ndarray], list[str]], lengths: np.ndarray):
        self.size = int(size)
        self.getter = getter
        self.lengths = np.asarray(lengths, dtype=np.float32)

    def batch(self, start: int, stop: int) -> list[str]:
        return self.getter(np.arange(start, stop, dtype=np.int64))

    def indexed(self, indices: np.ndarray) -> list[str]:
        return self.getter(np.asarray(indices, dtype=np.int64))


def string_values(series: pd.Series, indices: np.ndarray) -> list[str]:
    return series.iloc[indices].fillna("").astype(str).tolist()


def make_text_sources(db) -> dict[str, TextSource]:
    posts = db.table_dict["posts"].df
    history = db.table_dict["postHistory"].df
    comments = db.table_dict["comments"].df
    parts: dict[int, np.ndarray] = {}
    for history_type in (1, 2, 3):
        values = np.empty(len(posts), dtype=object)
        values[:] = ""
        selected = history.loc[
            history["PostHistoryTypeId"].eq(history_type), ["PostId", "Text"]
        ].drop_duplicates("PostId", keep="first")
        ids = selected["PostId"].to_numpy(dtype=np.int64)
        valid = (ids >= 0) & (ids < len(posts))
        values[ids[valid]] = selected["Text"].fillna("").astype(str).to_numpy()[valid]
        parts[history_type] = values

    def post_getter(indices: np.ndarray) -> list[str]:
        title = parts[1][indices]
        body = parts[2][indices]
        tags = parts[3][indices]
        return [f"{a} {b} {c}" for a, b, c in zip(title, body, tags)]

    post_lengths = np.fromiter(
        (
            len(str(a)) + len(str(b)) + len(str(c))
            for a, b, c in zip(parts[1], parts[2], parts[3])
        ),
        dtype=np.float32,
        count=len(posts),
    )
    history_text = history["Text"]
    history_comment = history["Comment"]

    def history_getter(indices: np.ndarray) -> list[str]:
        text = history_text.iloc[indices].fillna("").astype(str).tolist()
        comment = history_comment.iloc[indices].fillna("").astype(str).tolist()
        return [f"{a} {b}" for a, b in zip(text, comment)]

    history_lengths = (
        history_text.fillna("").astype(str).str.len().to_numpy(dtype=np.float32)
        + history_comment.fillna("").astype(str).str.len().to_numpy(dtype=np.float32)
    )
    comment_lengths = comments["Text"].fillna("").astype(str).str.len().to_numpy(dtype=np.float32)
    return {
        "posts": TextSource(len(posts), post_getter, post_lengths),
        "postHistory": TextSource(len(history), history_getter, history_lengths),
        "comments": TextSource(
            len(comments),
            lambda indices: string_values(comments["Text"], indices),
            comment_lengths,
        ),
    }


# Text embedding cache

@dataclass
class TextResources:
    embeddings: dict[str, np.ndarray]
    lengths: dict[str, np.ndarray]
    content_key: str


def register_artifact(shared_root: Path, entry: dict[str, str]) -> None:
    path = shared_root / "artifacts.json"
    lock_path = shared_root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(path.read_text()) if path.exists() else []
            if not any(record.get("content_key") == entry["content_key"] for record in records):
                records.append(entry)
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(records, indent=2))
                os.replace(temporary, path)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def projection_matrix(device: torch.device, dimensions: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(73129)
    value = torch.randn(384, dimensions, generator=generator, dtype=torch.float32)
    value = torch.linalg.qr(value, mode="reduced").Q
    return value.to(device)


def materialize_text_embeddings(
    db,
    shared_root: Path,
    debug: bool,
    device: torch.device,
    dimensions: int = 64,
) -> TextResources:
    content_key = "rel-stack-safe-minilm-l6-v2-rp64-lane3-v1"
    sources = make_text_sources(db)
    lengths = {name: source.lengths for name, source in sources.items()}
    if debug:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
        model.eval()
        projection = projection_matrix(device, dimensions)
        embeddings: dict[str, np.ndarray] = {}
        for name, source in sources.items():
            value = np.zeros((source.size, dimensions), dtype=np.float16)
            indices = np.linspace(0, source.size - 1, min(4096, source.size), dtype=np.int64)
            with torch.inference_mode():
                encoded = model.encode(
                    source.indexed(indices),
                    batch_size=256,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
                reduced = torch.nn.functional.normalize(encoded @ projection, dim=1)
            value[indices] = reduced.cpu().numpy().astype(np.float16)
            embeddings[name] = value
            print(f"[text] debug {name}: encoded={len(indices)} total={source.size}")
        return TextResources(embeddings, lengths, content_key + "-debug")

    root = shared_root / content_key
    root.mkdir(parents=True, exist_ok=True)
    complete_path = root / "complete.json"
    expected = {name: source.size for name, source in sources.items()}
    if complete_path.exists():
        metadata = json.loads(complete_path.read_text())
        if metadata.get("rows") == expected and metadata.get("dimensions") == dimensions:
            embeddings = {
                name: np.load(root / f"{name}.npy", mmap_mode="r") for name in sources
            }
            print(f"[text] reused cache {content_key}")
            return TextResources(embeddings, lengths, content_key)

    from sentence_transformers import SentenceTransformer

    started = time.time()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=str(device))
    model.eval()
    projection = projection_matrix(device, dimensions)
    embeddings = {}
    for name, source in sources.items():
        path = root / f"{name}.npy"
        value = np.lib.format.open_memmap(
            path,
            mode="w+",
            dtype=np.float16,
            shape=(source.size, dimensions),
        )
        table_started = time.time()
        for start in range(0, source.size, 8192):
            stop = min(source.size, start + 8192)
            texts = source.batch(start, stop)
            chunks: list[np.ndarray] = []
            for inner in range(0, len(texts), 256):
                with torch.inference_mode():
                    encoded = model.encode(
                        texts[inner : inner + 256],
                        batch_size=256,
                        show_progress_bar=False,
                        convert_to_tensor=True,
                        normalize_embeddings=True,
                    )
                    reduced = torch.nn.functional.normalize(encoded @ projection, dim=1)
                chunks.append(reduced.cpu().numpy().astype(np.float16))
            value[start:stop] = np.concatenate(chunks)
            if start and start % 131072 == 0:
                value.flush()
                rate = stop / max(time.time() - table_started, 1e-8)
                print(f"[text] {name}: {stop}/{source.size} rate={rate:.0f}/s")
        value.flush()
        embeddings[name] = np.load(path, mmap_mode="r")
        elapsed = time.time() - table_started
        print(f"[text] {name}: complete rows={source.size} elapsed={elapsed:.1f}s")
    complete_path.write_text(
        json.dumps({"content_key": content_key, "rows": expected, "dimensions": dimensions})
    )
    register_artifact(
        shared_root,
        {
            "name": "lane3 safe MiniLM node encodings",
            "path": content_key,
            "description": "Frozen all-MiniLM-L6-v2 encodings of initial post history, immutable comments, and timestamped history text, projected to 64 dimensions.",
            "content_key": content_key,
            "rebuild_hint": "Run the full lane-3 candidate; it extends or rebuilds cutoff-independent text arrays.",
        },
    )
    print(f"[text] all tables complete elapsed={time.time() - started:.1f}s")
    return TextResources(embeddings, lengths, content_key)


# Graph construction

def unix_seconds(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype="datetime64[s]").astype(np.int64)


def prefix_count(table, cutoff: pd.Timestamp) -> int:
    times = table.df[table.time_col]
    mask = times.le(cutoff).to_numpy()
    count = int(mask.sum())
    if not mask[:count].all() or mask[count:].any():
        raise RuntimeError(f"{table.time_col} cutoff is not a primary-key prefix")
    return count


def categorical_matrix(df: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, list[int]]:
    if not columns:
        return np.empty((len(df), 0), dtype=np.int64), []
    values: list[np.ndarray] = []
    cardinalities: list[int] = []
    for column in columns:
        codes, uniques = pd.factorize(df[column], sort=True, use_na_sentinel=True)
        encoded = codes.astype(np.int64) + 1
        values.append(encoded)
        cardinalities.append(len(uniques) + 1)
    return np.column_stack(values), cardinalities


def numerical_matrix(
    name: str,
    df: pd.DataFrame,
    text_resources: TextResources,
) -> np.ndarray:
    constant = np.ones(len(df), dtype=np.float32)
    if name == "users":
        account = pd.to_numeric(df["AccountId"], errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        return np.column_stack(
            [constant, np.log1p(np.maximum(account, 0)) / 20.0, (account > 0).astype(np.float32)]
        ).astype(np.float32)
    if name in text_resources.lengths:
        length = text_resources.lengths[name][: len(df)]
        return np.column_stack([constant, np.log1p(length) / 10.0]).astype(np.float32)
    if name == "badges":
        return np.column_stack([constant, df["TagBased"].fillna(False).to_numpy(dtype=np.float32)])
    return constant[:, None]


def graph_categories(name: str) -> list[str]:
    return {
        "posts": ["PostTypeId"],
        "votes": ["VoteTypeId"],
        "users": [],
        "badges": ["Class", "Name", "TagBased"],
        "postLinks": ["LinkTypeId"],
        "postHistory": ["PostHistoryTypeId", "ContentLicense"],
        "comments": ["ContentLicense"],
    }[name]


def build_graph_view(
    db,
    cutoff: pd.Timestamp,
    text_resources: TextResources,
) -> tuple[HeteroData, dict[str, dict[str, object]], dict[str, int]]:
    started = time.time()
    data = HeteroData()
    counts = {
        name: prefix_count(table, cutoff) for name, table in db.table_dict.items()
    }
    feature_meta: dict[str, dict[str, object]] = {}
    for name, table in db.table_dict.items():
        count = counts[name]
        df = table.df.iloc[:count]
        numerical = numerical_matrix(name, df, text_resources)
        categories, cardinalities = categorical_matrix(df, graph_categories(name))
        data[name].num_nodes = count
        data[name].x_num = torch.from_numpy(np.ascontiguousarray(numerical))
        data[name].cat = torch.from_numpy(np.ascontiguousarray(categories))
        data[name].time = torch.from_numpy(unix_seconds(df[table.time_col]))
        text_dim = 0
        if name in text_resources.embeddings:
            text = text_resources.embeddings[name][:count]
            data[name].x_text = torch.from_numpy(np.asarray(text))
            text_dim = int(text.shape[1])
        feature_meta[name] = {
            "numerical_dim": int(numerical.shape[1]),
            "text_dim": text_dim,
            "cardinalities": cardinalities,
        }
    edge_count = 0
    for source_name, table in db.table_dict.items():
        source_count = counts[source_name]
        source_df = table.df.iloc[:source_count]
        for foreign_key, target_name in table.fkey_col_to_pkey_table.items():
            raw = pd.to_numeric(source_df[foreign_key], errors="coerce").to_numpy(dtype=np.float64)
            valid = np.isfinite(raw) & (raw >= 0) & (raw < counts[target_name])
            source_ids = np.flatnonzero(valid).astype(np.int64)
            target_ids = raw[valid].astype(np.int64)
            forward = sort_edge_index(
                torch.from_numpy(np.stack([source_ids, target_ids])),
                num_nodes=max(source_count, counts[target_name]),
            )
            reverse = sort_edge_index(
                torch.from_numpy(np.stack([target_ids, source_ids])),
                num_nodes=max(source_count, counts[target_name]),
            )
            data[(source_name, f"f2p_{foreign_key}", target_name)].edge_index = forward
            data[(target_name, f"rev_f2p_{foreign_key}", source_name)].edge_index = reverse
            edge_count += 2 * len(source_ids)
    data.validate(raise_on_error=True)
    print(
        f"[graph] cutoff={cutoff} nodes={sum(counts.values())} edges={edge_count} "
        f"tables={len(counts)} elapsed={time.time() - started:.1f}s"
    )
    return data, feature_meta, counts


def valid_query_mask(db, table_df: pd.DataFrame, entity_col: str, time_col: str, user_count: int) -> np.ndarray:
    node_ids = table_df[entity_col].to_numpy(dtype=np.int64)
    query_time = unix_seconds(table_df[time_col])
    valid = (node_ids >= 0) & (node_ids < user_count)
    user_times = unix_seconds(db.table_dict["users"].df["CreationDate"])
    positions = np.flatnonzero(valid)
    valid[positions] &= user_times[node_ids[positions]] <= query_time[positions]
    return valid


def content_fingerprint(paths: list[Path]) -> str:
    value = hashlib.sha256()
    for path in paths:
        value.update(path.name.encode())
        value.update(str(path.stat().st_size).encode())
    return value.hexdigest()[:16]
