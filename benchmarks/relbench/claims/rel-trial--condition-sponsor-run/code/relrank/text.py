import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .cache import SCHEMA_VERSION, cache_root, content_hash, register_artifact


QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"


def _configure() -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _encode(texts, model_name: str, max_length: int, batch_size: int, output_path: Path) -> np.ndarray:
    _configure()
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval().to(device)
    total = len(texts)
    output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float16, shape=(total, model.config.hidden_size))
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            batch = [str(x) for x in texts[start : start + batch_size]]
            tokens = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(device) for k, v in tokens.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                hidden = model(**tokens).last_hidden_state[:, 0]
            values = torch.nn.functional.normalize(hidden.float(), dim=1)
            output[start : start + len(batch)] = values.cpu().numpy().astype(np.float16)
    output.flush()
    del output, model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r")


def _join_limited(values, limit: int) -> str:
    clean = [str(x) for x in values if pd.notna(x) and str(x)]
    return " ".join(clean[:limit])


def _base_documents(assets):
    studies = assets.tables["studies"]
    elig = assets.tables["eligibilities"][["nct_id", "criteria"]]
    interventions = assets.tables["interventions_studies"][["nct_id", "intervention_id"]].merge(
        assets.tables["interventions"][["intervention_id", "mesh_term"]], on="intervention_id", how="left"
    )
    intervention_text = interventions.groupby("nct_id", sort=False)["mesh_term"].agg(lambda x: _join_limited(x, 12))
    criteria = elig.drop_duplicates("nct_id").set_index("nct_id")["criteria"]
    docs = []
    for row in studies.itertuples(index=False):
        fields = [
            getattr(row, "brief_title", ""),
            getattr(row, "official_title", ""),
            getattr(row, "brief_summaries", ""),
            criteria.get(row.nct_id, ""),
            intervention_text.get(row.nct_id, ""),
        ]
        docs.append(" ".join(str(x) for x in fields if pd.notna(x) and str(x)))
    return docs


def _result_documents(assets):
    outcomes = assets.tables["outcomes"]
    grouped = outcomes.groupby(["nct_id", "date"], sort=False)
    metadata = grouped.size().reset_index(name="count")
    titles = grouped["title"].agg(lambda x: _join_limited(x, 16)).reset_index(name="title")
    descriptions = grouped["description"].agg(lambda x: _join_limited(x, 8)).reset_index(name="description")
    metadata = metadata.merge(titles, on=["nct_id", "date"]).merge(descriptions, on=["nct_id", "date"])
    docs = (metadata["title"].fillna("") + " " + metadata["description"].fillna("")).tolist()
    return metadata[["nct_id", "date"]], docs


class TextAssets:
    def __init__(self, assets, debug: bool):
        self.root = cache_root()
        self.enabled = True
        self.result_meta = None
        key = content_hash([SCHEMA_VERSION, QUERY_MODEL, ARTICLE_MODEL, len(assets.tables["studies"]), len(assets.tables["outcomes"])])
        query_path = self.root / f"condition_query_{key}.npy"
        study_path = self.root / f"study_article_{key}.npy"
        result_path = self.root / f"result_article_{key}.npy"
        result_meta_path = self.root / f"result_meta_{key}.parquet"
        started = time.time()
        if not query_path.exists():
            queries = assets.tables["conditions"].sort_values("condition_id")["mesh_term"].fillna("").tolist()
            _encode(queries, QUERY_MODEL, 64, 512 if torch.cuda.is_available() else 32, query_path)
            register_artifact("lane0 MedCPT condition queries", query_path, "Frozen MedCPT condition-query embeddings", key, "Run main.py to rebuild")
        if debug and not study_path.exists():
            self.enabled = False
            self.query = np.load(query_path, mmap_mode="r")
            self.study = None
            self.result = None
            print(f"[text] debug query batch ready in {time.time() - started:.1f}s; full article build deferred")
            return
        if not study_path.exists():
            docs = _base_documents(assets)
            _encode(docs, ARTICLE_MODEL, 256, 256 if torch.cuda.is_available() else 16, study_path)
            register_artifact("lane0 MedCPT study articles", study_path, "Frozen MedCPT embeddings for cutoff-visible study documents", key, "Run main.py to rebuild")
        if not result_path.exists() or not result_meta_path.exists():
            metadata, docs = _result_documents(assets)
            _encode(docs, ARTICLE_MODEL, 256, 256 if torch.cuda.is_available() else 16, result_path)
            metadata.to_parquet(result_meta_path, index=False)
            register_artifact("lane0 MedCPT result articles", result_path, "Frozen MedCPT embeddings for dated outcome documents", key, "Run main.py to rebuild")
            register_artifact("lane0 MedCPT result metadata", result_meta_path, "Cutoff dates and study ids for outcome embeddings", key, "Run main.py to rebuild")
        self.query = np.load(query_path, mmap_mode="r")
        self.study = np.load(study_path, mmap_mode="r")
        self.result = np.load(result_path, mmap_mode="r")
        self.result_meta = pd.read_parquet(result_meta_path)
        print(f"[text] MedCPT assets ready in {time.time() - started:.1f}s")


def portfolio_embeddings(snapshot, text_assets: TextAssets, cutoff_key: str) -> np.ndarray | None:
    if not text_assets.enabled:
        return None
    root = cache_root()
    key = content_hash([SCHEMA_VERSION, "portfolio", cutoff_key, snapshot.sponsor_study.nnz, len(text_assets.result_meta)])
    path = root / f"sponsor_portfolio_{key}.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    dimensions = text_assets.study.shape[1]
    output = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(snapshot.n_sponsors, dimensions))
    counts = np.asarray(snapshot.sponsor_study.sum(axis=1)).ravel().astype(np.float32)
    metadata = text_assets.result_meta
    visible = metadata["date"].to_numpy() <= snapshot.cutoff.to_datetime64()
    result_rows = metadata.loc[visible, ["nct_id"]].copy()
    result_rows["result_index"] = np.flatnonzero(visible)
    sponsor_rows = snapshot.sponsor_rel[["nct_id", "sponsor_id"]].merge(result_rows, on="nct_id", how="inner")
    from scipy import sparse

    result_matrix = sparse.coo_matrix(
        (
            np.ones(len(sponsor_rows), dtype=np.float32),
            (sponsor_rows["sponsor_id"].to_numpy(), sponsor_rows["result_index"].to_numpy()),
        ),
        shape=(snapshot.n_sponsors, len(metadata)),
    ).tocsr()
    counts += np.asarray(result_matrix.sum(axis=1)).ravel().astype(np.float32)
    for start in range(0, dimensions, 64):
        end = min(dimensions, start + 64)
        study_values = np.asarray(text_assets.study[:, start:end], dtype=np.float32)
        result_values = np.asarray(text_assets.result[:, start:end], dtype=np.float32)
        values = snapshot.sponsor_study @ study_values
        values += result_matrix @ result_values
        values /= np.maximum(counts[:, None], 1.0)
        output[:, start:end] = values.astype(np.float16)
    output.flush()
    del output
    values = np.load(path, mmap_mode="r+")
    for start in range(0, len(values), 2048):
        block = np.asarray(values[start : start + 2048], dtype=np.float32)
        values[start : start + len(block)] = _normalize(block).astype(np.float16)
    values.flush()
    del values
    register_artifact(
        f"lane0 MedCPT sponsor portfolio {cutoff_key}",
        path,
        "Time-censored sponsor portfolio embedding from visible studies and outcomes",
        key,
        "Run main.py to rebuild the cutoff snapshot",
    )
    return np.load(path, mmap_mode="r")


def semantic_topk(condition_ids: np.ndarray, query: np.ndarray, portfolio: np.ndarray | None, k: int):
    if portfolio is None:
        return None, None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sponsor_tensor = torch.as_tensor(np.asarray(portfolio, dtype=np.float32), device=device)
    all_ids = []
    all_scores = []
    with torch.inference_mode():
        for start in range(0, len(condition_ids), 128):
            q = torch.as_tensor(np.asarray(query[condition_ids[start : start + 128]], dtype=np.float32), device=device)
            scores = q @ sponsor_tensor.T
            values, ids = torch.topk(scores, k=k, dim=1)
            all_ids.append(ids.cpu().numpy())
            all_scores.append(values.cpu().numpy())
    del sponsor_tensor
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.vstack(all_ids), np.vstack(all_scores).astype(np.float32)
