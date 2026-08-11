from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from feature_pipeline import _register_artifact, source_hash


EMBEDDING_VERSION = "lane0_pubmedbert_384_v2"
MODEL_NAME = "NeuML/pubmedbert-base-embeddings"


def _normalized(values: np.ndarray) -> np.ndarray:
    denominator = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(denominator, 1e-12)


class FrozenEmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.path = cache_dir / f"{EMBEDDING_VERSION}.npz"
        self.hashes: list[str] = []
        self.vectors = np.empty((0, 768), dtype=np.float32)
        if self.path.exists():
            loaded = np.load(self.path, allow_pickle=False)
            self.hashes = loaded["hashes"].astype(str).tolist()
            self.vectors = loaded["vectors"].astype(np.float32, copy=False)
        self.index = {key: i for i, key in enumerate(self.hashes)}

    def encode(self, texts: list[str], limit: int | None = None) -> np.ndarray:
        hashes = [source_hash(text) for text in texts]
        unique = {}
        for key, value in zip(hashes, texts):
            if key not in self.index and key not in unique:
                unique[key] = value
        missing_hashes = list(unique)
        if limit is not None:
            missing_hashes = missing_hashes[:limit]
        if missing_hashes:
            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            from huggingface_hub.utils import disable_progress_bars
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as transformers_logging
            disable_progress_bars()
            transformers_logging.set_verbosity_error()
            started = time.time()
            model = SentenceTransformer(MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
            model.max_seq_length = 384
            new_vectors = model.encode([unique[key] for key in missing_hashes], batch_size=128 if torch.cuda.is_available() else 16, show_progress_bar=False, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
            elapsed = time.time() - started
            rate = len(missing_hashes) / max(elapsed, 1e-9)
            print(f"[text] encoded {len(missing_hashes)} unique chunks in {elapsed:.1f}s rate={rate:.1f}/s")
            start_index = len(self.hashes)
            self.hashes.extend(missing_hashes)
            self.vectors = np.concatenate([self.vectors, new_vectors], axis=0)
            self.index.update({key: start_index + i for i, key in enumerate(missing_hashes)})
            temporary = self.path.with_suffix(".tmp.npz")
            np.savez(temporary, hashes=np.asarray(self.hashes, dtype="U64"), vectors=self.vectors)
            os.replace(temporary, self.path)
            _register_artifact(self.cache_dir, "lane0 PubMedBERT source-hash embeddings", self.path, "Normalized frozen NeuML/pubmedbert-base-embeddings vectors cached by SHA-256 source hash.", EMBEDDING_VERSION)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = np.zeros((len(texts), 768), dtype=np.float32)
        for i, key in enumerate(hashes):
            index = self.index.get(key)
            if index is not None:
                result[i] = self.vectors[index]
        return result


def embed_documents(bank: dict, cache_dir: Path, debug: bool) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict]:
    ids = sorted(bank["protocol_documents"])
    protocol_texts = [bank["protocol_documents"].get(nct_id, "") for nct_id in ids]
    chunk_ids = []
    chunk_texts = []
    for nct_id in ids:
        for chunk in bank["eligibility_chunks"].get(nct_id, [""])[:3]:
            chunk_ids.append(nct_id)
            chunk_texts.append(chunk)
    cache = FrozenEmbeddingCache(cache_dir)
    before = len(cache.hashes)
    budget = 500 if debug else None
    protocol_vectors = cache.encode(protocol_texts, limit=budget)
    remaining = None if budget is None else max(0, budget - (len(cache.hashes) - before))
    eligibility_vectors = cache.encode(chunk_texts, limit=remaining)
    protocol = {nct_id: protocol_vectors[i] for i, nct_id in enumerate(ids)}
    eligibility = {}
    positions = {}
    for i, nct_id in enumerate(chunk_ids):
        positions.setdefault(nct_id, []).append(i)
    for nct_id in ids:
        indices = positions.get(nct_id, [])
        if indices:
            vector = eligibility_vectors[indices].mean(axis=0, keepdims=True)
            eligibility[nct_id] = _normalized(vector)[0]
        else:
            eligibility[nct_id] = np.zeros(768, dtype=np.float32)
    coverage_protocol = float(np.mean([np.linalg.norm(protocol[nct_id]) > 0 for nct_id in ids]))
    coverage_eligibility = float(np.mean([np.linalg.norm(eligibility[nct_id]) > 0 for nct_id in ids]))
    diagnostics = {"model": MODEL_NAME, "max_tokens": 384, "protocol_coverage": coverage_protocol, "eligibility_coverage": coverage_eligibility, "cache_vectors": len(cache.hashes)}
    print(f"[text] coverage protocol={coverage_protocol:.3f} eligibility={coverage_eligibility:.3f} cached={len(cache.hashes)}")
    return protocol, eligibility, diagnostics


def _evidence_at(timestamp: pd.Timestamp, valid: pd.DataFrame) -> pd.DataFrame:
    current = valid[valid["date"] <= timestamp]
    evidence = current.groupby("nct_id").agg(min_p=("p_value", "min"), count=("p_value", "size"), evidence_date=("date", "max")).reset_index()
    evidence["label"] = (evidence["min_p"] <= 0.05).astype(float)
    return evidence


def _neighbor_values(similarities: np.ndarray, labels: np.ndarray, k: int) -> tuple[float, float, float, float, float, float]:
    size = min(k, len(labels))
    if size == 0:
        return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0
    sim = similarities[:size]
    lab = labels[:size]
    weights = np.exp(6.0 * (sim - np.max(sim)))
    probability = float(np.sum(weights * lab) / np.maximum(np.sum(weights), 1e-12))
    mean_similarity = float(np.mean(sim))
    gap = float(sim[0] - sim[-1])
    entropy = float(-(probability * math_log(probability) + (1 - probability) * math_log(1 - probability)))
    effective = float(np.sum(weights) ** 2 / np.maximum(np.sum(weights ** 2), 1e-12))
    coverage = float(size / k)
    return probability, mean_similarity, gap, entropy, effective, coverage


def math_log(value: float) -> float:
    return float(np.log(np.clip(value, 1e-8, 1 - 1e-8)))


def _condition_overlap(query_id: int, history_ids: np.ndarray, condition_sets: dict[int, set]) -> np.ndarray:
    query = condition_sets.get(query_id, set())
    if not query:
        return np.zeros(len(history_ids), dtype=bool)
    return np.asarray([bool(query.intersection(condition_sets.get(int(nct_id), set()))) for nct_id in history_ids], dtype=bool)


def build_retrieval_features(bank: dict, protocol: dict[int, np.ndarray], eligibility: dict[int, np.ndarray], debug: bool) -> pd.DataFrame:
    rows = bank["rows"]
    valid = bank["valid_analyses"]
    phases = bank["phases"]
    condition_sets = bank["condition_sets"]
    vectors = {}
    for nct_id in protocol:
        combined = protocol[nct_id] + eligibility.get(nct_id, 0)
        norm = float(np.linalg.norm(combined))
        vectors[nct_id] = combined / norm if norm > 0 else np.zeros(768, dtype=np.float32)
    outputs = []
    ks = [25] if debug else [10, 25, 50, 100]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for timestamp, block in rows.groupby("timestamp", sort=True):
        evidence = _evidence_at(timestamp, valid)
        evidence = evidence[~evidence["nct_id"].isin(block["nct_id"])]
        evidence = evidence[evidence["nct_id"].isin(vectors)]
        history_ids = evidence["nct_id"].to_numpy(dtype=np.int64)
        history_labels = evidence["label"].to_numpy(dtype=np.float32)
        if len(history_ids) == 0:
            current = block[["_global_id"]].copy()
            for k in ks:
                for suffix in ["success", "mean_similarity", "gap", "entropy", "effective_n", "coverage"]:
                    current[f"retrieval_k{k}_{suffix}"] = 0.5 if suffix == "success" else 0.0
            outputs.append(current)
            continue
        query_ids = block["nct_id"].to_numpy(dtype=np.int64)
        pool = min(500, len(history_ids))
        records = [{"_global_id": int(value)} for value in block["_global_id"].to_numpy(dtype=np.int64)]
        families = {
            "": vectors,
            "protocol": protocol,
            "eligibility": eligibility,
        }
        for family, mapping in families.items():
            history_matrix = np.stack([mapping.get(int(nct_id), np.zeros(768, dtype=np.float32)) for nct_id in history_ids]).astype(np.float32)
            query_matrix = np.stack([mapping.get(int(nct_id), np.zeros(768, dtype=np.float32)) for nct_id in query_ids]).astype(np.float32)
            history_tensor = torch.from_numpy(history_matrix).to(device)
            similarity_parts = []
            index_parts = []
            for start in range(0, len(query_ids), 512):
                query_tensor = torch.from_numpy(query_matrix[start:start + 512]).to(device)
                similarities = query_tensor @ history_tensor.T
                local_ids = query_ids[start:start + len(query_tensor)]
                for i, nct_id in enumerate(local_ids):
                    same = np.flatnonzero(history_ids == nct_id)
                    if len(same):
                        similarities[i, torch.as_tensor(same, device=device)] = -2.0
                top_values, top_indices = torch.topk(similarities, k=pool, dim=1)
                similarity_parts.append(top_values.cpu().numpy())
                index_parts.append(top_indices.cpu().numpy())
            similarities = np.concatenate(similarity_parts)
            indices = np.concatenate(index_parts)
            prefix = "retrieval" if not family else f"retrieval_{family}"
            for position, nct_id in enumerate(query_ids):
                candidate_indices = indices[position]
                candidate_ids = history_ids[candidate_indices]
                candidate_labels = history_labels[candidate_indices]
                candidate_sim = similarities[position]
                for k in ks:
                    values = _neighbor_values(candidate_sim, candidate_labels, k)
                    for suffix, value in zip(["success", "mean_similarity", "gap", "entropy", "effective_n", "coverage"], values):
                        records[position][f"{prefix}_k{k}_{suffix}"] = value
                if not family:
                    phase = phases.get(int(nct_id), "__MISSING__")
                    phase_mask = np.asarray([phases.get(int(history_id), "__MISSING__") == phase for history_id in candidate_ids])
                    condition_mask = _condition_overlap(int(nct_id), candidate_ids, condition_sets)
                    for variant, mask in [("phase", phase_mask), ("condition", condition_mask)]:
                        selected_sim = candidate_sim[mask]
                        selected_labels = candidate_labels[mask]
                        for k in ([25] if debug else [25, 100]):
                            values = _neighbor_values(selected_sim, selected_labels, k)
                            for suffix, value in zip(["success", "mean_similarity", "gap", "entropy", "effective_n", "coverage"], values):
                                records[position][f"retrieval_{variant}_k{k}_{suffix}"] = value
            del history_tensor
        outputs.append(pd.DataFrame(records))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    result = pd.concat(outputs, ignore_index=True)
    print(f"[retrieval] built {len(result)} rows with {len(result.columns) - 1} semantic-neighbor features")
    return result


def add_retrieval_ranks(features: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in features if column.startswith("retrieval_") and column.endswith("_success")]
    for column in columns:
        grouped = features.groupby("timestamp")[column]
        features[f"rank_{column}"] = grouped.rank(pct=True, method="average")
        standard = grouped.transform("std").replace(0, np.nan)
        features[f"z_{column}"] = (features[column] - grouped.transform("mean")) / standard
        features[f"gap_{column}"] = grouped.transform("max") - features[column]
    return features
