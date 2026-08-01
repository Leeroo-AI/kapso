from __future__ import annotations

import os
import time

import hnswlib
import numpy as np


class CausalRetrieval:
    def __init__(self, embeddings: np.ndarray, signatures: dict[str, tuple[np.ndarray, np.ndarray]], max_elements: int, debug: bool):
        self.embeddings = embeddings
        self.signatures = signatures
        self.debug = debug
        self.neighbors = [16] if debug else [16, 64, 256]
        self.candidate_count = 16 if debug else 512
        self.thresholds = [0.85, 0.90, 0.95]
        self.temperature = 0.05
        self.threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
        self.index = hnswlib.Index(space="cosine", dim=embeddings.shape[1])
        self.index.init_index(max_elements=max_elements, ef_construction=180, M=32, random_seed=23)
        self.index.set_num_threads(self.threads)
        self.index.set_ef(max(64, self.candidate_count + 64))
        self.label_values = np.full(len(embeddings), np.nan, dtype=np.float32)
        self.total = 0
        self.positive_total = 0.0
        self.feature_names = ["retrieval_global_rate", "retrieval_history_log_support"]
        for k in self.neighbors:
            self.feature_names.extend([
                f"retrieval_k{k}_weighted_rate", f"retrieval_k{k}_rate", f"retrieval_k{k}_top_similarity",
                f"retrieval_k{k}_mean_similarity", f"retrieval_k{k}_positive_nearest_distance",
                f"retrieval_k{k}_negative_nearest_distance", f"retrieval_k{k}_effective_sample_size",
                f"retrieval_k{k}_similarity_gap", f"retrieval_k{k}_support",
            ])
        for threshold in self.thresholds:
            self.feature_names.extend([
                f"retrieval_threshold_{threshold:.2f}_weighted_rate",
                f"retrieval_threshold_{threshold:.2f}_rate",
                f"retrieval_threshold_{threshold:.2f}_support",
            ])
        for name in ["lead_sponsor", "condition", "intervention", "country", "phase", "study_type"]:
            self.feature_names.extend([
                f"retrieval_match_{name}_weighted_rate", f"retrieval_match_{name}_rate",
                f"retrieval_match_{name}_count", f"retrieval_match_{name}_top_similarity",
                f"retrieval_match_{name}_mean_similarity",
            ])

    def _global_rate(self) -> float:
        return self.positive_total / self.total if self.total else 0.5

    def _rates(self, similarities: np.ndarray, labels: np.ndarray, mask: np.ndarray, fallback: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        counts = mask.sum(axis=1).astype(np.float32)
        unweighted = np.divide((labels * mask).sum(axis=1), counts, out=np.full(len(mask), fallback, dtype=np.float32), where=counts > 0)
        masked_similarities = np.where(mask, similarities, -np.inf)
        maximum = np.max(masked_similarities, axis=1)
        maximum = np.where(counts > 0, maximum, 0.0).astype(np.float32)
        shifted = np.where(mask, (similarities - maximum[:, None]) / self.temperature, -80.0)
        weights = np.exp(np.clip(shifted, -80, 0)).astype(np.float32) * mask
        weight_sum = weights.sum(axis=1)
        weighted = np.divide((weights * labels).sum(axis=1), weight_sum, out=np.full(len(mask), fallback, dtype=np.float32), where=weight_sum > 0)
        mean_similarity = np.divide((similarities * mask).sum(axis=1), counts, out=np.zeros(len(mask), dtype=np.float32), where=counts > 0)
        return weighted, unweighted, counts, maximum, mean_similarity

    def _batch_features(self, query_indices: np.ndarray) -> np.ndarray:
        count = self.index.get_current_count()
        fallback = self._global_rate()
        output = np.empty((len(query_indices), len(self.feature_names)), dtype=np.float32)
        if count == 0:
            output.fill(0)
            output[:, 0] = fallback
            rate_columns = [index for index, name in enumerate(self.feature_names) if name.endswith("rate")]
            output[:, rate_columns] = fallback
            output[:, [index for index, name in enumerate(self.feature_names) if "nearest_distance" in name]] = 2.0
            return output
        query = np.asarray(self.embeddings[query_indices], dtype=np.float32)
        k_query = min(self.candidate_count, count)
        neighbor_indices, distances = self.index.knn_query(query, k=k_query, num_threads=self.threads)
        similarities = 1.0 - distances.astype(np.float32)
        labels = self.label_values[neighbor_indices]
        if not np.all(np.isfinite(labels)):
            raise RuntimeError("retrieval index contains an unlabeled row")
        columns = [np.full(len(query_indices), fallback, dtype=np.float32), np.full(len(query_indices), np.log1p(count), dtype=np.float32)]
        for k in self.neighbors:
            width = min(k, k_query)
            sims = similarities[:, :width]
            ys = labels[:, :width]
            mask = np.ones_like(sims, dtype=bool)
            weighted, unweighted, supports, top, mean = self._rates(sims, ys, mask, fallback)
            positive = np.where(ys > 0.5, sims, -np.inf).max(axis=1)
            negative = np.where(ys <= 0.5, sims, -np.inf).max(axis=1)
            positive_ok = np.isfinite(positive)
            negative_ok = np.isfinite(negative)
            positive_distance = np.where(positive_ok, 1.0 - positive, 2.0).astype(np.float32)
            negative_distance = np.where(negative_ok, 1.0 - negative, 2.0).astype(np.float32)
            gap = np.where(positive_ok & negative_ok, positive - negative, 0.0).astype(np.float32)
            shifted = (sims - sims.max(axis=1, keepdims=True)) / self.temperature
            weights = np.exp(np.clip(shifted, -80, 0)).astype(np.float32)
            effective = (weights.sum(axis=1) ** 2 / np.maximum((weights ** 2).sum(axis=1), 1e-12)).astype(np.float32)
            columns.extend([weighted, unweighted, top, mean, positive_distance, negative_distance, effective, gap, supports])
        for threshold in self.thresholds:
            mask = similarities >= threshold
            weighted, unweighted, supports, _, _ = self._rates(similarities, labels, mask, fallback)
            columns.extend([weighted, unweighted, supports])
        for name in ["lead_sponsor", "condition", "intervention", "country", "phase", "study_type"]:
            first, second = self.signatures[name]
            query_first = first[query_indices, None]
            query_second = second[query_indices, None]
            mask = ((first[neighbor_indices] & query_first) != 0) & ((second[neighbor_indices] & query_second) != 0)
            weighted, unweighted, supports, top, mean = self._rates(similarities, labels, mask, fallback)
            columns.extend([weighted, unweighted, supports, top, mean])
        output[:] = np.column_stack(columns)
        return output

    def _add(self, indices: np.ndarray, labels: np.ndarray) -> None:
        vectors = np.asarray(self.embeddings[indices], dtype=np.float32)
        self.index.add_items(vectors, np.asarray(indices, dtype=np.int64), num_threads=self.threads)
        self.label_values[indices] = np.asarray(labels, dtype=np.float32)
        self.total += len(indices)
        self.positive_total += float(np.asarray(labels).sum())

    def process_causal(self, indices: np.ndarray, dates: np.ndarray, labels: np.ndarray, phase_name: str) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        dates = np.asarray(dates)
        labels = np.asarray(labels, dtype=np.float32)
        order = np.argsort(dates, kind="stable")
        output = np.empty((len(indices), len(self.feature_names)), dtype=np.float32)
        cursor = 0
        start = time.time()
        checkpoint = start
        while cursor < len(order):
            end = cursor + 1
            while end < len(order) and dates[order[end]] == dates[order[cursor]]:
                end += 1
            batch_positions = order[cursor:end]
            batch_indices = indices[batch_positions]
            output[batch_positions] = self._batch_features(batch_indices)
            self._add(batch_indices, labels[batch_positions])
            cursor = end
            now = time.time()
            if now - checkpoint >= 60 or cursor == len(order):
                rate = cursor / max(now - start, 1e-6)
                print(f"[retrieval] phase={phase_name} rows={cursor}/{len(order)} rate={rate:.1f}/s history={self.total} elapsed={now - start:.1f}s", flush=True)
                checkpoint = now
        return output

    def transform(self, indices: np.ndarray, phase_name: str, batch_size: int = 1024) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64)
        output = np.empty((len(indices), len(self.feature_names)), dtype=np.float32)
        start = time.time()
        for begin in range(0, len(indices), batch_size):
            end = min(begin + batch_size, len(indices))
            output[begin:end] = self._batch_features(indices[begin:end])
        print(f"[retrieval] phase={phase_name} rows={len(indices)} history={self.total} elapsed={time.time() - start:.1f}s", flush=True)
        return output
