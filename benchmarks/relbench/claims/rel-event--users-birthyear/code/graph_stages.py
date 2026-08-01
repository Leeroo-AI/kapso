from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from relational import GraphData


SOFT_NAMES = [
    "soft_one_mean", "soft_one_median", "soft_one_variance", "soft_one_hard_fraction", "soft_one_hard_mass",
    "soft_hard_mean", "soft_hard_disagreement", "soft_two_mean", "soft_two_median", "soft_two_variance",
    "soft_two_hard_fraction", "soft_any_friend", "soft_effective_degree", "soft_gate"
]


def _neighbor_summary(
    center_pos: int,
    cutoff_ns: int,
    graph: GraphData,
    predictions: np.ndarray,
    label_values: dict[int, float],
    label_times: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center_id = int(graph.ids[center_pos])
    neighbors = graph.neighbors(center_id, cutoff_ns)
    if len(neighbors) == 0:
        return neighbors, np.empty(0), np.empty(0), np.empty(0, dtype=bool)
    neighbor_ids = graph.ids[neighbors]
    hard = np.asarray([int(uid) in label_values and label_times[int(uid)] < cutoff_ns for uid in neighbor_ids], dtype=bool)
    values = predictions[neighbors].astype(float).copy()
    if hard.any():
        values[hard] = np.asarray([label_values[int(uid)] for uid in neighbor_ids[hard]])
    weights = 1.0 / np.log2(2.0 + graph.degree[neighbors].astype(float))
    return neighbors, values, weights, hard


def soft_neighbor_features(seeds: pd.DataFrame, graph: GraphData, predictions: np.ndarray, labels: pd.DataFrame) -> pd.DataFrame:
    label_values = {int(row.user_id): float(row.birthyear) for row in labels.itertuples(index=False)}
    label_times = {int(row.user_id): int(pd.Timestamp(row.joinedAt).value) for row in labels.itertuples(index=False)}
    output = np.zeros((len(seeds), len(SOFT_NAMES)), dtype=np.float64)
    for i, seed in enumerate(seeds.itertuples(index=False)):
        uid = int(seed.user_id)
        cutoff = int(pd.Timestamp(seed.joinedAt).value)
        center = graph.id_to_pos.get(uid)
        if center is None:
            continue
        neighbors, values, weights, hard = _neighbor_summary(center, cutoff, graph, predictions, label_values, label_times)
        if len(neighbors) == 0:
            continue
        mass = float(weights.sum())
        mean = float(np.dot(values, weights) / mass)
        variance = float(np.dot(np.square(values - mean), weights) / mass)
        hard_mass = float(weights[hard].sum())
        hard_mean = float(np.dot(values[hard], weights[hard]) / hard_mass) if hard_mass else mean
        second_values = np.empty(len(neighbors), dtype=np.float64)
        second_hard = np.empty(len(neighbors), dtype=bool)
        for j, node in enumerate(neighbors):
            node_id = int(graph.ids[node])
            is_hard = node_id in label_values and label_times[node_id] < cutoff
            second_hard[j] = is_hard
            if is_hard:
                second_values[j] = label_values[node_id]
                continue
            _, local_values, local_weights, _ = _neighbor_summary(int(node), cutoff, graph, predictions, label_values, label_times)
            second_values[j] = float(np.dot(local_values, local_weights) / local_weights.sum()) if len(local_values) else predictions[node]
        second_mean = float(np.dot(second_values, weights) / mass)
        second_variance = float(np.dot(np.square(second_values - second_mean), weights) / mass)
        effective = mass * mass / float(np.dot(weights, weights))
        disagreement = abs(hard_mean - float(np.dot(values[~hard], weights[~hard]) / weights[~hard].sum())) if hard.any() and (~hard).any() else 0.0
        hard_fraction = hard_mass / mass
        gate = (0.25 + 0.75 * (1.0 - np.exp(-hard_mass))) * np.exp(-disagreement / 20.0)
        output[i] = [
            mean, float(np.median(values)), variance, hard_fraction, hard_mass, hard_mean, disagreement,
            second_mean, float(np.median(second_values)), second_variance, float(np.dot(second_hard.astype(float), weights) / mass),
            1.0, effective, gate
        ]
    return pd.DataFrame(output, columns=SOFT_NAMES)


def correct_and_smooth_features(
    graph: GraphData,
    pool: pd.DataFrame,
    oof_prediction: np.ndarray,
    node_prediction: np.ndarray,
    targets: pd.DataFrame,
    correction_iterations: int = 5,
    correction_alpha: float = 0.8,
    smoothing_iterations: int = 3,
    smoothing_alpha: float = 0.5,
) -> pd.DataFrame:
    n = len(graph.ids)
    pool_end = int(pd.to_datetime(pool["joinedAt"]).astype("int64").max())
    active = graph.joined_ns <= pool_end
    edge_mask = active[graph.edges[:, 0]] & active[graph.edges[:, 1]]
    edges = graph.edges[edge_mask]
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    weight = 1.0 / np.sqrt(np.maximum(1.0, graph.degree[src].astype(float)) * np.maximum(1.0, graph.degree[dst].astype(float)))
    adjacency = sparse.csr_matrix((weight, (src, dst)), shape=(n, n))
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    transition = sparse.diags(1.0 / np.maximum(row_sum, 1e-12)) @ adjacency
    y = pool["birthyear"].to_numpy(dtype=float)
    mean = float(y.mean())
    scale = float(y.std()) or 1.0
    seed_numerator = np.zeros(n, dtype=np.float64)
    seed_mass = np.zeros(n, dtype=np.float64)
    valid_oof = np.isfinite(oof_prediction)
    for row, pred, valid in zip(pool.itertuples(index=False), oof_prediction, valid_oof):
        if not valid:
            continue
        pos = graph.id_to_pos.get(int(row.user_id))
        if pos is not None:
            seed_numerator[pos] = (float(row.birthyear) - pred) / scale
            seed_mass[pos] = 1.0
    numerator = seed_numerator.copy()
    mass = seed_mass.copy()
    for _ in range(correction_iterations):
        numerator = (1.0 - correction_alpha) * seed_numerator + correction_alpha * transition.dot(numerator)
        mass = (1.0 - correction_alpha) * seed_mass + correction_alpha * transition.dot(mass)
    correction = np.divide(numerator, mass, out=np.zeros_like(numerator), where=mass > 1e-10)
    mass_scale = (mass / (mass + 0.5)) * np.sqrt(graph.degree / np.maximum(graph.degree + 5.0, 1.0))
    correction = np.clip(correction * mass_scale, -0.5, 0.5)
    smooth = (node_prediction - mean) / scale
    fixed = np.zeros(n, dtype=bool)
    fixed_value = np.zeros(n, dtype=float)
    for row in pool.itertuples(index=False):
        pos = graph.id_to_pos.get(int(row.user_id))
        if pos is not None:
            fixed[pos] = True
            fixed_value[pos] = (float(row.birthyear) - mean) / scale
    smooth[fixed] = fixed_value[fixed]
    for _ in range(smoothing_iterations):
        updated = (1.0 - smoothing_alpha) * smooth + smoothing_alpha * transition.dot(smooth)
        updated[fixed] = fixed_value[fixed]
        smooth = updated
    result = np.zeros((len(targets), 5), dtype=np.float64)
    for i, row in enumerate(targets.itertuples(index=False)):
        uid = int(row.user_id)
        cutoff = int(pd.Timestamp(row.joinedAt).value)
        neighbors = graph.neighbors(uid, cutoff)
        neighbors = neighbors[active[neighbors]]
        if len(neighbors) == 0:
            continue
        weights = 1.0 / np.log2(2.0 + graph.degree[neighbors].astype(float))
        total = float(weights.sum())
        corr = float(np.dot(correction[neighbors], weights) / total)
        propagated_mass = float(np.dot(mass[neighbors], weights) / total)
        smooth_prediction = mean + scale * float(np.dot(smooth[neighbors], weights) / total)
        effective = total * total / float(np.dot(weights, weights))
        result[i] = [corr * scale, propagated_mass, effective, smooth_prediction, 1.0]
    return pd.DataFrame(result, columns=["cs_correction", "cs_mass", "cs_effective_degree", "cs_smooth_prediction", "cs_coverage"])
