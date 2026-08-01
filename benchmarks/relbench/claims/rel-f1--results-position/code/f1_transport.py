from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata


@dataclass(frozen=True)
class TransportConfig:
    pmf_mixture: float
    entropy: float
    rank_width: float
    blend: float


def isotonic_rectify(values: np.ndarray) -> np.ndarray:
    levels: list[float] = []
    weights: list[int] = []
    lengths: list[int] = []
    for value in np.clip(np.asarray(values, dtype=float), 0.0, 1.0):
        levels.append(float(value))
        weights.append(1)
        lengths.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            weight = weights[-2] + weights[-1]
            level = (levels[-2] * weights[-2] + levels[-1] * weights[-1]) / weight
            length = lengths[-2] + lengths[-1]
            levels[-2:] = [level]
            weights[-2:] = [weight]
            lengths[-2:] = [length]
    output = np.empty(len(values), dtype=float)
    offset = 0
    for level, length in zip(levels, lengths):
        output[offset:offset + length] = level
        offset += length
    return output


def ordinal_pmf(probabilities: np.ndarray, thresholds: np.ndarray, size: int) -> np.ndarray:
    if size <= 1:
        return np.ones(1, dtype=float)
    usable = thresholds < size
    selected_thresholds = thresholds[usable].astype(float)
    selected_probabilities = isotonic_rectify(probabilities[usable])
    nodes = np.concatenate(([0.0], selected_thresholds, [float(size)]))
    cumulative = np.concatenate(([0.0], selected_probabilities, [1.0]))
    full_cumulative = np.interp(np.arange(1, size + 1, dtype=float), nodes, cumulative)
    mass = np.diff(np.concatenate(([0.0], full_cumulative)))
    mass = np.clip(mass, 1e-12, None)
    return mass / mass.sum()


def normalized_score_rank(scores: np.ndarray) -> np.ndarray:
    size = len(scores)
    if size <= 1:
        return np.zeros(size, dtype=float)
    return (rankdata(-np.asarray(scores), method="average") - 1.0) / (size - 1.0)


def gaussian_rank_pmf(center: float, width: float, size: int) -> np.ndarray:
    positions = np.arange(1, size + 1, dtype=float)
    mass = np.exp(-0.5 * ((positions - center) / width) ** 2)
    mass = np.clip(mass, 1e-12, None)
    return mass / mass.sum()


def sinkhorn(likelihood: np.ndarray, entropy: float, iterations: int) -> tuple[np.ndarray, float]:
    log_kernel = np.log(np.clip(likelihood, 1e-12, None)) / entropy
    row_scaling = np.zeros(log_kernel.shape[0], dtype=float)
    column_scaling = np.zeros(log_kernel.shape[1], dtype=float)
    for _ in range(iterations):
        row_scaling = -logsumexp(log_kernel + column_scaling[None, :], axis=1)
        column_scaling = -logsumexp(log_kernel + row_scaling[:, None], axis=0)
    plan = np.exp(log_kernel + row_scaling[:, None] + column_scaling[None, :])
    error = max(
        float(np.max(np.abs(plan.sum(axis=1) - 1.0))),
        float(np.max(np.abs(plan.sum(axis=0) - 1.0))),
    )
    return plan, error


def rank_fractions(scores: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    output = np.zeros(len(scores), dtype=float)
    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)
        output[indices] = normalized_score_rank(scores[indices])
    return output


def clip_by_race(predictions: np.ndarray, race_ids: np.ndarray) -> np.ndarray:
    output = np.asarray(predictions, dtype=float).copy()
    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)
        output[indices] = np.clip(output[indices], 1.0, float(len(indices)))
    return output


def decode_transport(
    ordinal_probabilities: np.ndarray,
    thresholds: np.ndarray,
    rank_scores: np.ndarray,
    race_ids: np.ndarray,
    rank_calibrator,
    config: TransportConfig,
    iterations: int,
) -> tuple[np.ndarray, float]:
    output = np.zeros(len(rank_scores), dtype=float)
    maximum_error = 0.0
    for race_id in np.unique(race_ids):
        indices = np.flatnonzero(race_ids == race_id)
        size = len(indices)
        ordinal_matrix = np.vstack([
            ordinal_pmf(ordinal_probabilities[index], thresholds, size)
            for index in indices
        ])
        fractions = normalized_score_rank(rank_scores[indices])
        calibrated = np.clip(rank_calibrator.predict(fractions), 0.0, 1.0)
        centers = 1.0 + calibrated * max(size - 1, 1)
        rank_matrix = np.vstack([
            gaussian_rank_pmf(center, config.rank_width, size)
            for center in centers
        ])
        likelihood = config.pmf_mixture * ordinal_matrix + (1.0 - config.pmf_mixture) * rank_matrix
        plan, error = sinkhorn(likelihood, config.entropy, iterations)
        maximum_error = max(maximum_error, error)
        output[indices] = (plan @ np.arange(1, size + 1, dtype=float)) / plan.sum(axis=1)
    return output, maximum_error
