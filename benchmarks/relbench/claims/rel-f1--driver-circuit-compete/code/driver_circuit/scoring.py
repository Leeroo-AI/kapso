# Steps 2-4: scoring function, per-variant ranks, rank-average ensemble, emit
from __future__ import annotations

import numpy as np
import pandas as pd

from kapso_datasets.common import assert_temporal_safety

from driver_circuit.config import HeuristicConfig


def _age_days(dates: pd.Series, seed_time: pd.Timestamp) -> np.ndarray:
    return (seed_time - dates).dt.total_seconds().to_numpy() / 86400.0


def personal_repeat_matrix(
    hist: pd.DataFrame,
    unique_drivers: np.ndarray,
    seed_time: pd.Timestamp,
    half_life: float,
    num_dst: int,
) -> np.ndarray:
    matrix = np.zeros((len(unique_drivers), num_dst), dtype=np.float64)
    if len(hist) == 0:
        return matrix
    drivers = hist["driverId"].to_numpy()
    circuits = hist["circuitId"].to_numpy()
    weights = np.power(0.5, _age_days(hist["date"], seed_time) / half_life)
    row_of = {int(d): i for i, d in enumerate(unique_drivers)}
    mask = np.isin(drivers, unique_drivers) & (circuits < num_dst) & (circuits >= 0)
    if not mask.any():
        return matrix
    rows = np.fromiter((row_of[int(d)] for d in drivers[mask]), dtype=np.int64)
    np.add.at(matrix, (rows, circuits[mask].astype(np.int64)), weights[mask])
    return matrix


def circuit_survival_scores(
    hist: pd.DataFrame,
    races_hist: pd.DataFrame,
    seed_time: pd.Timestamp,
    seed_year: int,
    config: HeuristicConfig,
) -> np.ndarray:
    popularity = np.zeros(config.num_dst, dtype=np.float64)
    windowed = hist[hist["date"] > seed_time - pd.Timedelta(days=config.pop_window_days)]
    circuits = windowed["circuitId"].to_numpy()
    mask = (circuits < config.num_dst) & (circuits >= 0)
    np.add.at(popularity, circuits[mask].astype(np.int64), 1.0)

    bonus = np.zeros(config.num_dst, dtype=np.float64)
    lo = seed_year - config.recent_seasons
    window = races_hist[(races_hist["year"] >= lo) & (races_hist["year"] < seed_year)]
    seasons = window.groupby("circuitId")["year"].nunique()
    idx = seasons.index.to_numpy().astype(np.int64)
    keep = (idx < config.num_dst) & (idx >= 0)
    bonus[idx[keep]] = seasons.to_numpy()[keep]
    return popularity + config.lam * bonus


def _ranks_descending(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = np.empty_like(order)
    cols = np.arange(scores.shape[1])
    for i in range(scores.shape[0]):
        ranks[i, order[i]] = cols
    return ranks


def predict(
    seed_driver_ids,
    seed_time: pd.Timestamp,
    inter: pd.DataFrame,
    races_hist: pd.DataFrame,
    config: HeuristicConfig,
) -> np.ndarray:
    seed_time = pd.Timestamp(seed_time)
    hist = inter[inter["date"] <= seed_time]
    rhist = races_hist[races_hist["date"] <= seed_time]
    assert_temporal_safety(hist, "date", seed_time)
    assert_temporal_safety(rhist, "date", seed_time)

    seed_driver_ids = np.asarray(seed_driver_ids, dtype=np.int64)
    unique_drivers = np.array(sorted(set(int(d) for d in seed_driver_ids)), dtype=np.int64)

    survival = circuit_survival_scores(hist, rhist, seed_time, seed_time.year, config)
    personal_long = personal_repeat_matrix(
        hist, unique_drivers, seed_time, config.h_p_long, config.num_dst
    )
    personal_short = personal_repeat_matrix(
        hist, unique_drivers, seed_time, config.h_p_short, config.num_dst
    )

    score_long = config.m_repeat * personal_long + survival[None, :]
    score_short = config.m_repeat * personal_short + survival[None, :]
    score_survival = np.broadcast_to(survival[None, :], (len(unique_drivers), config.num_dst))

    ranks = (
        config.weights[0] * _ranks_descending(score_long)
        + config.weights[1] * _ranks_descending(score_short)
        + config.weights[2] * _ranks_descending(score_survival)
    )
    order = np.argsort(ranks, axis=1, kind="stable")
    topk = order[:, : config.eval_k].astype(np.int64)

    row_of = {int(d): i for i, d in enumerate(unique_drivers)}
    out = np.empty((len(seed_driver_ids), config.eval_k), dtype=np.int64)
    for i, d in enumerate(seed_driver_ids):
        out[i] = topk[row_of[int(d)]]
    return out
