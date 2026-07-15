from __future__ import annotations

import numpy as np
import pandas as pd

from calendar_forecast import NUM_DST, calendar_forecast
from driver_attendance import cohort_matrix, predict_attendance

DEFAULT_META = {
    "decay": 0.82,
    "blend": (0.55, 0.30, 0.15),
    "affinity_weight": 0.08,
    "affinity_cap": 1.0,
    "tie_weight": 0.03,
    "incumbent_gamma": 0.5,
    "incumbent_tau": 0.35,
    "n_rounds": None,
}


# Fraction of the previous season a driver actually started (incumbency strength)
def incumbent_strength(driver_events, driver_id, seed_time, latest_year, n_rounds_prev):
    e = driver_events.get(driver_id)
    if e is None or n_rounds_prev <= 0:
        return 0.0
    mask = (e["date"] <= np.datetime64(seed_time)) & (e["year"] == latest_year)
    if not mask.any():
        return 0.0
    rounds = np.unique(e["round"][mask])
    return float(min(1.0, len(rounds) / n_rounds_prev))


# Driver-circuit affinity from historical decayed appearances
def driver_affinity(driver_events, driver_id, seed_time, latest_year, decay):
    scores = np.zeros(NUM_DST)
    e = driver_events.get(driver_id)
    if e is None:
        return scores
    mask = e["date"] <= np.datetime64(seed_time)
    if not mask.any():
        return scores
    for c, yr in zip(e["circuitId"][mask], e["year"][mask]):
        scores[int(c)] += decay ** max(0, latest_year - int(yr))
    m = scores.max()
    if m > 0:
        scores = scores / m
    return scores


def global_tie(season_circuits, latest_year, decay):
    scores = np.zeros(NUM_DST)
    norm = 0.0
    for y in [yy for yy in season_circuits if yy <= latest_year]:
        w = decay ** (latest_year - y)
        norm += w
        for c in season_circuits[y]:
            if 0 <= c < NUM_DST:
                scores[c] += w
    if norm > 0:
        scores = scores / norm
    return scores


def round_assignment(round_pred, n_rounds):
    idx = np.clip(np.round(round_pred * (n_rounds - 1)).astype(int) + 1, 1, n_rounds)
    return idx


# Rank circuits for a full cohort at one seed
def rank_cohort(driver_events, driver_year_rounds, dob, cohort_ids, seed_time, ty,
                latest_year, season_circuits, season_round_pos, season_n_rounds,
                presence_model, round_model, attend_model, meta, k):
    decay = meta["decay"]
    p_cal, round_pred = calendar_forecast(
        season_circuits, season_round_pos, latest_year,
        presence_model, round_model, decay, meta["blend"],
    )
    n_rounds = meta["n_rounds"]
    if n_rounds is None:
        recent = [season_n_rounds[y] for y in range(latest_year - 2, latest_year + 1)
                  if y in season_n_rounds]
        n_rounds = int(round(np.mean(recent))) if recent else 17
    n_rounds_prev = season_n_rounds.get(latest_year, n_rounds)
    round_idx = round_assignment(round_pred, n_rounds)

    X, _, index = cohort_matrix(
        driver_events, driver_year_rounds, dob, cohort_ids, seed_time, ty,
        latest_year, n_rounds, n_rounds_prev, with_labels=False,
    )
    p_attend_flat = predict_attendance(attend_model, X)
    attend = {}
    for (d, r), p in zip(index, p_attend_flat):
        attend.setdefault(d, np.zeros(n_rounds + 1))[r] = p

    tie = global_tie(season_circuits, latest_year, decay)
    gamma = meta["incumbent_gamma"]

    circuits = np.arange(NUM_DST)
    preds = {}
    for d in cohort_ids:
        aff = driver_affinity(driver_events, d, seed_time, latest_year, decay)
        aff = np.minimum(aff, meta["affinity_cap"])
        a = np.clip(attend[d][1:], 0.0, 1.0)
        strength = incumbent_strength(driver_events, d, seed_time, latest_year, n_rounds_prev)
        eff_strength = min(1.0, strength / meta["incumbent_tau"]) if meta["incumbent_tau"] > 0 else 1.0
        soft = eff_strength + (1.0 - eff_strength) * gamma
        eff = np.concatenate([[0.0], soft + (1.0 - soft) * a])
        attend_by_circuit = eff[round_idx]
        score = (
            p_cal * attend_by_circuit
            + meta["affinity_weight"] * aff
            + meta["tie_weight"] * tie
        )
        order = np.argsort(-(score - 1e-9 * circuits))
        preds[d] = order[:k]
    return preds


def map_at_k(pred, true_set, k):
    if not true_set:
        return None
    hits = np.array([1 if p in true_set else 0 for p in pred[:k]])
    if hits.sum() == 0:
        return 0.0
    precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1)
    return float((precision_at * hits).sum() / min(len(true_set), k))


# Leave-one-season-out rolling-origin backtest (out-of-fold models per year)
def rolling_origin_score_loo(driver_events, driver_year_rounds, year_cohort, dob,
                             season_circuits, season_round_pos, season_n_rounds,
                             year_driver_circuits, loo_models, score_years, meta, k):
    per_year, weights = [], []
    max_year = max(score_years)
    for ty in score_years:
        latest_year = ty - 1
        truth = year_driver_circuits.get(ty, {})
        cohort = sorted(year_cohort.get(ty, set()))
        if not truth or not cohort or ty not in loo_models:
            continue
        presence_model, round_model, attend_model = loo_models[ty]
        seed_time = pd.Timestamp(f"{ty}-01-01")
        preds = rank_cohort(
            driver_events, driver_year_rounds, dob, cohort, seed_time, ty,
            latest_year, season_circuits, season_round_pos, season_n_rounds,
            presence_model, round_model, attend_model, meta, k,
        )
        maps = []
        for d in cohort:
            m = map_at_k(preds[d], truth.get(d, set()), k)
            if m is not None:
                maps.append(m)
        if maps:
            per_year.append(float(np.mean(maps)))
            weights.append(3.0 if ty >= max_year - 3 else 1.0)
    if not per_year:
        return 0.0, 0.0
    per_year = np.array(per_year)
    weights = np.array(weights)
    mean_map = float((per_year * weights).sum() / weights.sum())
    worst_map = float(per_year.min())
    return mean_map, worst_map
