"""Time-decayed repeat-behavior + popularity recommendation baseline.

On purchase/visit-style RelBench recommendation tasks, most of a source's
future destinations are repeats of its past ones. This baseline ranks each
source's historical destinations by exponentially time-decayed interaction
count and backfills with recent global popularity. It is a strong baseline
(often beating two-tower GNNs) and a mandatory sanity bar for anything fancier.

Usage from main.py:

    from kapso_datasets.common import load_task, save_predictions
    from kapso_datasets.recommendation_heuristics import repeat_and_popular

    ctx = load_task()
    # interactions: the event table linking src->dst over time. Pick the table
    # and columns that define the task's interaction (see task make_table SQL).
    val_pred = repeat_and_popular(ctx, split_table=ctx.val, ...)
    test_pred = repeat_and_popular(ctx, split_table=ctx.test, ...)
    save_predictions(val_pred, test_pred)
"""

from __future__ import annotations


import numpy as np
import pandas as pd


def repeat_and_popular(
    interactions: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: str,
    seed_table: pd.DataFrame,
    seed_src_col: str,
    seed_time_col: str,
    k: int,
    num_dst: int,
    half_life_days: float = 30.0,
    popularity_window_days: float = 60.0,
) -> np.ndarray:
    """Return (len(seed_table), k) ranked destination ids.

    All seed rows in RelBench recommendation val/test share one timestamp, so
    the history cutoff is computed once (works for per-row cutoffs too).
    """
    out = np.zeros((len(seed_table), k), dtype=np.int64)
    inter = interactions[[src_col, dst_col, time_col]].dropna()

    for seed_time, idx in seed_table.groupby(seed_time_col).groups.items():
        idx = np.asarray(idx)
        hist = inter[inter[time_col] <= seed_time]  # temporal censoring

        # Global popularity over a recent window (fallback ranking).
        recent = hist[hist[time_col] >= seed_time - pd.Timedelta(days=popularity_window_days)]
        pop_source = recent if len(recent) > 1000 else hist
        popular = pop_source[dst_col].value_counts().index.to_numpy()[: 5 * k].astype(np.int64)
        popular = popular[popular < num_dst]

        # Per-source time-decayed destination scores.
        age_days = (seed_time - hist[time_col]).dt.total_seconds() / 86400.0
        weights = np.power(0.5, age_days / half_life_days)
        scored = (
            pd.DataFrame({src_col: hist[src_col].values, dst_col: hist[dst_col].values, "w": weights.values})
            .groupby([src_col, dst_col])["w"]
            .sum()
            .reset_index()
            .sort_values([src_col, "w"], ascending=[True, False])
        )
        per_src = {s: g[dst_col].to_numpy(dtype=np.int64) for s, g in scored.groupby(src_col)}

        for row_i in idx:
            src = seed_table[seed_src_col].iloc[row_i]
            own = per_src.get(src, np.empty(0, dtype=np.int64))
            own = own[own < num_dst][:k]
            ranked = list(own)
            for p in popular:
                if len(ranked) >= k:
                    break
                if p not in own:
                    ranked.append(int(p))
            j = 0
            while len(ranked) < k:  # pathological fallback: fill distinct ids
                if j not in ranked:
                    ranked.append(j)
                j += 1
            out[row_i] = np.asarray(ranked[:k], dtype=np.int64)
    return out


def co_occurrence_candidates(
    interactions: pd.DataFrame,
    src_col: str,
    dst_col: str,
    time_col: str,
    cutoff,
    window_days: float = 7.0,
    top_n: int = 20,
) -> dict:
    """dst -> list of dsts frequently co-interacted by the same src within a
    time window. Feed as extra candidates for a GBDT re-ranker."""
    hist = interactions[interactions[time_col] <= cutoff].sort_values(time_col)
    pairs = {}
    for _, g in hist.groupby(src_col):
        dsts = g[dst_col].to_numpy()
        times = g[time_col].to_numpy()
        for i in range(len(dsts)):
            lo = np.searchsorted(times, times[i] - np.timedelta64(int(window_days * 86400), "s"))
            for j in range(lo, len(dsts)):
                if i == j or dsts[i] == dsts[j]:
                    continue
                key = (int(dsts[i]), int(dsts[j]))
                pairs[key] = pairs.get(key, 0) + 1
    cooc: dict = {}
    for (a, b), c in pairs.items():
        cooc.setdefault(a, []).append((c, b))
    return {a: [b for _, b in sorted(v, reverse=True)[:top_n]] for a, v in cooc.items()}
