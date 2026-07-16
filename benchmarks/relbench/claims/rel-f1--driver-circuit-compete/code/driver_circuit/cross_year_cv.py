# Step 5: cross-year k-fold CV synthesis and hyperparameter selection
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from kapso_datasets.common import eval_map_at_k

from driver_circuit.config import (
    DEFAULT_WEIGHTS,
    H_P_LONG_GRID,
    LAMBDA_GRID,
    POP_WINDOW_GRID,
    SURVIVAL_WEIGHT_CAP,
    WEIGHT_SIMPLEX_STEP,
)
from driver_circuit.scoring import predict


@dataclass
class CvFold:
    seed_time: pd.Timestamp
    seed_year: int
    drivers: np.ndarray
    truth: list


def build_cv_folds(inter, cv_years, timedelta_days):
    folds = []
    for year in cv_years:
        seed_time = pd.Timestamp(year=year, month=1, day=1)
        fut = inter[
            (inter["date"] > seed_time)
            & (inter["date"] <= seed_time + pd.Timedelta(days=timedelta_days))
        ]
        gt = {}
        for d, c in zip(fut["driverId"].to_numpy(), fut["circuitId"].to_numpy()):
            gt.setdefault(int(d), set()).add(int(c))
        drivers = np.array(sorted(gt), dtype=np.int64)
        truth = [gt[int(d)] for d in drivers]
        folds.append(CvFold(seed_time, year, drivers, truth))
    return folds


def evaluate_config_cv(config, folds, inter, races_hist, eval_k):
    per_year = {}
    for fold in folds:
        pred = predict(fold.drivers, fold.seed_time, inter, races_hist, config)
        per_year[fold.seed_year] = eval_map_at_k(pred, fold.truth, eval_k)
    values = list(per_year.values())
    mean_map = float(np.mean(values))
    min_map = float(np.min(values))
    return mean_map, min_map, per_year


def cv_objective(mean_map, min_map, alpha):
    return mean_map - alpha * (mean_map - min_map)


def weight_simplex(step, survival_cap):
    n = int(round(1.0 / step))
    points = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            w = (a / n, b / n, c / n)
            if w[2] <= survival_cap + 1e-9 and w[0] >= w[1] - 1e-9:
                points.append(w)
    return points


def search_config(inter, races_hist, base_config, folds, alpha, eval_k):
    log = []

    best_stage1 = None
    for h_p in H_P_LONG_GRID:
        for pop_window in POP_WINDOW_GRID:
            for lam in LAMBDA_GRID:
                cfg = replace(
                    base_config,
                    h_p_long=h_p,
                    pop_window_days=pop_window,
                    lam=lam,
                    weights=DEFAULT_WEIGHTS,
                )
                mean_map, min_map, _ = evaluate_config_cv(cfg, folds, inter, races_hist, eval_k)
                obj = cv_objective(mean_map, min_map, alpha)
                if best_stage1 is None or obj > best_stage1[0]:
                    best_stage1 = (obj, cfg, mean_map, min_map)
    obj1, cfg1, mean1, min1 = best_stage1
    log.append(
        f"stage1 obj={obj1:.4f} mean={mean1:.4f} min={min1:.4f} "
        f"h_p_long={cfg1.h_p_long:g} pop_window={cfg1.pop_window_days:g} lam={cfg1.lam:g}"
    )

    best_stage2 = None
    for weights in weight_simplex(WEIGHT_SIMPLEX_STEP, SURVIVAL_WEIGHT_CAP):
        cfg = replace(cfg1, weights=weights)
        mean_map, min_map, per_year = evaluate_config_cv(cfg, folds, inter, races_hist, eval_k)
        obj = cv_objective(mean_map, min_map, alpha)
        if best_stage2 is None or obj > best_stage2[0]:
            best_stage2 = (obj, cfg, mean_map, min_map, per_year)
    obj2, cfg2, mean2, min2, per_year2 = best_stage2
    log.append(
        f"stage2 obj={obj2:.4f} mean={mean2:.4f} min={min2:.4f} "
        f"weights=({cfg2.weights[0]:.1f},{cfg2.weights[1]:.1f},{cfg2.weights[2]:.1f})"
    )
    return cfg2, per_year2, log
