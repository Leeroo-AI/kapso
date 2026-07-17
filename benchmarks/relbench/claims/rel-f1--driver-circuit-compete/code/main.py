# Repeat-behavior + calendar-survival heuristic for rel-f1/driver-circuit-compete
from __future__ import annotations

import json
import time
import warnings
from dataclasses import replace

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from kapso_datasets.common import (
    eval_map_at_k,
    is_debug,
    load_task,
    save_predictions,
    shared_cache_dir,
)
from driver_circuit.config import (
    CV_YEARS,
    HeuristicConfig,
    MODEL_A_SEED_TIME,
    MODEL_B_SEED_TIME,
    SEED,
    WORST_YEAR_ALPHA,
)
from driver_circuit.cross_year_cv import (
    build_cv_folds,
    cv_objective,
    evaluate_config_cv,
    search_config,
)
from driver_circuit.interactions import (
    INTERACTIONS_VERSION,
    build_interactions,
    build_races_history,
)
from driver_circuit.scoring import predict


def main() -> None:
    start = time.time()
    np.random.seed(SEED)
    debug = is_debug()

    # Step 1: load task, build interactions and race history
    ctx = load_task()
    task = ctx.task
    eval_k = int(task.eval_k)
    num_dst = int(task.num_dst_nodes)
    timedelta_days = int(task.timedelta.days)
    cache = shared_cache_dir()

    inter = build_interactions(ctx.db, cache, INTERACTIONS_VERSION)
    races_hist = build_races_history(ctx.db)

    val_df = ctx.val.df
    test_df = ctx.test.df
    val_sizes = val_df["circuitId"].apply(len)
    print(
        f"[eda] interactions={len(inter)} drivers={inter['driverId'].nunique()} "
        f"circuits={inter['circuitId'].nunique()} num_dst={num_dst} eval_k={eval_k} "
        f"timedelta_days={timedelta_days} debug={debug}"
    )
    print(
        f"[eda] val_rows={len(val_df)} test_rows={len(test_df)} "
        f"gt_size mean={val_sizes.mean():.2f} median={val_sizes.median():g} max={val_sizes.max():g}"
    )
    print(f"[phase] load+eda elapsed={time.time() - start:.1f}s")

    base = HeuristicConfig(num_dst=num_dst, eval_k=eval_k)

    # Step 5: cross-year CV selection (skipped in debug per Step 7)
    if debug:
        config = base
        print(
            f"[debug] default config weights={config.weights} h_p_long={config.h_p_long:g} "
            f"pop_window={config.pop_window_days:g} lam={config.lam:g}"
        )
    else:
        folds = build_cv_folds(inter, CV_YEARS, timedelta_days)
        for fold in folds:
            nonempty = sum(1 for x in fold.truth if x)
            print(
                f"[cv-fold] year={fold.seed_year} active_drivers={len(fold.drivers)} "
                f"nonempty_truth={nonempty}"
            )
        d_mean, d_min, _ = evaluate_config_cv(base, folds, inter, races_hist, eval_k)
        print(
            f"[cv] default obj={cv_objective(d_mean, d_min, WORST_YEAR_ALPHA):.4f} "
            f"mean={d_mean:.4f} min={d_min:.4f}"
        )
        config, chosen_per_year, log = search_config(
            inter, races_hist, base, folds, WORST_YEAR_ALPHA, eval_k
        )
        for line in log:
            print(f"[cv-search] {line}")
        for year, value in sorted(chosen_per_year.items()):
            print(f"[cv] chosen year={year} map={value:.4f}")
        print(
            f"[cv] chosen config weights={config.weights} h_p_long={config.h_p_long:g} "
            f"pop_window={config.pop_window_days:g} lam={config.lam:g}"
        )
        print(f"[phase] cv-search elapsed={time.time() - start:.1f}s")

    # Step 6: two-model temporal protocol (val at V, test at T)
    val_seed_time = pd.Timestamp(MODEL_A_SEED_TIME)
    test_seed_time = pd.Timestamp(MODEL_B_SEED_TIME)
    val_pred = predict(val_df["driverId"].to_numpy(), val_seed_time, inter, races_hist, config)
    test_pred = predict(test_df["driverId"].to_numpy(), test_seed_time, inter, races_hist, config)

    # Step 8: diagnostics (val MAP is reported, never used for selection)
    val_truth = [set(int(c) for c in x) for x in val_df["circuitId"].to_numpy()]
    val_map = eval_map_at_k(val_pred, val_truth, eval_k)
    print(f"[val] official val MAP@{eval_k} = {val_map:.4f}")
    print(
        f"[dist] val_distinct_ids={len(np.unique(val_pred))} "
        f"test_distinct_ids={len(np.unique(test_pred))}"
    )

    save_predictions(val_pred, test_pred)

    # Step 8: cache ensemble and per-variant predictions for later iterations
    if not debug:
        np.save(cache / f"driver_circuit_val_pred_{INTERACTIONS_VERSION}.npy", val_pred)
        np.save(cache / f"driver_circuit_test_pred_{INTERACTIONS_VERSION}.npy", test_pred)
        variant_weights = {"repeat_long": (1.0, 0.0, 0.0), "repeat_short": (0.0, 1.0, 0.0), "survival": (0.0, 0.0, 1.0)}
        for name, weights in variant_weights.items():
            vcfg = replace(config, weights=weights)
            vp = predict(val_df["driverId"].to_numpy(), val_seed_time, inter, races_hist, vcfg)
            tp = predict(test_df["driverId"].to_numpy(), test_seed_time, inter, races_hist, vcfg)
            np.save(cache / f"driver_circuit_val_{name}_{INTERACTIONS_VERSION}.npy", vp)
            np.save(cache / f"driver_circuit_test_{name}_{INTERACTIONS_VERSION}.npy", tp)
        (cache / f"driver_circuit_config_{INTERACTIONS_VERSION}.json").write_text(
            json.dumps(
                {
                    "weights": list(config.weights),
                    "h_p_long": config.h_p_long,
                    "h_p_short": config.h_p_short,
                    "pop_window_days": config.pop_window_days,
                    "lam": config.lam,
                    "val_map": val_map,
                },
                indent=2,
            )
        )

    print(f"[phase] total elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
