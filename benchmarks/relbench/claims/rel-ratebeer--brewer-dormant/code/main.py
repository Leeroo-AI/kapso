from __future__ import annotations

import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import EventStore, brewer_static_table, build_episode_pools, build_features, cache_feature_matrix, contextual_features, demand_tables, immutable_seed, phase, widening_features
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions
from models import fit_selected, predict_selected, select_design


def aligned_matrix(all_seeds: pd.DataFrame, all_features: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(all_seeds[["timestamp", "brewer_id"]])
    query = pd.MultiIndex.from_frame(requested[["timestamp", "brewer_id"]])
    positions = index.get_indexer(query)
    if np.any(positions < 0):
        raise RuntimeError(f"feature alignment failed for {int(np.sum(positions < 0))} rows")
    return all_features.iloc[positions].reset_index(drop=True)


def prediction_diagnostics(prediction: np.ndarray) -> dict:
    return {
        "count": int(len(prediction)),
        "minimum": float(np.min(prediction)),
        "maximum": float(np.max(prediction)),
        "mean": float(np.mean(prediction)),
        "standard_deviation": float(np.std(prediction)),
        "unique_rounded_8": int(len(np.unique(np.round(prediction, 8)))),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = is_debug()
    np.random.seed(1337)
    print(f"[run] dense-landmark renewal ensemble debug={debug}", flush=True)
    context = load_task()
    phase("task and database load", started)
    validation_seed = immutable_seed(context.val.df)
    test_seed = immutable_seed(context.test.df)
    store = EventStore.from_context(context)
    store.assert_official(context.train.df)
    phase("event store and label assertion", started)
    pool_a, pool_b = build_episode_pools(context, store, debug)
    phase("episode generation", started)
    static = brewer_static_table(context)
    demand = demand_tables(context, debug)
    phase("relational demand aggregation", started)
    combined_seed = pd.concat(
        [pool_b[["timestamp", "brewer_id"]], validation_seed[["timestamp", "brewer_id"]], test_seed[["timestamp", "brewer_id"]]],
        ignore_index=True,
    )
    combined_seed["timestamp"] = pd.to_datetime(combined_seed["timestamp"])
    combined_seed = combined_seed.drop_duplicates(["timestamp", "brewer_id"]).sort_values(["timestamp", "brewer_id"], kind="mergesort").reset_index(drop=True)
    matrix_name = f"features_{'debug' if debug else 'full'}"
    all_features = cache_feature_matrix(matrix_name, combined_seed, lambda: build_features(combined_seed, store, static, demand, include_cohort=True))
    widening_name = f"widening_{'debug' if debug else 'full'}_v1"
    widening = cache_feature_matrix(widening_name, combined_seed, lambda: widening_features(combined_seed, store, all_features, context))
    all_features = pd.concat([all_features, widening], axis=1)
    contextual_name = f"contextual_{'debug' if debug else 'full'}_v1"
    contextual = cache_feature_matrix(contextual_name, combined_seed, lambda: contextual_features(combined_seed, store, static, all_features, context))
    all_features = pd.concat([all_features, contextual], axis=1)
    phase("feature matrix", started)
    X_a = aligned_matrix(combined_seed, all_features, pool_a)
    X_b = aligned_matrix(combined_seed, all_features, pool_b)
    X_val = aligned_matrix(combined_seed, all_features, validation_seed)
    X_test = aligned_matrix(combined_seed, all_features, test_seed)
    selection = select_design(X_a, pool_a, debug)
    print(f"[selection] {selection}", flush=True)
    phase("forward selection", started)
    model_a = fit_selected(X_a, pool_a, selection, seed=7331)
    validation_prediction, validation_components = predict_selected(model_a, X_val, selection)
    frozen_validation = validation_prediction.copy()
    output = run_data_dir()
    np.save(output / "val_predictions.npy", frozen_validation)
    print(f"[freeze] validation predictions frozen from Model A: {prediction_diagnostics(frozen_validation)}", flush=True)
    phase("model A fit and validation prediction", started)
    del model_a, X_a, X_val, validation_components
    gc.collect()
    model_b = fit_selected(X_b, pool_b, selection, seed=7331)
    test_prediction, test_components = predict_selected(model_b, X_test, selection)
    phase("model B fit and test prediction", started)
    if not np.array_equal(frozen_validation, validation_prediction):
        raise RuntimeError("frozen validation predictions changed after Model B")
    save_predictions(frozen_validation.astype(np.float64), test_prediction.astype(np.float64))
    metrics = {
        "debug": debug,
        "elapsed_seconds": time.time() - started,
        "model_a_rows": int(len(pool_a)),
        "model_b_rows": int(len(pool_b)),
        "feature_count": int(all_features.shape[1]),
        "validation": prediction_diagnostics(frozen_validation),
        "test": prediction_diagnostics(test_prediction),
        "selection": selection.diagnostics,
        "selected_design": {
            "half_life": selection.half_life,
            "lgb_rounds": selection.lgb_rounds,
            "lgb_seed_ensemble": selection.lgb_seed_ensemble,
            "lgb_rounds_2": selection.lgb_rounds_2,
            "keep_cat": selection.keep_cat,
            "cat_rounds": selection.cat_rounds,
            "keep_aft": selection.keep_aft,
            "aft_rounds": selection.aft_rounds,
            "aft_distribution": selection.aft_distribution,
            "aft_scale": selection.aft_scale,
            "blend_weights": selection.blend_weights,
            "blend_mode": selection.blend_mode,
            "keep_context": selection.keep_context,
        },
        "test_component_means": {name: float(np.mean(value)) for name, value in test_components.items()},
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    phase("prediction validation and write", started)


if __name__ == "__main__":
    main()
