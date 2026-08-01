from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from gated_calendar_lambdamart import (
    crossfit_gate,
    crossfit_generative,
    fit_final_predictions,
    fit_predict_generative,
    forward_select,
    input_rows,
    load_or_build_matrix,
    official_episodes,
    prepare_data,
    reconstruct_episodes,
    validate_predictions,
)
from kapso_datasets.common import load_task, run_data_dir, save_predictions, shared_cache_dir


def elapsed(started: float, phase: str) -> None:
    print(f"[timing] phase={phase} elapsed_seconds={time.time() - started:.3f}")


def main() -> None:
    started = time.time()
    warnings.filterwarnings("ignore")
    debug = "--debug" in sys.argv
    if "KAPSO_RUN_DATA_DIR" not in os.environ:
        os.environ["KAPSO_RUN_DATA_DIR"] = "./output_data_generic_exp_0"
    trees = 100 if debug else 600
    generative_trees = 60 if debug else 180
    seeds = [17] if debug else [17, 43, 89]
    context = load_task()
    data = prepare_data(context.db)
    elapsed(started, "load")
    model_a_episodes = official_episodes(context.train)
    validation_inputs = input_rows(context.val)
    test_inputs = input_rows(context.test)
    cache_root = shared_cache_dir()
    model_a_matrix, model_a_cached = load_or_build_matrix(
        "model_a_train",
        model_a_episodes,
        data,
        cache_root,
        True,
    )
    validation_matrix, validation_cached = load_or_build_matrix(
        "model_a_validation_inputs",
        validation_inputs,
        data,
        cache_root,
        False,
    )
    print(
        f"[features] model_a_pairs={len(model_a_matrix.labels)} "
        f"feature_count={model_a_matrix.features.shape[1]} "
        f"cache_hits={int(model_a_cached) + int(validation_cached)}/2"
    )
    elapsed(started, "features")
    model_a_oof = crossfit_generative(model_a_matrix, generative_trees, debug)
    model_a_gate_oof = crossfit_gate(model_a_matrix, trees, debug)
    selected, fold_strata = forward_select(model_a_matrix, model_a_oof, model_a_gate_oof, trees, seeds, debug)
    elapsed(started, "selection")
    validation_generative = fit_predict_generative(model_a_matrix, validation_matrix, generative_trees)
    validation_predictions, validation_diagnostics = fit_final_predictions(
        model_a_matrix,
        validation_matrix,
        model_a_oof,
        validation_generative,
        selected,
        trees,
        seeds,
    )
    validation_predictions = np.asarray(validation_predictions, dtype=np.int64).copy()
    frozen_validation_predictions = validation_predictions.copy()
    validate_predictions(validation_predictions, len(validation_inputs))
    np.save(run_data_dir() / "val_predictions.npy", validation_predictions)
    elapsed(started, "model_a")
    validation_episodes = official_episodes(context.val) if "circuitId" in context.val.df else pd.DataFrame(columns=["date", "driverId", "target"])
    if len(test_inputs):
        test_year = int(pd.Timestamp(test_inputs["date"].min()).year)
    else:
        test_year = 2010
    reconstructed = reconstruct_episodes(data, list(range(2006, test_year)))
    model_b_parts = [model_a_episodes]
    if len(validation_episodes):
        model_b_parts.append(validation_episodes)
    if len(reconstructed):
        model_b_parts.append(reconstructed)
    model_b_episodes = pd.concat(model_b_parts, ignore_index=True)
    model_b_matrix, model_b_cached = load_or_build_matrix(
        "model_b_train",
        model_b_episodes,
        data,
        cache_root,
        True,
    )
    test_matrix, test_cached = load_or_build_matrix(
        "model_b_test_inputs",
        test_inputs,
        data,
        cache_root,
        False,
    )
    print(
        f"[features] model_b_pairs={len(model_b_matrix.labels)} "
        f"cache_hits={int(model_b_cached) + int(test_cached)}/2"
    )
    model_b_oof = crossfit_generative(model_b_matrix, generative_trees, debug)
    test_generative = fit_predict_generative(model_b_matrix, test_matrix, generative_trees)
    test_predictions, test_diagnostics = fit_final_predictions(
        model_b_matrix,
        test_matrix,
        model_b_oof,
        test_generative,
        selected,
        trees,
        seeds,
    )
    test_predictions = np.asarray(test_predictions, dtype=np.int64)
    validate_predictions(test_predictions, len(test_inputs))
    if not np.array_equal(validation_predictions, frozen_validation_predictions):
        raise RuntimeError("validation predictions changed after Model B")
    elapsed(started, "model_b")
    save_predictions(validation_predictions, test_predictions)
    metrics = {
        "debug": debug,
        "trees": trees,
        "generative_trees": generative_trees,
        "seeds": seeds,
        "selection": selected,
        "forward_fold_strata": fold_strata,
        "model_a": validation_diagnostics,
        "model_b": test_diagnostics,
        "model_a_episode_count": int(model_a_matrix.num_groups),
        "model_b_episode_count": int(model_b_matrix.num_groups),
        "feature_count": int(model_a_matrix.features.shape[1]),
        "elapsed_seconds": float(time.time() - started),
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    elapsed(started, "save")


if __name__ == "__main__":
    main()
