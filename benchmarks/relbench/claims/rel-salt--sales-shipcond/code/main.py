from __future__ import annotations

import json
import gc
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from kapso_datasets.common import save_predictions
from rollforward import (
    RollforwardConfig,
    assemble_matrix,
    elapsed_message,
    load_or_build_bundle,
    make_labeled_features,
    predict_window,
    sample_chronological,
    save_diagnostics,
    select_rollforward,
    stable_top_classes,
    train_booster,
    update_true_history,
)


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = "--debug" in sys.argv
    seed = 1337
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    task = get_task(dataset_name, task_name, download=False)
    dataset = get_dataset(dataset_name, download=False)
    cache_root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    bundle = load_or_build_bundle(task, dataset, cache_root)
    elapsed_message(started, "relational features")
    train_rows_full = bundle.split_rows["train"]
    train_labels_full = bundle.split_labels["train"]
    val_rows = bundle.split_rows["val"]
    val_labels = bundle.split_labels["val"]
    test_rows = bundle.split_rows["test"]
    top_classes = stable_top_classes(train_labels_full)
    if debug:
        train_rows = sample_chronological(train_rows_full, bundle.days, 25000)
        train_positions = np.searchsorted(train_rows_full, train_rows)
        train_labels = train_labels_full[train_positions]
    else:
        train_rows = train_rows_full
        train_labels = train_labels_full
    train_dynamic, train_history = make_labeled_features(bundle, train_rows, train_labels, top_classes)
    train_matrix = assemble_matrix(bundle, train_rows, train_dynamic)
    elapsed_message(started, f"causal train matrix {train_matrix.shape}")
    simulation_mask = bundle.timestamps_ns[train_rows] < np.datetime64("2019-09-01").astype("datetime64[ns]").astype(np.int64)
    simulation_fit_mask = bundle.timestamps_ns[train_rows] < np.datetime64("2019-08-01").astype("datetime64[ns]").astype(np.int64)
    simulation_eval_mask = simulation_mask & ~simulation_fit_mask
    if debug or simulation_fit_mask.sum() < 1000 or simulation_eval_mask.sum() < 100:
        split_index = max(int(len(train_rows) * 0.8), 1)
        simulation_fit_indices = np.arange(split_index)
        simulation_eval_indices = np.arange(split_index, len(train_rows))
    else:
        simulation_fit_indices = np.flatnonzero(simulation_fit_mask)
        simulation_eval_indices = np.flatnonzero(simulation_eval_mask)
    max_rounds = 40 if debug else 1000
    simulation_booster = train_booster(
        train_matrix[simulation_fit_indices],
        train_labels[simulation_fit_indices],
        max_rounds,
        seed,
        None if debug else train_matrix[simulation_eval_indices],
        None if debug else train_labels[simulation_eval_indices],
    )
    if debug:
        selected_rounds = 40
    else:
        selected_rounds = max(40, min(1000, int(simulation_booster.best_iteration) + 1))
    elapsed_message(started, f"simulation model rounds={selected_rounds}")
    simulation_window_mask = (
        (bundle.timestamps_ns[train_rows_full] >= np.datetime64("2019-09-01").astype("datetime64[ns]").astype(np.int64))
        & (bundle.timestamps_ns[train_rows_full] < np.datetime64("2020-02-01").astype("datetime64[ns]").astype(np.int64))
    )
    simulation_rows_full = train_rows_full[simulation_window_mask]
    simulation_labels_full = train_labels_full[simulation_window_mask]
    if debug:
        simulation_rows = sample_chronological(simulation_rows_full, bundle.days, 5000)
        simulation_label_positions = np.searchsorted(simulation_rows_full, simulation_rows)
        simulation_labels = simulation_labels_full[simulation_label_positions]
        history_source_mask = bundle.timestamps_ns[train_rows] < np.datetime64("2019-09-01").astype("datetime64[ns]").astype(np.int64)
        history_rows = train_rows[history_source_mask]
        history_labels = train_labels[history_source_mask]
    else:
        simulation_rows = simulation_rows_full
        simulation_labels = simulation_labels_full
        history_source_mask = bundle.timestamps_ns[train_rows_full] < np.datetime64("2019-09-01").astype("datetime64[ns]").astype(np.int64)
        history_rows = train_rows_full[history_source_mask]
        history_labels = train_labels_full[history_source_mask]
    simulation_history = update_true_history(bundle, history_rows, history_labels)
    selected_config, simulation_diagnostics = select_rollforward(
        simulation_booster,
        bundle,
        simulation_rows,
        simulation_labels,
        simulation_history,
        top_classes,
        debug,
    )
    if debug:
        selected_config = RollforwardConfig(7, selected_config.weight_function, selected_config.blend_strength, selected_config.enabled)
    print(f"[simulation] {json.dumps(simulation_diagnostics, sort_keys=True)}")
    if "--simulation-only" in sys.argv:
        simulation_only_diagnostics = {
            "selected_rounds": selected_rounds,
            "top_classes": top_classes.tolist(),
            "simulation": simulation_diagnostics,
            "elapsed_seconds": time.time() - started,
        }
        save_diagnostics(Path("output_data_generic_exp_3"), simulation_only_diagnostics)
        elapsed_message(started, "simulation-only complete")
        return
    del simulation_booster, simulation_history
    gc.collect()
    elapsed_message(started, f"rollforward gate enabled={selected_config.enabled} block_days={selected_config.block_days}")
    model_a = train_booster(train_matrix, train_labels, selected_rounds, seed + 1)
    elapsed_message(started, "model A")
    val_predictions = predict_window(model_a, bundle, val_rows, train_history, top_classes, selected_config)
    del model_a
    gc.collect()
    elapsed_message(started, "validation one-pass rollforward")
    if debug:
        model_b_val_rows = sample_chronological(val_rows, bundle.days, 5000)
        model_b_val_positions = np.searchsorted(val_rows, model_b_val_rows)
        model_b_val_labels = val_labels[model_b_val_positions]
        model_b_train_keep = np.arange(max(0, len(train_rows) - 20000), len(train_rows))
    else:
        model_b_val_rows = val_rows
        model_b_val_labels = val_labels
        model_b_train_keep = np.arange(len(train_rows))
    val_dynamic_for_b, test_history = make_labeled_features(
        bundle,
        model_b_val_rows,
        model_b_val_labels,
        top_classes,
        train_history,
    )
    val_matrix_for_b = assemble_matrix(bundle, model_b_val_rows, val_dynamic_for_b)
    model_b_matrix = __import__("scipy").sparse.vstack(
        [train_matrix[model_b_train_keep], val_matrix_for_b], format="csr"
    )
    model_b_labels = np.concatenate([train_labels[model_b_train_keep], model_b_val_labels])
    model_b = train_booster(model_b_matrix, model_b_labels, selected_rounds, seed + 2)
    elapsed_message(started, "model B")
    test_predictions = predict_window(model_b, bundle, test_rows, test_history, top_classes, selected_config)
    elapsed_message(started, "test one-pass rollforward")
    save_predictions(val_predictions, test_predictions)
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_3"))
    diagnostics = {
        "debug": debug,
        "selected_rounds": selected_rounds,
        "top_classes": top_classes.tolist(),
        "simulation": simulation_diagnostics,
        "fit_provenance": {
            "validation_predictions": "model_a_train_labels_only",
            "test_predictions": "model_b_train_plus_validation_labels",
        },
        "elapsed_seconds": time.time() - started,
    }
    save_diagnostics(output_dir, diagnostics)
    print(f"[result] {json.dumps(diagnostics['fit_provenance'], sort_keys=True)}")
    elapsed_message(started, "complete")


if __name__ == "__main__":
    main()
