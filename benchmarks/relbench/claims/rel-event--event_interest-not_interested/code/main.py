from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from relbench.tasks import get_task
from sklearn.metrics import roc_auc_score

from feature_pipeline import build_direct_bundle, build_dynamic_bundle
from graph_pipeline import build_friendship_graph, build_seed_frame
from modeling import (
    debug_selection,
    fit_two_model_chains,
    make_expanding_folds,
    report_slices,
    select_heads,
)


def _validation_slices(labels, prediction, masks):
    result = {}
    for name, mask in masks.items():
        current_labels = labels[mask]
        current_prediction = prediction[mask]
        score = None
        if len(current_labels) and np.unique(current_labels).size == 2:
            score = float(roc_auc_score(current_labels, current_prediction))
        result[name] = {
            "count": int(len(current_labels)),
            "positives": int(current_labels.sum()) if len(current_labels) else 0,
            "roc_auc": score,
        }
    print(f"[validation_slices] {json.dumps(result, sort_keys=True)}")
    return result


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = "--debug" in sys.argv
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    cache_dir = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    task = get_task(dataset_name, task_name, download=False)
    cutoff_db = task.dataset.get_db(upto_test_timestamp=True)
    full_db = task.dataset.get_db(upto_test_timestamp=False)
    seeds = build_seed_frame(task, cutoff_db, full_db)
    print(
        f"[phase] load_and_map rows={len(seeds)} debug={debug} "
        f"seconds={time.time() - started:.1f}"
    )
    users = full_db.table_dict["users"].df
    friends = full_db.table_dict["user_friends"].df
    graph = build_friendship_graph(users, friends, cache_dir, debug)
    direct = build_direct_bundle(seeds, full_db, graph)
    split = seeds["split"].to_numpy()
    train_global = np.flatnonzero(split == "train")
    validation_global = np.flatnonzero(split == "val")
    test_global = np.flatnonzero(split == "test")
    labels = seeds["label"].to_numpy(dtype=np.float32)
    train_allowed = split == "train"
    train_validation_allowed = (split == "train") | (split == "val")
    dynamic_a = build_dynamic_bundle(seeds, direct, graph, train_allowed, debug)
    dynamic_b = build_dynamic_bundle(seeds, direct, graph, train_validation_allowed, debug)
    full_a = np.concatenate([direct.matrix, direct.graph_static, dynamic_a.matrix], axis=1)
    full_b = np.concatenate([direct.matrix, direct.graph_static, dynamic_b.matrix], axis=1)
    topology = np.concatenate([direct.matrix, direct.graph_static[:, :10]], axis=1)
    compact_a = np.concatenate([direct.matrix, direct.graph_static[:, :10], dynamic_a.matrix], axis=1)
    compact_b = np.concatenate([direct.matrix, direct.graph_static[:, :10], dynamic_b.matrix], axis=1)
    train_labels = labels[train_global].astype(np.int32)
    if debug:
        selection = debug_selection()
        slice_report = {}
    else:
        train_times = seeds.iloc[train_global]["timestamp"].to_numpy(dtype="datetime64[ns]")
        folds = make_expanding_folds(train_times, train_labels)
        fold_full_validation = []
        fold_compact_validation = []
        for fold_number, (fold_train, fold_validation) in enumerate(folds):
            fold_allowed = np.zeros(len(seeds), dtype=bool)
            fold_allowed[train_global[fold_train]] = True
            fold_dynamic = build_dynamic_bundle(seeds, direct, graph, fold_allowed, False)
            validation_rows = train_global[fold_validation]
            fold_full_validation.append(
                np.concatenate(
                    [
                        direct.matrix[validation_rows],
                        direct.graph_static[validation_rows],
                        fold_dynamic.matrix[validation_rows],
                    ],
                    axis=1,
                )
            )
            fold_compact_validation.append(
                np.concatenate(
                    [
                        direct.matrix[validation_rows],
                        direct.graph_static[validation_rows, :10],
                        fold_dynamic.matrix[validation_rows],
                    ],
                    axis=1,
                )
            )
            print(
                f"[selection] fold_features={fold_number + 1}/{len(folds)} "
                f"train={len(fold_train)} valid={len(fold_validation)}"
            )
        selection = select_heads(
            direct.matrix[train_global],
            topology[train_global],
            compact_a[train_global],
            full_a[train_global],
            train_labels,
            folds,
            fold_compact_validation,
            fold_full_validation,
        )
        train_user = seeds.iloc[train_global]["user_idx"].to_numpy(dtype=np.int32)
        train_event_missing = seeds.iloc[train_global]["event_missing"].to_numpy(dtype=bool)
        train_timestamp = seeds.iloc[train_global]["timestamp"].to_numpy(dtype="datetime64[ns]")
        train_future = (~train_event_missing) & (direct.event_start[train_global] > train_timestamp)
        train_past = (~train_event_missing) & (direct.event_start[train_global] <= train_timestamp)
        train_cold = dynamic_a.cold_user[train_global]
        train_resolved = graph.und_degree[train_user] > 0
        slice_report = report_slices(
            train_labels,
            selection.oof_prediction,
            selection.oof_indices,
            {
                "resolved_neighbor": train_resolved,
                "no_resolved_neighbor": ~train_resolved,
                "cold_user": train_cold,
                "history_user": ~train_cold,
                "future_event": train_future,
                "past_start_event": train_past,
                "missing_event": train_event_missing,
            },
        )
        direct_slice_report = report_slices(
            train_labels,
            selection.direct_oof_prediction,
            selection.oof_indices,
            {
                "resolved_neighbor": train_resolved,
                "no_resolved_neighbor": ~train_resolved,
                "cold_user": train_cold,
                "history_user": ~train_cold,
                "future_event": train_future,
                "past_start_event": train_past,
                "missing_event": train_event_missing,
            },
        )
        print(f"[oof_direct_slices] {json.dumps(direct_slice_report, sort_keys=True)}")
    train_validation_global = np.concatenate([train_global, validation_global])
    train_validation_labels = labels[train_validation_global].astype(np.int32)
    feature_matrices_a = {
        "direct": direct.matrix,
        "topology": topology,
        "compact_graph": compact_a,
        "graph": full_a,
    }
    feature_matrices_b = {
        "direct": direct.matrix,
        "topology": topology,
        "compact_graph": compact_b,
        "graph": full_b,
    }
    selected_a = feature_matrices_a[selection.feature_set]
    selected_b = feature_matrices_b[selection.feature_set]
    validation_prediction, test_prediction = fit_two_model_chains(
        selected_a[train_global],
        train_labels,
        selected_a[validation_global],
        selected_b[train_validation_global],
        train_validation_labels,
        selected_b[test_global],
        selection,
    )
    validation_user = seeds.iloc[validation_global]["user_idx"].to_numpy(dtype=np.int32)
    validation_event_missing = seeds.iloc[validation_global]["event_missing"].to_numpy(dtype=bool)
    validation_timestamp = seeds.iloc[validation_global]["timestamp"].to_numpy(dtype="datetime64[ns]")
    validation_future = (~validation_event_missing) & (direct.event_start[validation_global] > validation_timestamp)
    validation_past = (~validation_event_missing) & (direct.event_start[validation_global] <= validation_timestamp)
    validation_cold = dynamic_a.cold_user[validation_global]
    validation_resolved = graph.und_degree[validation_user] > 0
    validation_slices = _validation_slices(
        labels[validation_global].astype(np.int32),
        validation_prediction,
        {
            "resolved_neighbor": validation_resolved,
            "no_resolved_neighbor": ~validation_resolved,
            "cold_user": validation_cold,
            "history_user": ~validation_cold,
            "future_event": validation_future,
            "past_start_event": validation_past,
            "missing_event": validation_event_missing,
        },
    )
    if validation_prediction.shape != (len(validation_global),):
        raise RuntimeError("validation output has incorrect shape")
    if test_prediction.shape != (len(test_global),):
        raise RuntimeError("test output has incorrect shape")
    if not np.isfinite(validation_prediction).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("predictions contain non-finite values")
    validation_prediction = np.clip(validation_prediction, 0, 1).astype(np.float64)
    test_prediction = np.clip(test_prediction, 0, 1).astype(np.float64)
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_3"))
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_predictions.npy", validation_prediction)
    np.save(output_dir / "test_predictions.npy", test_prediction)
    diagnostics = {
        "debug": debug,
        "elapsed_seconds": time.time() - started,
        "features": {
            "direct": len(direct.names),
            "static_graph": len(direct.graph_names),
            "dynamic_graph": len(dynamic_a.names),
            "total": full_a.shape[1],
        },
        "selection": {
            "logistic_c": selection.logistic_c,
            "head": selection.selected_head,
            "feature_set": selection.feature_set,
            "tree_iterations": selection.tree_iterations,
            "metrics": selection.metrics,
            "fold_metrics": selection.fold_metrics,
        },
        "oof_slices": slice_report,
        "validation_slices": validation_slices,
        "label_chains": {
            "validation_model": "train_labels_only",
            "test_model": "train_plus_validation_labels",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(
        f"[output] val={validation_prediction.shape} test={test_prediction.shape} "
        f"features={selection.feature_set} head={selection.selected_head} elapsed_seconds={time.time() - started:.1f}"
    )


if __name__ == "__main__":
    main()
