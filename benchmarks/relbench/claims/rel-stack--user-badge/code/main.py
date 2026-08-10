from __future__ import annotations

import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from chronograph import build_graph_events, replay_memories, train_memory
from extra_features import build_extra_features, combine_expanded
from mechanism_features import build_mechanism_features
from modeling import fit_gbdt, fit_hazard, fit_meta, predict_gbdt, predict_hazard, predict_meta, train_oof
from temporal_features import build_compact_features, build_hazard_bins, feature_names, load_users


# Runtime

def phase(name: str, started: float) -> None:
    print(f"[runtime] phase={name} elapsed_seconds={time.time() - started:.1f}", flush=True)


def output_directory() -> Path:
    path = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_3"))
    path.mkdir(parents=True, exist_ok=True)
    return path


# Diagnostics

def safe_auc(target: np.ndarray, prediction: np.ndarray) -> float | None:
    if len(np.unique(target)) < 2:
        return None
    return float(roc_auc_score(target, prediction))


def validation_diagnostics(target: np.ndarray, prediction: np.ndarray, neural: np.ndarray, tree: np.ndarray, compact: np.ndarray, start: int, names: list[str]) -> dict:
    rng = np.random.default_rng(1337)
    bootstrap = []
    for _ in range(100):
        sample = rng.integers(0, len(target), len(target))
        bootstrap.append(roc_auc_score(target[sample], prediction[sample]))
    prior = np.asarray(compact[start : start + len(target), names.index("lifetime_g7")], dtype=np.float32)
    activity = np.zeros(len(target), dtype=np.float32)
    for group in range(7):
        activity += np.expm1(np.asarray(compact[start : start + len(target), names.index(f"lifetime_g{group}")], dtype=np.float32))
    age = np.asarray(compact[start : start + len(target), names.index("account_age_days")], dtype=np.float32)
    strata = {}
    masks = {
        "activity_none": activity == 0,
        "activity_sparse": (activity > 0) & (activity <= 5),
        "activity_rich": activity > 5,
        "prior_badge_none": prior == 0,
        "prior_badge_any": prior > 0,
        "account_age_lt_1y": age < 365,
        "account_age_1_to_4y": (age >= 365) & (age < 1460),
        "account_age_ge_4y": age >= 1460,
    }
    for name, mask in masks.items():
        strata[name] = {"count": int(mask.sum()), "roc_auc": safe_auc(target[mask], prediction[mask])}
    correlations = {
        "neural_gbdt": float(spearmanr(neural, tree).statistic),
        "neural_final": float(spearmanr(neural, prediction).statistic),
        "gbdt_final": float(spearmanr(tree, prediction).statistic),
    }
    return {
        "validation_roc_auc": float(roc_auc_score(target, prediction)),
        "bootstrap_auc_standard_error": float(np.std(bootstrap, ddof=1)),
        "bootstrap_auc_95_interval": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "prediction_rank_correlations": correlations,
        "strata": strata,
    }


# Entrypoint

def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(1337)
    random.seed(1337)
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1337)
    debug = "--debug" in sys.argv
    started = time.time()
    from relbench.tasks import get_task

    task = get_task(os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False)
    train = task.get_table("train").df.reset_index(drop=True)
    validation = task.get_table("val").df.reset_index(drop=True)
    test = task.get_table("test").df.reset_index(drop=True)
    target_column = task.target_col
    frames = [train[["timestamp", "UserId"]], validation[["timestamp", "UserId"]], test[["timestamp", "UserId"]]]
    phase("task_load", started)

    compact, names, offsets, mapped_users = build_compact_features(frames, debug=debug)
    extra = build_extra_features(frames, compact, names, mapped_users)
    compact_extra = combine_expanded(compact, extra, None, "compact_extra_v2.npy")
    mechanism = build_mechanism_features(frames, mapped_users)
    phase("compact_features", started)

    users = load_users()
    events, post_count = build_graph_events(users["Id"].to_numpy(dtype=np.int64))
    train_origins = np.unique(train["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64))
    pretrain_cutoff = int(train_origins[-2])
    graph = train_memory(events, len(users) + post_count, pretrain_cutoff, debug)
    memory = replay_memories(graph, events, len(users) + post_count, frames, mapped_users, debug)
    features = combine_expanded(compact_extra, mechanism, memory, "combined_debug_v3.npy" if debug else "combined_full_v3.npy")
    phase("graph_memory", started)

    bins = build_hazard_bins([train, validation])
    official = np.concatenate([train[target_column].to_numpy(dtype=np.uint8), validation[target_column].to_numpy(dtype=np.uint8)])
    mismatch = int(np.sum((bins < 13).astype(np.uint8) != official))
    if mismatch:
        raise RuntimeError(f"hazard-label mismatch: {mismatch} rows")
    n_train = len(train)
    n_validation = len(validation)
    n_test = len(test)
    train_indices = np.arange(n_train, dtype=np.int64)
    validation_indices = np.arange(n_train, n_train + n_validation, dtype=np.int64)
    test_indices = np.arange(n_train + n_validation, n_train + n_validation + n_test, dtype=np.int64)
    train_times = train["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)
    oof_neural, oof_tree, oof_target, oof_metrics = train_oof(features, bins[:n_train], train_times, debug)
    meta_a = fit_meta(oof_neural, oof_tree, oof_target)
    phase("purged_oof", started)

    supervised_train = train_indices[-250000:] if debug else train_indices
    hazard_a = fit_hazard(features, bins, supervised_train, "model_a", 1 if debug else 6, debug)
    tree_train = np.flatnonzero(np.isin(train_times, np.unique(train_times)[-16:]))
    if debug:
        tree_train = tree_train[-250000:]
    gbdt_a = fit_gbdt(features, bins, tree_train, "model_a", debug, 80 if debug else 500)
    validation_neural = predict_hazard(hazard_a, features, validation_indices)
    validation_tree = predict_gbdt(gbdt_a, features, validation_indices)
    validation_prediction = predict_meta(meta_a, validation_neural, validation_tree)
    validation_degree = np.zeros(n_validation, dtype=np.float32)
    for group in range(7):
        validation_degree += np.asarray(compact[validation_indices, names.index(f"lifetime_g{group}")], dtype=np.float32)
    validation_prediction[validation_degree == 0] = validation_tree[validation_degree == 0]
    validation_prediction = np.clip(validation_prediction, 1e-6, 1 - 1e-6)
    phase("model_a", started)

    supervised_test = np.arange(n_train + n_validation, dtype=np.int64)
    if debug:
        supervised_test = supervised_test[-300000:]
    hazard_b = fit_hazard(features, bins, supervised_test, "model_b", 1 if debug else 6, debug)
    labeled_times = np.concatenate([train_times, validation["timestamp"].to_numpy(dtype="datetime64[s]").astype(np.int64)])
    tree_test = np.flatnonzero(np.isin(labeled_times, np.unique(labeled_times)[-16:]))
    if debug:
        tree_test = tree_test[-300000:]
    gbdt_b = fit_gbdt(features, bins, tree_test, "model_b", debug, 80 if debug else 500)
    test_neural = predict_hazard(hazard_b, features, test_indices)
    test_tree = predict_gbdt(gbdt_b, features, test_indices)
    meta_b = fit_meta(
        np.concatenate([oof_neural, validation_neural]),
        np.concatenate([oof_tree, validation_tree]),
        np.concatenate([oof_target, validation[target_column].to_numpy(dtype=np.uint8)]),
    )
    test_prediction = predict_meta(meta_b, test_neural, test_tree)
    test_degree = np.zeros(n_test, dtype=np.float32)
    for group in range(7):
        test_degree += np.asarray(compact[test_indices, names.index(f"lifetime_g{group}")], dtype=np.float32)
    test_prediction[test_degree == 0] = test_tree[test_degree == 0]
    test_prediction = np.clip(test_prediction, 1e-6, 1 - 1e-6)
    phase("model_b", started)

    output = output_directory()
    np.save(output / "val_predictions.npy", validation_prediction.astype(np.float32))
    np.save(output / "test_predictions.npy", test_prediction.astype(np.float32))
    diagnostics = validation_diagnostics(
        validation[target_column].to_numpy(dtype=np.uint8),
        validation_prediction,
        validation_neural,
        validation_tree,
        compact,
        n_train,
        names,
    )
    diagnostics["oof"] = oof_metrics
    diagnostics["hazard_label_mismatch"] = mismatch
    diagnostics["graph_backend"] = "torch_geometric" if __import__("chronograph").GRAPH_BACKEND == "pyg" else "torch"
    diagnostics["elapsed_seconds"] = time.time() - started
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[diagnostics] {json.dumps(diagnostics, separators=(',', ':'))}", flush=True)
    print(f"[output] val={validation_prediction.shape} test={test_prediction.shape} elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
