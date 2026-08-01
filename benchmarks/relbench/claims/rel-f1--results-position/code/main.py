from __future__ import annotations

import fcntl
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from f1_features import build_base_features, build_label_history, make_model_frame
from f1_models import (
    HeadPredictions,
    ModelSettings,
    fit_calibrators,
    fit_heads,
    forward_fold_years,
    generate_oof,
    predict_chain,
    select_transport,
)
from kapso_datasets.common import load_task, run_data_dir, save_predictions, shared_cache_dir


CACHE_VERSION = "lane1_f1_soft_transport_v5"


def elapsed(start: float, phase: str) -> None:
    print(f"[f1] phase={phase} elapsed_seconds={time.time() - start:.2f}", flush=True)


def register_artifact(path: Path, description: str, content_key: str) -> None:
    cache = shared_cache_dir()
    registry = cache / "artifacts.json"
    lock_path = cache / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(registry.read_text()) if registry.exists() else []
        relative = str(path.relative_to(cache))
        if not any(record.get("path") == relative for record in records):
            records.append({
                "name": path.stem,
                "path": relative,
                "description": description,
                "content_key": content_key,
                "rebuild_hint": "Run main.py in the matching debug or full mode.",
            })
            temporary = cache / f"artifacts.lane1.{os.getpid()}.json"
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def cached_oof(
    tag: str,
    features: pd.DataFrame,
    labels: np.ndarray,
    race_ids: np.ndarray,
    dates: pd.Series,
    categorical: list[str],
    settings: ModelSettings,
    fold_feature_factory,
) -> tuple[HeadPredictions, np.ndarray, bool]:
    mode = "debug" if settings.debug else "full"
    path = shared_cache_dir() / f"{CACHE_VERSION}_{tag}_{mode}.npz"
    if path.exists():
        payload = np.load(path, allow_pickle=False)
        stored_thresholds = tuple(payload["thresholds"].astype(int).tolist())
        if len(payload["l2"]) == len(labels) and stored_thresholds == settings.thresholds:
            predictions = HeadPredictions(payload["l2"], payload["ordinal"], payload["ranker"])
            return predictions, payload["fold_ids"], True
    fold_features, fold_categorical = fold_feature_factory()
    predictions, fold_ids = generate_oof(
        features,
        labels,
        race_ids,
        dates,
        categorical,
        settings,
        fold_features,
        fold_categorical,
    )
    temporary = path.with_suffix(f".{os.getpid()}.npz")
    np.savez_compressed(
        temporary,
        l2=predictions.l2,
        ordinal=predictions.ordinal,
        ranker=predictions.ranker,
        fold_ids=fold_ids,
        thresholds=np.asarray(settings.thresholds),
    )
    generated = Path(str(temporary) + ".npz") if not temporary.exists() else temporary
    os.replace(generated, path)
    register_artifact(
        path,
        f"Forward OOF L2, ordinal, and LambdaMART predictions for {tag} ({mode}).",
        f"{CACHE_VERSION}:{tag}:{mode}",
    )
    return predictions, fold_ids, False


def task_queries(context) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = context.train.df.copy().reset_index(drop=True)
    val = context.val.df.copy().reset_index(drop=True)
    test = context.test.df.copy().reset_index(drop=True)
    parts = []
    for split, frame in [("train", train), ("val", val), ("test", test)]:
        current = frame.copy()
        current["_split"] = split
        current["_row_idx"] = np.arange(len(current), dtype=int)
        parts.append(current)
    query = pd.concat(parts, ignore_index=True, sort=False)
    query["_query_id"] = np.arange(len(query), dtype=int)
    return train, val, test, query


def oof_feature_factory(
    base: pd.DataFrame,
    labeled_indices: np.ndarray,
    labels: pd.DataFrame,
    dates: pd.Series,
):
    def build():
        subset = base.iloc[labeled_indices].reset_index(drop=True)
        matrices = []
        categoricals = []
        for lower in forward_fold_years(dates):
            allowed = labels.loc[labels["date"].dt.year < lower].reset_index(drop=True)
            history = build_label_history(subset, allowed)
            matrix, categorical = make_model_frame(subset, history)
            matrices.append(matrix)
            categoricals.append(categorical)
        return matrices, categoricals
    return build


def main() -> None:
    warnings.filterwarnings("ignore")
    start = time.time()
    debug = "--debug" in sys.argv
    thresholds = (1, 2, 3, 4, 5, 6) if debug else (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24)
    seeds = (17,) if debug else (17, 43, 89)
    settings = ModelSettings(debug, thresholds, seeds, 20 if debug else 50)
    context = load_task(upto_test_timestamp=False)
    train, val, test, query = task_queries(context)
    tables = {name: table.df.copy() for name, table in context.db.table_dict.items()}
    elapsed(start, "load")

    base = build_base_features(query, tables).sort_values("_query_id", kind="stable").reset_index(drop=True)
    train_labels = train[["date", "resultId", context.target_col]].copy()
    full_labels = pd.concat(
        [train_labels, val[["date", "resultId", context.target_col]]],
        ignore_index=True,
    ) if len(val) else train_labels.copy()
    train_history = build_label_history(base, train_labels)
    full_history = build_label_history(base, full_labels)
    train_features, categorical = make_model_frame(base, train_history)
    full_features, full_categorical = make_model_frame(base, full_history)
    if train_features.columns.tolist() != full_features.columns.tolist() or categorical != full_categorical:
        raise RuntimeError("feature chains do not have identical schemas")
    elapsed(start, f"features columns={train_features.shape[1]}")

    train_indices = np.flatnonzero(base["_split"].to_numpy() == "train")
    val_indices = np.flatnonzero(base["_split"].to_numpy() == "val")
    test_indices = np.flatnonzero(base["_split"].to_numpy() == "test")
    train_y = base.loc[train_indices, context.target_col].to_numpy(dtype=float)
    train_races = base.loc[train_indices, "raceId"].to_numpy()
    train_dates = base.loc[train_indices, "date"].reset_index(drop=True)

    train_oof, train_fold_ids, train_cache_hit = cached_oof(
        "train",
        train_features.iloc[train_indices].reset_index(drop=True),
        train_y,
        train_races,
        train_dates,
        categorical,
        settings,
        oof_feature_factory(base, train_indices, train_labels, train_dates),
    )
    train_calibrators = fit_calibrators(train_oof, train_y, train_races, settings.thresholds)
    selection = select_transport(
        train_oof,
        train_y,
        train_races,
        train_fold_ids,
        settings.thresholds,
        train_calibrators,
        settings.sinkhorn_iterations,
    )
    elapsed(start, f"train_oof cache_hit={train_cache_hit} selected_blend={selection.config.blend}")

    validation_heads = fit_heads(
        train_features.iloc[train_indices],
        train_y,
        train_races,
        categorical,
        settings,
    )
    if len(val_indices):
        val_predictions, val_marginal_error = predict_chain(
            validation_heads,
            train_features.iloc[val_indices],
            base.loc[val_indices, "raceId"].to_numpy(),
            settings.thresholds,
            train_calibrators,
            selection.config,
            settings.sinkhorn_iterations,
        )
    else:
        val_predictions = np.empty(0, dtype=float)
        val_marginal_error = 0.0
    elapsed(start, "validation_chain")

    if len(val_indices):
        labeled_indices = np.concatenate([train_indices, val_indices])
        full_y = base.loc[labeled_indices, context.target_col].to_numpy(dtype=float)
        full_races = base.loc[labeled_indices, "raceId"].to_numpy()
        full_dates = base.loc[labeled_indices, "date"].reset_index(drop=True)
        del validation_heads
        gc.collect()
        full_oof, full_fold_ids, full_cache_hit = cached_oof(
            "train_val",
            full_features.iloc[labeled_indices].reset_index(drop=True),
            full_y,
            full_races,
            full_dates,
            categorical,
            settings,
            oof_feature_factory(base, labeled_indices, full_labels, full_dates),
        )
        full_calibrators = fit_calibrators(full_oof, full_y, full_races, settings.thresholds)
        test_heads = fit_heads(
            full_features.iloc[labeled_indices],
            full_y,
            full_races,
            categorical,
            settings,
        )
        test_predictions, test_marginal_error = predict_chain(
            test_heads,
            full_features.iloc[test_indices],
            base.loc[test_indices, "raceId"].to_numpy(),
            settings.thresholds,
            full_calibrators,
            selection.config,
            settings.sinkhorn_iterations,
        )
    else:
        full_cache_hit = train_cache_hit
        test_predictions, test_marginal_error = predict_chain(
            validation_heads,
            train_features.iloc[test_indices],
            base.loc[test_indices, "raceId"].to_numpy(),
            settings.thresholds,
            train_calibrators,
            selection.config,
            settings.sinkhorn_iterations,
        )
    elapsed(start, f"test_chain cache_hit={full_cache_hit}")

    val_predictions = np.asarray(val_predictions, dtype=float)[np.argsort(base.loc[val_indices, "_row_idx"].to_numpy())]
    test_predictions = np.asarray(test_predictions, dtype=float)[np.argsort(base.loc[test_indices, "_row_idx"].to_numpy())]
    if val_predictions.shape != (len(val),) or test_predictions.shape != (len(test),):
        raise RuntimeError("prediction shape mismatch before save")
    if not np.all(np.isfinite(val_predictions)) or not np.all(np.isfinite(test_predictions)):
        raise RuntimeError("non-finite predictions before save")
    save_predictions(val_predictions, test_predictions)
    diagnostics = {
        "mode": "debug" if debug else "full",
        "elapsed_seconds": time.time() - start,
        "feature_count": int(train_features.shape[1]),
        "thresholds": list(settings.thresholds),
        "seeds": list(settings.seeds),
        "selection": {
            "pmf_mixture": selection.config.pmf_mixture,
            "entropy": selection.config.entropy,
            "rank_width": selection.config.rank_width,
            "blend": selection.config.blend,
            "baseline_r2": selection.baseline_r2,
            "selected_r2": selection.selected_r2,
            "raw_transport_r2": selection.raw_transport_r2,
            "selection_marginal_error": selection.marginal_error,
            "fold_metrics": selection.fold_metrics,
        },
        "validation_marginal_error": val_marginal_error,
        "test_marginal_error": test_marginal_error,
        "validation_chain_labels": "train_only",
        "test_chain_labels": "train_plus_validation" if len(val) else "train_only_rolling",
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[f1] diagnostics={json.dumps(diagnostics, separators=(',', ':'))}", flush=True)


if __name__ == "__main__":
    main()
