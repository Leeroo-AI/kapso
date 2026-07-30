from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from rel_f1_model import FeatureBuilder, build_cached_features, fit_predict, write_metrics
from tabpfn_residual import (
    append_crossfit_predictions,
    apply_residual_specialist,
    build_forward_oof,
    configure_environment,
    ensure_checkpoint,
    promotion_enabled,
    select_context_columns,
)


warnings.filterwarnings("ignore")


def _load():
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)
    train = task.get_table("train", mask_input_cols=False).df.copy()
    val = task.get_table("val", mask_input_cols=False).df.copy()
    test = task.get_table("test").df.copy()
    return task, db, train, val, test


def _save(path: Path, name: str, values: np.ndarray) -> None:
    np.save(path / name, np.asarray(values, dtype=np.float64))


def main() -> None:
    started = time.perf_counter()
    debug = "--debug" in sys.argv
    output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    cache_directory = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "output_data_generic_exp_4"))
    checkpoint = ensure_checkpoint(configure_environment(cache_directory, output))
    task, db, train, val, test = _load()
    print(
        f"[load] train={len(train)} val={len(val)} test={len(test)} "
        f"db_max={db.max_timestamp} elapsed={time.perf_counter() - started:.2f}s"
    )
    rolling = len(val) == 0
    labels = train[["date", "driverId", task.target_col]].copy()
    builder = FeatureBuilder(db, labels)
    if rolling and debug:
        latest = pd.to_datetime(train["date"]).max()
        model_train = train.loc[pd.to_datetime(train["date"]) >= latest - pd.DateOffset(years=20)].copy()
    else:
        model_train = train
    train_features = build_cached_features(builder, model_train, cache_directory, enabled=rolling)
    test_features = build_cached_features(builder, test, cache_directory, enabled=rolling)
    print(
        f"[features] columns={train_features.shape[1]} train={train_features.shape} "
        f"test={test_features.shape} elapsed={time.perf_counter() - started:.2f}s"
    )
    train_target = model_train[task.target_col].to_numpy(dtype=np.float64)
    train_oof = build_forward_oof(
        model_train,
        train_features,
        train_target,
        cache_directory,
        debug=debug,
    )
    selected = select_context_columns(
        train_features,
        train_target,
        model_train["date"],
        cache_directory,
    )
    enabled = promotion_enabled(cache_directory)
    execution_enabled = enabled or debug
    diagnostics = {
        "rolling": rolling,
        "debug": debug,
        "train_rows": len(train),
        "test_rows": len(test),
        "feature_count": train_features.shape[1],
        "context_feature_count": len(selected) + 4,
        "residual_specialist_enabled": enabled,
        "residual_specialist_exercised": execution_enabled,
    }
    if rolling:
        test_champion, test_parts = fit_predict(
            train_features,
            train_target,
            test_features,
            debug=debug,
        )
        test_prediction, specialist = apply_residual_specialist(
            model_train,
            train_features,
            train_target,
            train_oof,
            test_features,
            test_champion,
            test_parts,
            selected,
            checkpoint,
            debug,
            execution_enabled,
        )
        _save(output, "test_predictions.npy", test_prediction)
        diagnostics["specialist"] = specialist
    else:
        val_features = builder.build(val)
        val_champion, val_parts = fit_predict(
            train_features,
            train_target,
            val_features,
            debug=debug,
        )
        val_prediction, val_specialist = apply_residual_specialist(
            model_train,
            train_features,
            train_target,
            train_oof,
            val_features,
            val_champion,
            val_parts,
            selected,
            checkpoint,
            debug,
            execution_enabled,
        )
        _save(output, "val_predictions.npy", val_prediction)
        combined = pd.concat([train, val], ignore_index=True)
        combined_labels = combined[["date", "driverId", task.target_col]].copy()
        combined_builder = FeatureBuilder(db, combined_labels)
        combined_features = combined_builder.build(combined)
        combined_test_features = combined_builder.build(test)
        combined_target = combined[task.target_col].to_numpy(dtype=np.float64)
        test_champion, test_parts = fit_predict(
            combined_features,
            combined_target,
            combined_test_features,
            debug=debug,
        )
        combined_oof = append_crossfit_predictions(
            train_oof,
            val,
            val_champion,
            val_parts,
        )
        test_prediction, test_specialist = apply_residual_specialist(
            combined,
            combined_features,
            combined_target,
            combined_oof,
            combined_test_features,
            test_champion,
            test_parts,
            selected,
            checkpoint,
            debug,
            execution_enabled,
        )
        _save(output, "test_predictions.npy", test_prediction)
        diagnostics["val_rows"] = len(val)
        diagnostics["val_prediction_mean"] = float(val_prediction.mean())
        diagnostics["test_refit_rows"] = len(combined)
        diagnostics["val_specialist"] = val_specialist
        diagnostics["test_specialist"] = test_specialist
    diagnostics["prediction_mean"] = float(test_prediction.mean())
    diagnostics["prediction_std"] = float(test_prediction.std())
    diagnostics["component_means"] = {name: float(values.mean()) for name, values in test_parts.items()}
    diagnostics["elapsed_seconds"] = time.perf_counter() - started
    write_metrics(output / "metrics.json", diagnostics)
    print(
        f"[save] rolling={rolling} test={test_prediction.shape} "
        f"elapsed={time.perf_counter() - started:.2f}s"
    )


if __name__ == "__main__":
    main()
