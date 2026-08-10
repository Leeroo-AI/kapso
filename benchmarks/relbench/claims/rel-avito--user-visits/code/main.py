from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from feature_pipeline import load_or_build_feature_frame, register_artifact
from kapso_datasets.common import load_task, run_data_dir, shared_cache_dir
from model_pipeline import choose_feature_set, choose_model, debug_predictions, fit_chain


def align_predictions(table: pd.DataFrame, feature_rows: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:
    official = table[["UserID", "timestamp"]].copy()
    official["row_id"] = np.arange(len(official), dtype=np.int64)
    lookup = feature_rows[["UserID", "origin"]].copy()
    lookup["prediction"] = predictions
    aligned = official.merge(
        lookup, left_on=["UserID", "timestamp"], right_on=["UserID", "origin"],
        how="left", sort=False, validate="one_to_one",
    ).sort_values("row_id")
    if aligned["prediction"].isna().any():
        raise RuntimeError(f"Missing predictions for {int(aligned['prediction'].isna().sum())} official rows")
    return aligned["prediction"].to_numpy(dtype=np.float64)


def validate_arrays(validation: np.ndarray, test: np.ndarray, expected_validation: int, expected_test: int) -> tuple[np.ndarray, np.ndarray]:
    validation = np.clip(np.asarray(validation, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    test = np.clip(np.asarray(test, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    if validation.shape != (expected_validation,) or test.shape != (expected_test,):
        raise RuntimeError(f"Prediction shapes {validation.shape} and {test.shape} violate the contract")
    if not np.isfinite(validation).all() or not np.isfinite(test).all():
        raise RuntimeError("Predictions contain non-finite values")
    return validation, test


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    context = load_task()
    output = run_data_dir()
    cache = shared_cache_dir()
    frame, feature_timings = load_or_build_feature_frame(debug, cache)
    print("[phase] " + json.dumps({"features": feature_timings, "elapsed_seconds": time.time() - started}, separators=(",", ":")), flush=True)
    diagnostics: dict[str, object] = {"feature_timings": feature_timings}
    if debug:
        validation_raw, test_raw, debug_diagnostics = debug_predictions(frame)
        validation_rows = frame.loc[frame["source"] == "validation"]
        test_rows = frame.loc[frame["source"] == "test"]
        validation = align_predictions(context.val.df, validation_rows, validation_raw)
        test = align_predictions(context.test.df, test_rows, test_raw)
        diagnostics.update(debug_diagnostics)
    else:
        feature_started = time.time()
        columns, foundation, feature_diagnostics = choose_feature_set(frame)
        foundation_cache = cache / "lane0_stage1_foundation_cv_v15.npz"
        if not foundation_cache.exists():
            np.savez_compressed(
                foundation_cache,
                labels=np.concatenate([foundation.labels[origin] for origin in sorted(foundation.labels)]),
                predictions=np.concatenate([foundation.predictions[origin] for origin in sorted(foundation.predictions)]),
            )
        register_artifact(
            cache,
            foundation_cache,
            name="lane0_stage1_foundation_cv_v9",
            description="Purged-forward out-of-sample labels and predictions from the banked Stage 1 LightGBM foundation",
            content_key="lane0_stage1_foundation_cv_v9",
        )
        print(f"[phase] feature_selection_seconds={time.time() - feature_started:.3f}", flush=True)
        model_started = time.time()
        model_choice, iterations, model_diagnostics = choose_model(frame, columns, foundation)
        print(f"[phase] model_selection_seconds={time.time() - model_started:.3f}", flush=True)
        chain_a_started = time.time()
        validation_raw = fit_chain(
            frame, columns, model_choice, iterations,
            train_end=pd.Timestamp("2015-05-04"), predict_origin=pd.Timestamp("2015-05-08"),
        )
        validation_rows = frame.loc[frame["origin"] == pd.Timestamp("2015-05-08")]
        validation = align_predictions(context.val.df, validation_rows, validation_raw)
        validation = np.clip(validation, 1e-5, 1.0 - 1e-5)
        if validation.shape != (len(context.val.df),) or not np.isfinite(validation).all():
            raise RuntimeError("Chain A validation predictions violate the contract")
        np.save(output / "val_predictions.npy", validation)
        print(f"[phase] chain_a_seconds={time.time() - chain_a_started:.3f} validation_frozen=true", flush=True)
        chain_b_started = time.time()
        test_raw = fit_chain(
            frame, columns, model_choice, iterations,
            train_end=pd.Timestamp("2015-05-10"), predict_origin=pd.Timestamp("2015-05-14"),
        )
        test_rows = frame.loc[frame["source"] == "test"]
        test = align_predictions(context.test.df, test_rows, test_raw)
        print(f"[phase] chain_b_seconds={time.time() - chain_b_started:.3f}", flush=True)
        diagnostics.update(
            {
                "feature_selection": feature_diagnostics,
                "model_selection": model_diagnostics,
                "selected_feature_count": len(columns),
                "selected_model": model_choice,
                "iterations": iterations,
            }
        )
    validation, test = validate_arrays(validation, test, len(context.val.df), len(context.test.df))
    np.save(output / "val_predictions.npy", validation)
    np.save(output / "test_predictions.npy", test)
    diagnostics["elapsed_seconds"] = time.time() - started
    diagnostics["validation_prediction_std"] = float(np.std(validation))
    diagnostics["test_prediction_std"] = float(np.std(test))
    (output / "metrics.json").write_text(json.dumps(diagnostics, default=str, sort_keys=True))
    print(
        f"[complete] val={validation.shape} test={test.shape} elapsed_seconds={time.time() - started:.3f} "
        f"val_std={np.std(validation):.6f} test_std={np.std(test):.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
