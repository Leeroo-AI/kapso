from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import (
    FEATURE_VERSION,
    add_target_history,
    cache_root,
    data_paths,
    ensure_outcomes,
    ensure_raw_features,
    load_outcomes,
    load_raw_features,
    register_artifact,
    validate_feature_rows,
)
from kapso_datasets.common import run_data_dir, save_predictions
from model_pipeline import GateResult, fit_predict_components, fit_selected, run_forward_gate
from recurrence import raw_recurrence, recurrence_features


MODEL_VERSION = "compound_lane3_backoff_v2"


def elapsed(start: float, phase: str) -> None:
    print(f"[timing] phase={phase} elapsed={time.time() - start:.1f}s")


def atomic_save(path: Path, values: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npy")
    np.save(tmp, values)
    os.replace(tmp, path)


def add_recurrence(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    if values.shape != (len(frame), 3):
        raise RuntimeError(f"recurrence shape {values.shape} does not match {len(frame)}")
    result = frame.copy()
    result["bgnbd_expected_count91"] = values[:, 0]
    result["bgnbd_probability_alive"] = values[:, 1]
    result["bgnbd_probability_no_review"] = values[:, 2]
    return result


def add_backoff_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for prefix in ("n", "spend", "unique_products"):
        result[f"{prefix}_block_31_91"] = result[f"{prefix}91"] - result[f"{prefix}30"]
        result[f"{prefix}_block_92_182"] = result[f"{prefix}182"] - result[f"{prefix}91"]
        result[f"{prefix}_block_183_365"] = result[f"{prefix}365"] - result[f"{prefix}182"]
    lifetime_price = result["price_mean_lifetime"]
    result["price_mean30_backoff"] = (result["spend30"] + 5.0 * lifetime_price) / (result["n30"] + 5.0)
    result["price_mean91_backoff"] = (result["spend91"] + 8.0 * lifetime_price) / (result["n91"] + 8.0)
    result["price_fast_slow_ratio"] = result["price_mean91_backoff"] / (lifetime_price + 1.0)
    result["rating_mean91_backoff"] = (result["rating_sum91"] + 8.0 * result["rating_mean_lifetime"]) / (result["n91"] + 8.0)
    result["verified_share91_backoff"] = (result["verified_sum91"] + 8.0 * result["verified_share_lifetime"]) / (result["n91"] + 8.0)
    result["text_mean91_backoff"] = (result["text_sum91"] + 8.0 * result["text_mean_lifetime"]) / (result["n91"] + 8.0)
    result["demand_mean91_backoff"] = (result["demand_sum91"] + 8.0 * result["demand_mean_lifetime"]) / (result["n91"] + 8.0)
    result["frequency_per_active_year"] = result["frequency"] * 365.25 / (result["age_days"] + 30.0)
    result["recency_gap_ratio"] = result["recency_days"] / (result["gap_mean"] + 1.0)
    result["gap_coefficient_variation"] = result["gap_sd"] / (result["gap_mean"] + 1.0)
    result["target_hist_recent_ratio"] = result["target_hist_last"] / (result["target_hist_mean"] + 5.0)
    result["target_hist_mean_backoff"] = (
        result["target_hist_n"] * result["target_hist_mean"] + 2.0 * result["spend91"]
    ) / (result["target_hist_n"] + 2.0)
    return result


def source_frame(frame: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    if not np.array_equal(frame["row_id"].to_numpy(), outcomes["row_id"].to_numpy()):
        raise RuntimeError("feature and outcome row IDs differ")
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "customer_id": frame["customer_id"].to_numpy(),
            "y91": outcomes["y91"].to_numpy(dtype=np.float64),
        }
    )


def gate_from_json(data: dict) -> GateResult:
    return GateResult(
        design=data["design"],
        temperature=float(data["temperature"]),
        fold_metrics=data["fold_metrics"],
        aggregate=data["aggregate"],
        slice_metrics=data["slice_metrics"],
    )


def gate_to_json(gate: GateResult) -> dict:
    return {
        "design": gate.design,
        "temperature": gate.temperature,
        "fold_metrics": gate.fold_metrics,
        "aggregate": gate.aggregate,
        "slice_metrics": gate.slice_metrics,
    }


def validate_predictions(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    prediction = np.asarray(values, dtype=np.float64).reshape(-1)
    if prediction.shape != (expected,):
        raise RuntimeError(f"{name} shape {prediction.shape} != {(expected,)}")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError(f"{name} contains non-finite predictions")
    if np.any(prediction < 0):
        raise RuntimeError(f"{name} contains negative predictions")
    return prediction


def register_banked_predictions(bank_val: Path, bank_test: Path) -> None:
    register_artifact(
        f"{FEATURE_VERSION}-{MODEL_VERSION}-model-a-validation",
        bank_val,
        "Frozen train-only Model A out-of-sample validation predictions",
        "Delete only the banked prediction files to refit after a model-code change",
    )
    register_artifact(
        f"{FEATURE_VERSION}-{MODEL_VERSION}-model-b-test",
        bank_test,
        "Frozen train-plus-validation Model B test predictions paired with Model A",
        "Delete only the banked prediction files to refit after a model-code change",
    )


def cheap_frame(table: pd.DataFrame) -> pd.DataFrame:
    timestamp = pd.to_datetime(table["timestamp"])
    customer = table["customer_id"].to_numpy(dtype=np.int64)
    result = pd.DataFrame(
        {
            "row_id": np.arange(len(table), dtype=np.int64),
            "timestamp": timestamp,
            "customer_id": customer,
            "customer_hash1": (customer % 1000003) / 1000003.0,
            "customer_hash2": ((customer * 2654435761) % 1000033) / 1000033.0,
            "database_age_years": (timestamp - pd.Timestamp("2008-01-01")).dt.days / 365.25,
            "month_sin": np.sin(2 * np.pi * timestamp.dt.month / 12.0),
            "month_cos": np.cos(2 * np.pi * timestamp.dt.month / 12.0),
            "frequency": 1.0,
            "frequency_after_first": 0.0,
            "age_days": 30.0,
            "recency_days": 1.0,
            "recurrence_rate91_raw": 1.5,
            "recurrence_alive_raw": 0.97,
            "recurrence_p0_raw": np.exp(-1.5),
        }
    )
    return result


def debug_main() -> None:
    start = time.time()
    paths = data_paths()
    train_table = pd.read_parquet(paths["train"])
    val_table = pd.read_parquet(paths["val"], columns=["timestamp", "customer_id"])
    test_table = pd.read_parquet(paths["test"], columns=["timestamp", "customer_id"])
    train_limit = min(80000, len(train_table))
    predict_limit = 25000
    train = cheap_frame(train_table.iloc[:train_limit].reset_index(drop=True))
    val_small = cheap_frame(val_table.iloc[:predict_limit].reset_index(drop=True))
    test_small = cheap_frame(test_table.iloc[:predict_limit].reset_index(drop=True))
    train = add_recurrence(train, raw_recurrence(train))
    val_small = add_recurrence(val_small, raw_recurrence(val_small))
    test_small = add_recurrence(test_small, raw_recurrence(test_small))
    y = train_table["ltv"].iloc[:train_limit].to_numpy(dtype=np.float32)
    approximate_count = np.where(y > 0, np.clip(np.rint(y / 15.0), 1, 20), 0).astype(np.int64)
    outcomes = pd.DataFrame({"row_id": np.arange(train_limit), "y91": y, "n91": approximate_count})
    val_components = fit_predict_components(train, outcomes, val_small, 24, 64, 1.0)
    test_components = fit_predict_components(train, outcomes, test_small, 24, 64, 1.0)
    val_prediction = np.zeros(len(val_table), dtype=np.float32)
    test_prediction = np.zeros(len(test_table), dtype=np.float32)
    val_prediction[: len(val_small)] = val_components["compound"]
    test_prediction[: len(test_small)] = test_components["compound"]
    save_predictions(val_prediction, test_prediction)
    metrics = {
        "mode": "debug",
        "supervised_rows": train_limit,
        "predicted_val_rows": len(val_small),
        "predicted_test_rows": len(test_small),
        "compound_samples": 64,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    elapsed(start, "debug_complete")


def full_main() -> None:
    started = time.time()
    paths = data_paths()
    train_size = len(pd.read_parquet(paths["train"], columns=["customer_id"]))
    val_size = len(pd.read_parquet(paths["val"], columns=["customer_id"]))
    test_size = len(pd.read_parquet(paths["test"], columns=["customer_id"]))
    bank = cache_root()
    bank_val = bank / f"{MODEL_VERSION}_model_a_val_predictions.npy"
    bank_test = bank / f"{MODEL_VERSION}_model_b_test_predictions.npy"
    bank_metrics = bank / f"{MODEL_VERSION}_model_metrics.json"
    if bank_val.exists() and bank_test.exists() and bank_metrics.exists():
        val_prediction = validate_predictions(np.load(bank_val, allow_pickle=False), val_size, "cached validation")
        test_prediction = validate_predictions(np.load(bank_test, allow_pickle=False), test_size, "cached test")
        save_predictions(val_prediction, test_prediction)
        (run_data_dir() / "metrics.json").write_text(bank_metrics.read_text())
        register_banked_predictions(bank_val, bank_test)
        print(f"[cache] loaded banked Model A/B predictions version={FEATURE_VERSION}")
        elapsed(started, "full_complete_cached")
        return
    train_raw = load_raw_features("train")
    val_raw = load_raw_features("val")
    validate_feature_rows(train_raw, train_size)
    validate_feature_rows(val_raw, val_size)
    elapsed(started, "rfm_train_val")
    train_outcomes = load_outcomes("train")
    if len(train_outcomes) != train_size:
        raise RuntimeError("train outcome row count mismatch")
    train_source = source_frame(train_raw, train_outcomes)
    train = add_target_history(train_raw, train_source)
    val = add_target_history(val_raw, train_source)
    train = add_recurrence(train, recurrence_features(train, "train"))
    val = add_recurrence(val, recurrence_features(val, "val"))
    train = add_backoff_features(train)
    val = add_backoff_features(val)
    elapsed(started, "recurrence_train_val")
    gate_path = bank / f"{MODEL_VERSION}_forward_gate.json"
    if gate_path.exists():
        gate = gate_from_json(json.loads(gate_path.read_text()))
        print(f"[gate] loaded selection design={gate.design} temperature={gate.temperature}")
    else:
        gate = run_forward_gate(train, train_outcomes, trees=320, samples=128, train_cap=700000)
        tmp = gate_path.with_name(f"{gate_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(gate_to_json(gate), indent=2))
        os.replace(tmp, gate_path)
    elapsed(started, "forward_gate")
    if bank_val.exists():
        val_prediction = validate_predictions(np.load(bank_val, allow_pickle=False), val_size, "banked Model A validation")
        model_a_diagnostics = {"loaded": True}
    else:
        val_prediction, model_a_diagnostics = fit_selected(
            train,
            train_outcomes,
            val,
            gate.design,
            gate.temperature,
            trees=760,
            samples=128,
        )
        val_prediction = validate_predictions(val_prediction, val_size, "Model A validation")
        atomic_save(bank_val, val_prediction)
    np.save(run_data_dir() / "val_predictions.npy", val_prediction)
    print(f"[model_a] cached validation predictions before loading validation outcomes shape={val_prediction.shape}")
    elapsed(started, "model_a")
    val_outcomes = load_outcomes("val")
    if len(val_outcomes) != val_size:
        raise RuntimeError("validation outcome row count mismatch")
    test_raw = load_raw_features("test")
    validate_feature_rows(test_raw, test_size)
    val_source = source_frame(val_raw, val_outcomes)
    combined_source = pd.concat([train_source, val_source], ignore_index=True)
    test = add_target_history(test_raw, combined_source)
    test = add_recurrence(test, recurrence_features(test, "test"))
    test = add_backoff_features(test)
    model_b_train = pd.concat([train, val], ignore_index=True)
    model_b_outcomes = pd.concat([train_outcomes, val_outcomes], ignore_index=True)
    model_b_outcomes["row_id"] = np.arange(len(model_b_outcomes), dtype=np.int64)
    elapsed(started, "model_b_features")
    if bank_test.exists():
        test_prediction = validate_predictions(np.load(bank_test, allow_pickle=False), test_size, "banked Model B test")
        model_b_diagnostics = {"loaded": True}
    else:
        test_prediction, model_b_diagnostics = fit_selected(
            model_b_train,
            model_b_outcomes,
            test,
            gate.design,
            gate.temperature,
            trees=760,
            samples=128,
        )
        test_prediction = validate_predictions(test_prediction, test_size, "Model B test")
        atomic_save(bank_test, test_prediction)
    metrics = {
        "feature_version": FEATURE_VERSION,
        "selection": gate_to_json(gate),
        "model_a": model_a_diagnostics,
        "model_b": model_b_diagnostics,
        "validation_prediction": {
            "count": len(val_prediction),
            "mean": float(np.mean(val_prediction)),
            "zero_share": float(np.mean(val_prediction == 0)),
            "maximum": float(np.max(val_prediction)),
        },
        "test_prediction": {
            "count": len(test_prediction),
            "mean": float(np.mean(test_prediction)),
            "zero_share": float(np.mean(test_prediction == 0)),
            "maximum": float(np.max(test_prediction)),
        },
        "elapsed_seconds": time.time() - started,
    }
    tmp_metrics = bank_metrics.with_name(f"{bank_metrics.name}.{os.getpid()}.tmp")
    tmp_metrics.write_text(json.dumps(metrics, indent=2))
    os.replace(tmp_metrics, bank_metrics)
    register_banked_predictions(bank_val, bank_test)
    save_predictions(val_prediction, test_prediction)
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    elapsed(started, "full_complete")


def main() -> None:
    warnings.filterwarnings("ignore")
    np.random.seed(20260731)
    if "--debug" in sys.argv:
        debug_main()
    else:
        full_main()


if __name__ == "__main__":
    main()
