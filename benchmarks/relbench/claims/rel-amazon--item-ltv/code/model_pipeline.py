from __future__ import annotations

# Imports

import gc
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


# Configuration

SEED = 3407
MAX_ITERATIONS = 2400


@dataclass
class FoldResult:
    training_decay: float
    origin: str
    rows: int
    best_iteration: int
    raw_mae: float
    rounded_mae: float
    cold_raw_mae: float
    warm_raw_mae: float


# Model configuration

def probe_device() -> str:
    rng = np.random.default_rng(SEED)
    matrix = rng.normal(size=(2048, 12)).astype(np.float32)
    target = np.abs(matrix[:, 0] * 2 + matrix[:, 1]).astype(np.float32)
    try:
        dataset = lgb.Dataset(matrix, label=target, free_raw_data=True)
        lgb.train(
            {
                "objective": "regression_l1",
                "device_type": "gpu",
                "verbosity": -1,
                "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
                "seed": SEED,
            },
            dataset,
            num_boost_round=2,
            callbacks=[lgb.log_evaluation(0)],
        )
        return "gpu"
    except Exception as error:
        print(f"[model] LightGBM GPU probe failed, using assigned CPU threads: {type(error).__name__}: {error}", flush=True)
        return "cpu"


def parameters(device_type: str) -> dict[str, object]:
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "num_leaves": 127,
        "learning_rate": 0.035,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.2,
        "lambda_l2": 10.0,
        "max_bin": 127,
        "device_type": device_type,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "verbosity": -1,
        "deterministic": device_type == "cpu",
        "force_col_wise": device_type == "cpu",
    }


# Metrics

def dollar_mae(truth_count: np.ndarray, prediction_count: np.ndarray, price: np.ndarray) -> float:
    return float(np.mean(price * np.abs(truth_count - prediction_count)))


def supported_count_prediction(prediction: np.ndarray, rounded: bool) -> np.ndarray:
    prediction = np.maximum(np.asarray(prediction, dtype=np.float64), 1.0)
    if rounded:
        prediction = np.maximum(np.rint(prediction), 1.0)
    return prediction


def normalized_weight(price: np.ndarray, multiplier: np.ndarray | None = None) -> np.ndarray:
    weight = np.asarray(price, dtype=np.float64)
    if multiplier is not None:
        weight = weight * np.asarray(multiplier, dtype=np.float64)
    return (weight / np.mean(weight)).astype(np.float32)


def quarter_decay_multiplier(timestamps: np.ndarray, reference_origin: np.datetime64, decay: float) -> np.ndarray:
    age_days = (reference_origin - timestamps.astype("datetime64[ns]")).astype("timedelta64[s]").astype(np.float64) / 86400.0
    return np.power(decay, np.maximum(age_days, 0.0) / 91.0)


# Internal selection

def expanding_fold_origins(frame: pd.DataFrame) -> list[pd.Timestamp]:
    available = set(pd.to_datetime(frame.loc[frame["split"] == "train", "timestamp"].unique()))
    desired = pd.to_datetime(["2013-01-03", "2013-10-03", "2014-01-02", "2014-10-02", "2015-01-01"])
    origins = [pd.Timestamp(origin) for origin in desired if pd.Timestamp(origin) in available]
    if len(origins) < 3:
        ordered = sorted(available)
        origins = [pd.Timestamp(origin) for origin in ordered[-5:]]
    return origins


def run_internal_folds(
    matrix: np.ndarray,
    frame: pd.DataFrame,
    categorical_indices: list[int],
    device_type: str,
) -> tuple[int, bool, float, list[FoldResult]]:
    train_mask = frame["split"].to_numpy() == "train"
    positive = np.isfinite(frame["price"].to_numpy(dtype=np.float64)) & (frame["price"].to_numpy(dtype=np.float64) > 0)
    timestamps = pd.to_datetime(frame["timestamp"]).to_numpy()
    target = frame["ltv"].to_numpy(dtype=np.float64) / np.where(positive, frame["price"].to_numpy(dtype=np.float64), 1.0)
    results = []
    for training_decay in (1.0, 0.98):
        for fold_origin in expanding_fold_origins(frame):
            fold_time = np.datetime64(fold_origin)
            fit_indices = np.flatnonzero(train_mask & positive & (timestamps < fold_time))
            validation_indices = np.flatnonzero(train_mask & positive & (timestamps == fold_time))
            fit_multiplier = quarter_decay_multiplier(timestamps[fit_indices], fold_time, training_decay)
            fit_weight = normalized_weight(
                frame.iloc[fit_indices]["price"].to_numpy(dtype=np.float64), fit_multiplier
            )
            validation_weight = normalized_weight(frame.iloc[validation_indices]["price"].to_numpy(dtype=np.float64))
            fit_set = lgb.Dataset(
                matrix[fit_indices],
                label=target[fit_indices].astype(np.float32),
                weight=fit_weight,
                categorical_feature=categorical_indices,
                free_raw_data=True,
            )
            validation_set = lgb.Dataset(
                matrix[validation_indices],
                label=target[validation_indices].astype(np.float32),
                weight=validation_weight,
                categorical_feature=categorical_indices,
                reference=fit_set,
                free_raw_data=True,
            )
            started = time.time()
            model = lgb.train(
                parameters(device_type),
                fit_set,
                num_boost_round=MAX_ITERATIONS,
                valid_sets=[validation_set],
                callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
            )
            raw = supported_count_prediction(model.predict(matrix[validation_indices], num_iteration=model.best_iteration), False)
            rounded = supported_count_prediction(raw, True)
            truth = target[validation_indices]
            price = frame.iloc[validation_indices]["price"].to_numpy(dtype=np.float64)
            cold = frame.iloc[validation_indices]["c91"].to_numpy(dtype=np.float64) == 0
            result = FoldResult(
                training_decay=training_decay,
                origin=str(fold_origin.date()),
                rows=int(len(validation_indices)),
                best_iteration=int(model.best_iteration),
                raw_mae=dollar_mae(truth, raw, price),
                rounded_mae=dollar_mae(truth, rounded, price),
                cold_raw_mae=dollar_mae(truth[cold], raw[cold], price[cold]) if cold.any() else float("nan"),
                warm_raw_mae=dollar_mae(truth[~cold], raw[~cold], price[~cold]) if (~cold).any() else float("nan"),
            )
            results.append(result)
            print(
                f"[fold] training_decay={result.training_decay} origin={result.origin} rows={result.rows} "
                f"best_iteration={result.best_iteration} raw_mae={result.raw_mae:.6f} "
                f"rounded_mae={result.rounded_mae:.6f} cold_mae={result.cold_raw_mae:.6f} "
                f"warm_mae={result.warm_raw_mae:.6f} elapsed={time.time() - started:.2f}s",
                flush=True,
            )
            del model, fit_set, validation_set
            gc.collect()
    candidate_medians = {}
    for training_decay in (1.0, 0.98):
        candidate = [result for result in results if result.training_decay == training_decay]
        candidate_medians[training_decay] = min(
            float(np.median([result.raw_mae for result in candidate])),
            float(np.median([result.rounded_mae for result in candidate])),
        )
    selected_training_decay = min(candidate_medians, key=candidate_medians.get)
    selected_results = [result for result in results if result.training_decay == selected_training_decay]
    raw_median = float(np.median([result.raw_mae for result in selected_results]))
    rounded_median = float(np.median([result.rounded_mae for result in selected_results]))
    rounded_selected = rounded_median < raw_median
    final_iterations = int(np.median([result.best_iteration for result in selected_results]))
    final_iterations = max(50, min(final_iterations, MAX_ITERATIONS))
    selected_scores = [result.rounded_mae if rounded_selected else result.raw_mae for result in selected_results]
    dispersion = float(np.std(selected_scores, ddof=1)) if len(selected_scores) > 1 else 0.0
    print(
        f"[fold] referee candidate_medians={candidate_medians} selected_training_decay={selected_training_decay} "
        f"median_raw_mae={raw_median:.6f} median_rounded_mae={rounded_median:.6f} "
        f"dispersion={dispersion:.6f} rounded_selected={rounded_selected} final_iterations={final_iterations}",
        flush=True,
    )
    return final_iterations, rounded_selected, selected_training_decay, results


# Final chains

def fit_count_model(
    matrix: np.ndarray,
    target: np.ndarray,
    price: np.ndarray,
    indices: np.ndarray,
    categorical_indices: list[int],
    device_type: str,
    iterations: int,
    timestamps: np.ndarray,
    reference_origin: np.datetime64,
    training_decay: float,
) -> lgb.Booster:
    multiplier = quarter_decay_multiplier(timestamps[indices], reference_origin, training_decay)
    dataset = lgb.Dataset(
        matrix[indices],
        label=target[indices].astype(np.float32),
        weight=normalized_weight(price[indices], multiplier),
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    model = lgb.train(
        parameters(device_type),
        dataset,
        num_boost_round=iterations,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def fit_direct_fallback(
    matrix: np.ndarray,
    target_dollars: np.ndarray,
    indices: np.ndarray,
    categorical_indices: list[int],
    device_type: str,
    iterations: int,
) -> lgb.Booster | None:
    if len(indices) < 300:
        return None
    dataset = lgb.Dataset(
        matrix[indices],
        label=target_dollars[indices].astype(np.float32),
        categorical_feature=categorical_indices,
        free_raw_data=True,
    )
    return lgb.train(
        parameters(device_type),
        dataset,
        num_boost_round=min(iterations, 500),
        callbacks=[lgb.log_evaluation(0)],
    )


def dollar_predictions(
    model: lgb.Booster,
    matrix: np.ndarray,
    indices: np.ndarray,
    price: np.ndarray,
    rounded: bool,
    invalid_model: lgb.Booster | None,
    invalid_center: float,
) -> np.ndarray:
    output = np.zeros(len(indices), dtype=np.float64)
    local_price = price[indices]
    positive = np.isfinite(local_price) & (local_price > 0)
    if positive.any():
        count = supported_count_prediction(model.predict(matrix[indices[positive]]), rounded)
        output[positive] = local_price[positive] * count
    invalid = ~np.isfinite(local_price) | (local_price < 0)
    if invalid.any():
        if invalid_model is not None:
            output[invalid] = np.maximum(invalid_model.predict(matrix[indices[invalid]]), 0.0)
        else:
            output[invalid] = max(invalid_center, 0.0)
    output[local_price == 0] = 0.0
    return output


def train_prediction_chains(
    frame: pd.DataFrame,
    feature_names: list[str],
    categorical_indices: list[int],
    output_dir: Path,
    debug: bool,
) -> dict[str, object]:
    started = time.time()
    matrix = frame[feature_names].to_numpy(dtype=np.float32)
    split = frame["split"].to_numpy()
    price = frame["price"].to_numpy(dtype=np.float64)
    target_dollars = frame["ltv"].to_numpy(dtype=np.float64)
    positive = np.isfinite(price) & (price > 0)
    target_count = target_dollars / np.where(positive, price, 1.0)
    train_indices_all = np.flatnonzero((split == "train") & positive)
    val_indices = np.flatnonzero(split == "val")
    test_indices = np.flatnonzero(split == "test")
    device_type = probe_device()
    model_categorical_indices = [] if device_type == "gpu" else categorical_indices
    print(
        f"[model] device={device_type} matrix={matrix.shape} debug={debug} "
        f"categorical_features={len(model_categorical_indices)}",
        flush=True,
    )
    if debug:
        rng = np.random.default_rng(SEED)
        train_indices = np.sort(rng.choice(train_indices_all, size=min(100000, len(train_indices_all)), replace=False))
        final_iterations = 100
        rounded_selected = False
        selected_training_decay = 1.0
        fold_results: list[FoldResult] = []
    else:
        train_indices = train_indices_all
        final_iterations, rounded_selected, selected_training_decay, fold_results = run_internal_folds(
            matrix, frame, model_categorical_indices, device_type
        )
    invalid_train = np.flatnonzero((split == "train") & (~np.isfinite(price) | (price < 0)))
    timestamps = pd.to_datetime(frame["timestamp"]).to_numpy()
    validation_origin = np.min(timestamps[val_indices])
    test_origin = np.min(timestamps[test_indices])
    invalid_center_a = float(np.nanmedian(target_dollars[invalid_train])) if len(invalid_train) else 0.0
    model_a = fit_count_model(
        matrix, target_count, price, train_indices, model_categorical_indices, device_type,
        final_iterations, timestamps, validation_origin, selected_training_decay
    )
    invalid_model_a = fit_direct_fallback(
        matrix, target_dollars, invalid_train, model_categorical_indices, device_type, final_iterations
    )
    val_predictions = dollar_predictions(
        model_a, matrix, val_indices, price, rounded_selected, invalid_model_a, invalid_center_a
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "val_predictions.npy", val_predictions)
    print(
        f"[model] Model A train_rows={len(train_indices)} iterations={final_iterations} retained_val={len(val_predictions)} "
        f"elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    del model_a, invalid_model_a
    gc.collect()
    val_positive_indices = np.flatnonzero((split == "val") & positive)
    if debug:
        rng = np.random.default_rng(SEED + 1)
        val_addition = np.sort(
            rng.choice(val_positive_indices, size=min(50000, len(val_positive_indices)), replace=False)
        )
        model_b_indices = np.concatenate([train_indices, val_addition])
    else:
        model_b_indices = np.concatenate([train_indices_all, val_positive_indices])
    invalid_model_b_indices = np.flatnonzero(((split == "train") | (split == "val")) & (~np.isfinite(price) | (price < 0)))
    invalid_center_b = float(np.nanmedian(target_dollars[invalid_model_b_indices])) if len(invalid_model_b_indices) else 0.0
    model_b = fit_count_model(
        matrix, target_count, price, model_b_indices, model_categorical_indices, device_type,
        final_iterations, timestamps, test_origin, selected_training_decay
    )
    invalid_model_b = fit_direct_fallback(
        matrix, target_dollars, invalid_model_b_indices, model_categorical_indices, device_type, final_iterations
    )
    test_predictions = dollar_predictions(
        model_b, matrix, test_indices, price, rounded_selected, invalid_model_b, invalid_center_b
    )
    np.save(output_dir / "test_predictions.npy", test_predictions)
    retained = np.load(output_dir / "val_predictions.npy", allow_pickle=False)
    if not np.array_equal(retained, val_predictions):
        raise RuntimeError("retained Model A validation predictions changed during Model B")
    report = {
        "device": device_type,
        "features": len(feature_names),
        "final_iterations": final_iterations,
        "rounded_counts": rounded_selected,
        "training_time_decay": selected_training_decay,
        "model_a_train_rows": int(len(train_indices)),
        "model_b_train_rows": int(len(model_b_indices)),
        "folds": [result.__dict__ for result in fold_results],
        "prediction_slices": {
            "validation": {
                "rows": int(len(val_predictions)),
                "mean": float(np.mean(val_predictions)),
                "median": float(np.median(val_predictions)),
                "p99": float(np.quantile(val_predictions, 0.99)),
            },
            "test": {
                "rows": int(len(test_predictions)),
                "mean": float(np.mean(test_predictions)),
                "median": float(np.median(test_predictions)),
                "p99": float(np.quantile(test_predictions, 0.99)),
            },
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(
        f"[model] Model B train_rows={len(model_b_indices)} retained_test={len(test_predictions)} "
        f"elapsed={time.time() - started:.2f}s",
        flush=True,
    )
    return report
