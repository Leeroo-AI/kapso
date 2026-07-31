from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import qmc


PRICE_QUANTILES = np.array([0.10, 0.25, 0.50, 0.75, 0.90], dtype=np.float64)
COUNT_CLASSES = 8
LEARNING_RATE = 0.04
NUM_LEAVES = 48
MIN_LEAF_ROWS = 2000
FEATURE_FRACTION = 0.80
L2 = 10.0
MAX_TREES = 1400
SOBOL_SEED = 20260731


@dataclass
class GateResult:
    design: str
    temperature: float
    fold_metrics: list[dict]
    aggregate: dict
    slice_metrics: list[dict]


def count_class(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.int64)
    result = values.copy()
    result[(values >= 6) & (values <= 9)] = 6
    result[values >= 10] = 7
    return result.astype(np.int8)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"row_id", "timestamp", "customer_id", "y91", "n91"}
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    values = frame[columns].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(values, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    return values


def base_params(objective: str) -> dict:
    params = {
        "objective": objective,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "min_data_in_leaf": MIN_LEAF_ROWS,
        "feature_fraction": FEATURE_FRACTION,
        "lambda_l2": L2,
        "max_bin": 127,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        "deterministic": True,
        "force_col_wise": True,
        "seed": SOBOL_SEED,
        "feature_fraction_seed": SOBOL_SEED,
    }
    return params


def train_count(x: np.ndarray, counts: np.ndarray, trees: int) -> lgb.Booster:
    params = base_params("multiclass")
    params["num_class"] = COUNT_CLASSES
    data = lgb.Dataset(x, label=count_class(counts), free_raw_data=True)
    return lgb.train(params, data, num_boost_round=min(trees, MAX_TREES), callbacks=[lgb.log_evaluation(0)])


def train_binary(x: np.ndarray, labels: np.ndarray, trees: int) -> lgb.Booster:
    params = base_params("binary")
    data = lgb.Dataset(x, label=np.asarray(labels, dtype=np.int8), free_raw_data=True)
    return lgb.train(params, data, num_boost_round=min(trees, MAX_TREES), callbacks=[lgb.log_evaluation(0)])


def train_regression(x: np.ndarray, target: np.ndarray, objective: str, trees: int, alpha: float | None = None) -> lgb.Booster:
    params = base_params(objective)
    if alpha is not None:
        params["alpha"] = float(alpha)
    data = lgb.Dataset(x, label=np.asarray(target, dtype=np.float32), free_raw_data=True)
    return lgb.train(params, data, num_boost_round=min(trees, MAX_TREES), callbacks=[lgb.log_evaluation(0)])


def train_quantiles(x: np.ndarray, target: np.ndarray, trees: int) -> list[lgb.Booster]:
    return [train_regression(x, target, "quantile", trees, float(alpha)) for alpha in PRICE_QUANTILES]


def predict_quantiles(models: list[lgb.Booster], x: np.ndarray) -> np.ndarray:
    result = np.column_stack([model.predict(x) for model in models]).astype(np.float32)
    result = np.maximum.accumulate(result, axis=1)
    return np.maximum(result, 0)


def predict_conditional_price(models: list[lgb.Booster], x: np.ndarray) -> np.ndarray:
    result = np.zeros((len(x), COUNT_CLASSES, len(PRICE_QUANTILES)), dtype=np.float32)
    for cls in range(1, COUNT_CLASSES):
        augmented = np.column_stack([x, np.full(len(x), cls, dtype=np.float32)])
        result[:, cls, :] = predict_quantiles(models, augmented)
    return result


def temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature == 1.0:
        return probabilities
    logits = np.log(np.clip(probabilities, 1e-8, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def tail_values(counts: np.ndarray) -> dict[int, np.ndarray]:
    values = np.asarray(counts, dtype=np.int64)
    six = np.sort(values[(values >= 6) & (values <= 9)])
    ten = np.sort(values[values >= 10])
    if len(six) == 0:
        six = np.arange(6, 10)
    if len(ten) == 0:
        ten = np.array([10], dtype=np.int64)
    return {6: six, 7: ten}


def interpolation_indices(uniform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    high = np.searchsorted(PRICE_QUANTILES, uniform, side="right")
    high = np.clip(high, 1, len(PRICE_QUANTILES) - 1)
    low = high - 1
    below = uniform <= PRICE_QUANTILES[0]
    above = uniform >= PRICE_QUANTILES[-1]
    low[below] = 0
    high[below] = 0
    low[above] = len(PRICE_QUANTILES) - 1
    high[above] = len(PRICE_QUANTILES) - 1
    denominator = PRICE_QUANTILES[high] - PRICE_QUANTILES[low]
    weight = np.divide(
        uniform - PRICE_QUANTILES[low],
        denominator,
        out=np.zeros_like(uniform),
        where=denominator > 0,
    )
    return low, high, np.clip(weight, 0, 1)


def decode_compound(
    probabilities: np.ndarray,
    price_predictions: np.ndarray,
    tails: dict[int, np.ndarray],
    temperature: float,
    samples: int,
) -> np.ndarray:
    scaled = temperature_scale(probabilities, temperature)
    power = int(round(math.log2(samples)))
    if 2 ** power != samples:
        raise ValueError("compound sample count must be a power of two")
    points = qmc.Sobol(d=3, scramble=True, seed=SOBOL_SEED).random_base2(power)
    low, high, weight = interpolation_indices(points[:, 2])
    output = np.empty(len(scaled), dtype=np.float32)
    batch = 25000
    for start in range(0, len(scaled), batch):
        stop = min(start + batch, len(scaled))
        probs = scaled[start:stop]
        prices = price_predictions[start:stop]
        cdf = np.cumsum(probs, axis=1)
        classes = (points[None, :, 0, None] > cdf[:, None, :]).sum(axis=2)
        classes = np.minimum(classes, COUNT_CLASSES - 1).astype(np.int8)
        counts = classes.astype(np.float32)
        for cls in (6, 7):
            empirical = tails[cls]
            positions = np.minimum((points[:, 1] * len(empirical)).astype(np.int64), len(empirical) - 1)
            sampled = empirical[positions].astype(np.float32)
            counts = np.where(classes == cls, sampled[None, :], counts)
        rows = np.arange(stop - start)[:, None]
        lower_price = prices[rows, classes, low[None, :]]
        upper_price = prices[rows, classes, high[None, :]]
        average_price = lower_price + (upper_price - lower_price) * weight[None, :]
        draws = counts * np.maximum(average_price, 0)
        output[start:stop] = np.median(draws, axis=1).astype(np.float32)
    return np.maximum(output, 0)


def analytic_hurdle_prediction(probability_positive: np.ndarray, positive_quantiles: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability_positive, dtype=np.float64), 1e-6, 1 - 1e-6)
    conditional_q = np.clip(1.0 - 0.5 / probability, 0.0, 0.5)
    low, high, weight = interpolation_indices(conditional_q)
    rows = np.arange(len(probability))
    positive = positive_quantiles[rows, low] + (positive_quantiles[rows, high] - positive_quantiles[rows, low]) * weight
    return np.where(probability > 0.5, np.maximum(positive, 0), 0).astype(np.float32)


def deterministic_sample(frame: pd.DataFrame, cap: int) -> np.ndarray:
    if len(frame) <= cap:
        return np.arange(len(frame), dtype=np.int64)
    hashes = pd.util.hash_pandas_object(frame[["customer_id", "timestamp"]], index=False).to_numpy(dtype=np.uint64)
    return np.argpartition(hashes, cap)[:cap]


def fit_predict_components(
    train_frame: pd.DataFrame,
    train_outcomes: pd.DataFrame,
    predict_frame: pd.DataFrame,
    trees: int,
    samples: int,
    temperature: float,
    train_cap: int | None = None,
) -> dict[str, np.ndarray]:
    columns = feature_columns(train_frame)
    if columns != feature_columns(predict_frame):
        raise RuntimeError("train and prediction feature columns differ")
    selected = np.arange(len(train_frame)) if train_cap is None else deterministic_sample(train_frame, train_cap)
    x_train = matrix(train_frame.iloc[selected], columns)
    x_predict = matrix(predict_frame, columns)
    y = train_outcomes["y91"].to_numpy(dtype=np.float32)[selected]
    n = train_outcomes["n91"].to_numpy(dtype=np.int64)[selected]
    count_model = train_count(x_train, n, trees)
    count_probabilities = np.asarray(count_model.predict(x_predict), dtype=np.float32)
    positive = n > 0
    average_price = np.divide(y[positive], n[positive], dtype=np.float32)
    price_train = np.column_stack([x_train[positive], count_class(n[positive]).astype(np.float32)])
    price_models = train_quantiles(price_train, average_price, trees)
    price_predictions = predict_conditional_price(price_models, x_predict)
    compound = decode_compound(count_probabilities, price_predictions, tail_values(n), temperature, samples)
    direct_model = train_regression(x_train, y, "regression_l1", trees)
    direct = np.maximum(direct_model.predict(x_predict), 0).astype(np.float32)
    hurdle_model = train_binary(x_train, y > 0, trees)
    hurdle_probability = np.asarray(hurdle_model.predict(x_predict), dtype=np.float32)
    hurdle_quantile_models = train_quantiles(x_train[y > 0], y[y > 0], trees)
    hurdle_quantiles = predict_quantiles(hurdle_quantile_models, x_predict)
    hurdle = analytic_hurdle_prediction(hurdle_probability, hurdle_quantiles)
    return {
        "count_probabilities": count_probabilities,
        "price_predictions": price_predictions,
        "tail6": tail_values(n)[6],
        "tail7": tail_values(n)[7],
        "compound": compound,
        "direct": direct,
        "hurdle": hurdle,
    }


def fold_definitions(timestamps: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    available = pd.Index(timestamps.drop_duplicates()).sort_values()
    desired = [
        pd.Timestamp("2013-01-03"),
        pd.Timestamp("2014-01-02"),
        pd.Timestamp("2014-07-03"),
        pd.Timestamp("2015-01-01"),
        pd.Timestamp("2015-07-02"),
    ]
    folds = []
    for validation_time in desired:
        if validation_time not in available:
            validation_time = available[available.get_indexer([validation_time], method="nearest")[0]]
        eligible = available[available < validation_time - pd.Timedelta(days=91)]
        if len(eligible) == 0:
            continue
        folds.append((eligible[-1], validation_time))
    if len(folds) != 5:
        raise RuntimeError(f"expected five forward folds, found {len(folds)}")
    return folds


def mae(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(prediction))))


def clustered_bootstrap_se(customer_id: np.ndarray, losses: np.ndarray, repetitions: int = 200) -> float:
    frame = pd.DataFrame({"customer_id": customer_id, "loss": losses})
    clusters = frame.groupby("customer_id", sort=False, observed=True)["loss"].mean().to_numpy(dtype=np.float64)
    rng = np.random.default_rng(SOBOL_SEED)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        estimates[index] = clusters[rng.integers(0, len(clusters), len(clusters))].mean()
    return float(estimates.std(ddof=1))


def slice_report(frame: pd.DataFrame, outcomes: pd.DataFrame, predictions: np.ndarray, fold: str) -> list[dict]:
    y = outcomes["y91"].to_numpy(dtype=np.float64)
    n = outcomes["n91"].to_numpy(dtype=np.int64)
    frequency = frame["frequency"].to_numpy(dtype=np.float64)
    slices = []
    class_values = count_class(n)
    for cls in range(COUNT_CLASSES):
        mask = class_values == cls
        if mask.any():
            slices.append({"fold": fold, "axis": "count_class", "stratum": str(cls), "count": int(mask.sum()), "mae": mae(y[mask], predictions[mask])})
    bins = np.array([0, 1, 2, 5, 10, np.inf])
    labels = ["1", "2", "3-5", "6-10", "11+"]
    for left, right, label in zip(bins[:-1], bins[1:], labels):
        mask = (frequency > left) & (frequency <= right)
        if mask.any():
            slices.append({"fold": fold, "axis": "history_frequency", "stratum": label, "count": int(mask.sum()), "mae": mae(y[mask], predictions[mask])})
    return slices


def run_forward_gate(
    train_frame: pd.DataFrame,
    outcomes: pd.DataFrame,
    trees: int = 420,
    samples: int = 128,
    train_cap: int = 700000,
) -> GateResult:
    fold_outputs = []
    folds = fold_definitions(train_frame["timestamp"])
    for index, (train_end, validation_time) in enumerate(folds, start=1):
        train_mask = train_frame["timestamp"].le(train_end).to_numpy()
        validation_mask = train_frame["timestamp"].eq(validation_time).to_numpy()
        fold_train = train_frame.loc[train_mask].reset_index(drop=True)
        fold_outcomes = outcomes.loc[train_mask].reset_index(drop=True)
        fold_validation = train_frame.loc[validation_mask].reset_index(drop=True)
        validation_outcomes = outcomes.loc[validation_mask].reset_index(drop=True)
        started = time.time()
        components = fit_predict_components(
            fold_train,
            fold_outcomes,
            fold_validation,
            trees,
            samples,
            1.0,
            train_cap,
        )
        fold_outputs.append(
            {
                "index": index,
                "train_end": str(train_end.date()),
                "validation_time": str(validation_time.date()),
                "frame": fold_validation,
                "outcomes": validation_outcomes,
                "components": components,
            }
        )
        print(f"[gate] fold={index} train_end={train_end.date()} validation={validation_time.date()} rows={len(fold_validation)} elapsed={time.time() - started:.1f}s")
        gc.collect()
    temperatures = [0.75, 1.0, 1.25, 1.5]
    temperature_scores = {}
    for temperature in temperatures:
        fold_scores = []
        for fold in fold_outputs:
            components = fold["components"]
            decoded = decode_compound(
                components["count_probabilities"],
                components["price_predictions"],
                {6: components["tail6"], 7: components["tail7"]},
                temperature,
                samples,
            )
            fold_scores.append(mae(fold["outcomes"]["y91"].to_numpy(), decoded))
        temperature_scores[temperature] = float(np.median(fold_scores))
    best_temperature = min(temperature_scores, key=temperature_scores.get)
    if best_temperature != 1.0 and temperature_scores[best_temperature] >= temperature_scores[1.0]:
        best_temperature = 1.0
    fold_metrics = []
    slices = []
    pooled_customers = []
    pooled_differences = {"direct": [], "hurdle": []}
    for fold in fold_outputs:
        components = fold["components"]
        y = fold["outcomes"]["y91"].to_numpy(dtype=np.float64)
        compound = decode_compound(
            components["count_probabilities"],
            components["price_predictions"],
            {6: components["tail6"], 7: components["tail7"]},
            best_temperature,
            samples,
        )
        metrics = {
            "fold": fold["index"],
            "train_end": fold["train_end"],
            "validation_time": fold["validation_time"],
            "count": len(y),
            "compound_mae": mae(y, compound),
            "direct_mae": mae(y, components["direct"]),
            "hurdle_mae": mae(y, components["hurdle"]),
        }
        fold_metrics.append(metrics)
        slices.extend(slice_report(fold["frame"], fold["outcomes"], compound, str(fold["validation_time"])))
        pooled_customers.append(fold["frame"]["customer_id"].to_numpy())
        pooled_differences["direct"].append(np.abs(y - components["direct"]) - np.abs(y - compound))
        pooled_differences["hurdle"].append(np.abs(y - components["hurdle"]) - np.abs(y - compound))
    median_scores = {
        design: float(np.median([entry[f"{design}_mae"] for entry in fold_metrics]))
        for design in ("compound", "direct", "hurdle")
    }
    best_reference = min(("direct", "hurdle"), key=median_scores.get)
    customers = np.concatenate(pooled_customers)
    improvement = float(np.mean(np.concatenate(pooled_differences[best_reference])))
    standard_error = clustered_bootstrap_se(customers, np.concatenate(pooled_differences[best_reference]))
    compound_admitted = median_scores["compound"] < median_scores[best_reference] and improvement > standard_error
    design = "compound" if compound_admitted else best_reference
    aggregate = {
        "median_mae": median_scores,
        "temperature_scores": {str(key): value for key, value in temperature_scores.items()},
        "selected_temperature": best_temperature,
        "best_reference": best_reference,
        "compound_improvement": improvement,
        "clustered_bootstrap_se": standard_error,
        "compound_admitted": compound_admitted,
        "selected_design": design,
    }
    print(f"[gate] aggregate={json.dumps(aggregate, sort_keys=True)}")
    return GateResult(design, float(best_temperature), fold_metrics, aggregate, slices)


def fit_selected(
    train_frame: pd.DataFrame,
    outcomes: pd.DataFrame,
    predict_frame: pd.DataFrame,
    design: str,
    temperature: float,
    trees: int,
    samples: int,
) -> tuple[np.ndarray, dict]:
    columns = feature_columns(train_frame)
    if columns != feature_columns(predict_frame):
        raise RuntimeError("selected-model feature columns differ")
    x_train = matrix(train_frame, columns)
    x_predict = matrix(predict_frame, columns)
    y = outcomes["y91"].to_numpy(dtype=np.float32)
    n = outcomes["n91"].to_numpy(dtype=np.int64)
    count_model = train_count(x_train, n, trees)
    train_probabilities = np.asarray(count_model.predict(x_train), dtype=np.float32)
    predict_probabilities = np.asarray(count_model.predict(x_predict), dtype=np.float32)
    diagnostics = {"design": design, "temperature": temperature, "trees": trees}
    if design == "compound":
        positive = n > 0
        average_price = np.divide(y[positive], n[positive], dtype=np.float32)
        price_train = np.column_stack([x_train[positive], count_class(n[positive]).astype(np.float32)])
        price_models = train_quantiles(price_train, average_price, trees)
        price_predictions = predict_conditional_price(price_models, x_predict)
        prediction = decode_compound(predict_probabilities, price_predictions, tail_values(n), temperature, samples)
    else:
        augmented_train = np.column_stack([x_train, train_probabilities])
        augmented_predict = np.column_stack([x_predict, predict_probabilities])
        if design == "direct":
            reference = train_regression(augmented_train, y, "regression_l1", trees)
            prediction = np.maximum(reference.predict(augmented_predict), 0).astype(np.float32)
        elif design == "hurdle":
            hurdle = train_binary(augmented_train, y > 0, trees)
            probability = np.asarray(hurdle.predict(augmented_predict), dtype=np.float32)
            quantile_models = train_quantiles(augmented_train[y > 0], y[y > 0], trees)
            quantiles = predict_quantiles(quantile_models, augmented_predict)
            prediction = analytic_hurdle_prediction(probability, quantiles)
        else:
            raise ValueError(design)
    diagnostics["prediction_zero_share"] = float(np.mean(prediction == 0))
    diagnostics["prediction_mean"] = float(np.mean(prediction))
    diagnostics["prediction_max"] = float(np.max(prediction))
    return np.maximum(prediction, 0), diagnostics
