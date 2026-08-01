from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import Ridge

from feature_factory import DAY, Events, FeatureBlocks, build_internal_frozen_features


@dataclass
class Selection:
    xgb_rounds: int
    lgb_rounds: int
    xgb_weight: float
    lgb_weight: float
    blend_intercept: float
    calibration_coef: list[float]
    calibration_intercept: float
    recency_weighted: bool
    fold_records: list[dict]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "Selection":
        return cls(**json.loads(path.read_text()))


def assemble(base: np.ndarray, label: np.ndarray, index: np.ndarray | slice) -> np.ndarray:
    return np.column_stack((np.asarray(base[index], dtype=np.float32), np.asarray(label, dtype=np.float32))).astype(np.float32, copy=False)


def recency_weights(times: np.ndarray) -> np.ndarray:
    age_years = (times.max() - times) / (365.25 * DAY)
    return np.exp2(-age_years / 4).astype(np.float32)


def _xgb_params() -> dict:
    return {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": "cuda",
        "grow_policy": "lossguide",
        "max_leaves": 192,
        "max_depth": 0,
        "learning_rate": 0.04,
        "min_child_weight": 100,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "max_bin": 256,
        "reg_lambda": 10.0,
        "reg_alpha": 0.05,
        "verbosity": 0,
        "seed": 1337,
        "nthread": 11,
    }


def _lgb_params() -> dict:
    return {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 192,
        "learning_rate": 0.04,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "max_bin": 255,
        "lambda_l2": 10.0,
        "lambda_l1": 0.05,
        "verbosity": -1,
        "num_threads": 11,
        "seed": 1337,
        "force_col_wise": True,
    }


def _xgb_internal(train_x, train_y, train_weight, eval_x, eval_y, eval_weight, rounds):
    dtrain = xgb.DMatrix(train_x, label=train_y, weight=train_weight, missing=np.nan, nthread=11)
    deval = xgb.DMatrix(eval_x, label=eval_y, weight=eval_weight, missing=np.nan, nthread=11)
    model = xgb.train(_xgb_params(), dtrain, num_boost_round=rounds, evals=[(deval, "forward")], early_stopping_rounds=30, verbose_eval=False)
    prediction = model.predict(deval, iteration_range=(0, model.best_iteration + 1))
    best = model.best_iteration + 1
    del dtrain, deval, model
    gc.collect()
    return prediction.astype(np.float32), best


def _lgb_internal(train_x, train_y, train_weight, eval_x, eval_y, eval_weight, rounds):
    train_data = lgb.Dataset(train_x, label=train_y, weight=train_weight, free_raw_data=True)
    eval_data = lgb.Dataset(eval_x, label=eval_y, weight=eval_weight, reference=train_data, free_raw_data=True)
    model = lgb.train(_lgb_params(), train_data, num_boost_round=rounds, valid_sets=[eval_data], callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    prediction = model.predict(eval_x, num_iteration=model.best_iteration)
    best = model.best_iteration
    del train_data, eval_data, model
    gc.collect()
    return prediction.astype(np.float32), best


def _sample(index: np.ndarray, limit: int) -> np.ndarray:
    if len(index) <= limit:
        return index
    positions = np.linspace(0, len(index) - 1, limit, dtype=np.int64)
    return index[positions]


def _mixture_weights(frozen: np.ndarray, times: np.ndarray, cutoff: int) -> np.ndarray:
    user_cold = frozen[:, 0] == 0
    beer_cold = frozen[:, 13] == 0
    category = user_cold.astype(np.int8) + 2 * beer_cold.astype(np.int8)
    target = np.array([0.350372, 0.065583, 0.554897, 0.029148], dtype=np.float64)
    observed = np.bincount(category, minlength=4).astype(np.float64)
    observed /= observed.sum()
    category_weight = target[category] / np.maximum(observed[category], 1e-6)
    horizon = np.maximum(times - cutoff, 0)
    max_horizon = max(int(horizon.max()), 1)
    bucket = np.minimum((horizon.astype(np.float64) / max_horizon * 6).astype(np.int8), 5)
    target_horizon = np.array([589663, 542710, 495939, 456797, 378606, 31645], dtype=np.float64)
    target_horizon /= target_horizon.sum()
    observed_horizon = np.bincount(bucket, minlength=6).astype(np.float64)
    observed_horizon /= observed_horizon.sum()
    horizon_weight = target_horizon[bucket] / np.maximum(observed_horizon[bucket], 1e-6)
    result = np.clip(category_weight * horizon_weight, 0.1, 10.0)
    return result.astype(np.float32)


def weighted_r2(y: np.ndarray, prediction: np.ndarray, weight: np.ndarray) -> float:
    center = np.average(y, weights=weight)
    numerator = np.sum(weight * (y - prediction) ** 2)
    denominator = np.sum(weight * (y - center) ** 2)
    return float(1 - numerator / denominator)


def _calibration_design(times: np.ndarray, cutoffs: np.ndarray | int) -> np.ndarray:
    cutoff_array = np.asarray(cutoffs, dtype=np.float64)
    horizon = np.clip((times - cutoff_array) / (30.4375 * DAY), 0, 72) / 72
    month_index = times.astype("datetime64[s]").astype("datetime64[M]").astype(np.int64)
    month = month_index % 12
    return np.column_stack((horizon, horizon * horizon, np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12))).astype(np.float32)


def internal_select(events: Events, blocks: FeatureBlocks, debug: bool, log) -> Selection:
    cutoffs = [1375315200, 1493596800]
    fold_predictions = []
    fold_labels = []
    fold_weights = []
    fold_times = []
    fold_cutoffs = []
    fold_records = []
    recency_scores = {False: [], True: []}
    xgb_best = []
    lgb_best = []
    train_times = events.time[:events.n_train]
    rounds = 120 if debug else 500
    train_limit = 100000 if debug else 2000000
    eval_limit = 30000 if debug else 500000
    for fold_number, cutoff in enumerate(cutoffs, start=1):
        fit_all = np.flatnonzero(train_times <= cutoff).astype(np.int64)
        eval_all = np.flatnonzero(train_times > cutoff).astype(np.int64)
        fit_index = _sample(fit_all, train_limit)
        eval_index = _sample(eval_all, eval_limit)
        frozen = build_internal_frozen_features(events, cutoff, eval_index, np.asarray(blocks.residual_tv))
        train_x = assemble(blocks.base, np.asarray(blocks.strict_tv)[fit_index], fit_index)
        eval_x = assemble(blocks.base, frozen, eval_index)
        train_y = events.train_y[fit_index]
        eval_y = events.train_y[eval_index]
        eval_weight = _mixture_weights(frozen, train_times[eval_index], cutoff)
        predictions_by_recency = {}
        best_by_recency = {}
        for weighted in [False, True]:
            train_weight = recency_weights(train_times[fit_index]) if weighted else np.ones(len(fit_index), np.float32)
            started = time.time()
            prediction, best = _xgb_internal(train_x, train_y, train_weight, eval_x, eval_y, eval_weight, rounds)
            score = weighted_r2(eval_y, prediction, eval_weight)
            recency_scores[weighted].append(score)
            predictions_by_recency[weighted] = prediction
            best_by_recency[weighted] = best
            log(f"blackout={fold_number} model=xgb recency={weighted} train={len(fit_index)} eval={len(eval_index)} rounds={best} weighted_r2={score:.6f} seconds={time.time() - started:.1f}")
        started = time.time()
        lgb_prediction, lgb_round = _lgb_internal(train_x, train_y, recency_weights(train_times[fit_index]), eval_x, eval_y, eval_weight, rounds)
        lgb_score = weighted_r2(eval_y, lgb_prediction, eval_weight)
        log(f"blackout={fold_number} model=lightgbm train={len(fit_index)} eval={len(eval_index)} rounds={lgb_round} weighted_r2={lgb_score:.6f} seconds={time.time() - started:.1f}")
        fold_records.append({"fold": fold_number, "cutoff": cutoff, "fit_rows": len(fit_index), "eval_rows": len(eval_index), "xgb_unweighted_r2": recency_scores[False][-1], "xgb_recency_r2": recency_scores[True][-1], "lgb_recency_r2": lgb_score})
        chosen_now = np.mean(recency_scores[True]) >= np.mean(recency_scores[False])
        fold_predictions.append((predictions_by_recency[chosen_now], lgb_prediction))
        fold_labels.append(eval_y)
        fold_weights.append(eval_weight)
        fold_times.append(train_times[eval_index])
        fold_cutoffs.append(np.full(len(eval_index), cutoff, np.int64))
        xgb_best.append(best_by_recency[chosen_now])
        lgb_best.append(lgb_round)
        del frozen, train_x, eval_x
        gc.collect()
    recency_weighted = float(np.mean(recency_scores[True])) >= float(np.mean(recency_scores[False]))
    chosen_predictions = []
    for fold_number, pair in enumerate(fold_predictions):
        if recency_weighted != (np.mean(recency_scores[True][:fold_number + 1]) >= np.mean(recency_scores[False][:fold_number + 1])):
            cutoff = cutoffs[fold_number]
            fit_all = np.flatnonzero(train_times <= cutoff).astype(np.int64)
            eval_all = np.flatnonzero(train_times > cutoff).astype(np.int64)
            fit_index = _sample(fit_all, train_limit)
            eval_index = _sample(eval_all, eval_limit)
            frozen = build_internal_frozen_features(events, cutoff, eval_index, np.asarray(blocks.residual_tv))
            train_x = assemble(blocks.base, np.asarray(blocks.strict_tv)[fit_index], fit_index)
            eval_x = assemble(blocks.base, frozen, eval_index)
            train_weight = recency_weights(train_times[fit_index]) if recency_weighted else np.ones(len(fit_index), np.float32)
            xgb_prediction, best = _xgb_internal(train_x, events.train_y[fit_index], train_weight, eval_x, events.train_y[eval_index], fold_weights[fold_number], rounds)
            xgb_best[fold_number] = best
            chosen_predictions.append((xgb_prediction, pair[1]))
            del frozen, train_x, eval_x
        else:
            chosen_predictions.append(pair)
    prediction_matrix = np.concatenate([np.column_stack(pair) for pair in chosen_predictions])
    labels = np.concatenate(fold_labels)
    weights = np.concatenate(fold_weights)
    blend = Ridge(alpha=10.0, positive=True, fit_intercept=True)
    blend.fit(prediction_matrix, labels, sample_weight=weights)
    coefficients = np.maximum(blend.coef_, 0)
    if coefficients.sum() <= 0:
        coefficients = np.array([0.5, 0.5])
    else:
        coefficients /= coefficients.sum()
    blended = prediction_matrix @ coefficients + float(blend.intercept_)
    times = np.concatenate(fold_times)
    calibration_cutoffs = np.concatenate(fold_cutoffs)
    calibration = Ridge(alpha=100.0, fit_intercept=True)
    calibration.fit(_calibration_design(times, calibration_cutoffs), labels - blended, sample_weight=weights)
    correction = np.clip(calibration.predict(_calibration_design(times, calibration_cutoffs)), -0.05, 0.05)
    calibrated_score = weighted_r2(labels, blended + correction, weights)
    offset = 0
    ensemble_fold_scores = []
    for fold_index, fold_y in enumerate(fold_labels):
        end = offset + len(fold_y)
        fold_prediction = blended[offset:end] + correction[offset:end]
        fold_score = weighted_r2(fold_y, fold_prediction, fold_weights[fold_index])
        fold_records[fold_index]["ensemble_calibrated_r2"] = fold_score
        ensemble_fold_scores.append(fold_score)
        offset = end
    xgb_rounds = 120 if debug else min(2000, max(300, int(np.median(xgb_best) * 1.20)))
    lgb_rounds = 120 if debug else min(1500, max(300, int(np.median(lgb_best) * 1.20)))
    log(f"internal selection recency={recency_weighted} xgb_rounds={xgb_rounds} lgb_rounds={lgb_rounds} weights=({coefficients[0]:.4f},{coefficients[1]:.4f}) intercept={float(blend.intercept_):.5f} calibrated_weighted_r2={calibrated_score:.6f} fold_mean={np.mean(ensemble_fold_scores):.6f} fold_scores={ensemble_fold_scores}")
    return Selection(xgb_rounds=xgb_rounds, lgb_rounds=lgb_rounds, xgb_weight=float(coefficients[0]), lgb_weight=float(coefficients[1]), blend_intercept=float(blend.intercept_), calibration_coef=calibration.coef_.astype(float).tolist(), calibration_intercept=float(calibration.intercept_), recency_weighted=recency_weighted, fold_records=fold_records)


def apply_blend(selection: Selection, xgb_prediction: np.ndarray, lgb_prediction: np.ndarray, times: np.ndarray, cutoff: int) -> np.ndarray:
    prediction = selection.xgb_weight * xgb_prediction + selection.lgb_weight * lgb_prediction + selection.blend_intercept
    design = _calibration_design(times, cutoff)
    correction = design @ np.asarray(selection.calibration_coef) + selection.calibration_intercept
    return np.clip(prediction + np.clip(correction, -0.05, 0.05), 0.1, 5.0).astype(np.float32)


def fit_xgb_final(train_x: np.ndarray, train_y: np.ndarray, train_time: np.ndarray, rounds: int, weighted: bool, model_path: Path):
    weight = recency_weights(train_time) if weighted else np.ones(len(train_y), np.float32)
    data = xgb.DMatrix(train_x, label=train_y, weight=weight, missing=np.nan, nthread=11)
    model = xgb.train(_xgb_params(), data, num_boost_round=rounds, verbose_eval=False)
    model.save_model(model_path)
    del data
    gc.collect()
    return model


def predict_xgb(model, matrix: np.ndarray) -> np.ndarray:
    data = xgb.DMatrix(matrix, missing=np.nan, nthread=11)
    prediction = model.predict(data)
    del data
    gc.collect()
    return prediction.astype(np.float32)


def fit_lgb_final(train_x: np.ndarray, train_y: np.ndarray, train_time: np.ndarray, rounds: int, weighted: bool, model_path: Path):
    weight = recency_weights(train_time) if weighted else np.ones(len(train_y), np.float32)
    data = lgb.Dataset(train_x, label=train_y, weight=weight, free_raw_data=True)
    model = lgb.train(_lgb_params(), data, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    model.save_model(str(model_path))
    del data
    gc.collect()
    return model


def predict_lgb(model, matrix: np.ndarray) -> np.ndarray:
    return model.predict(matrix).astype(np.float32)
