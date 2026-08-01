from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from scipy.special import ndtr
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


CAT_COLUMNS = ["country_id", "state_id", "type_id", "origin_month", "cx_last_style_id", "cx_last_category_id", "cx_mode_style_id", "cx_mode_category_id", "cx_recent_mode_style_id", "cx_recent_mode_category_id"]


@dataclass
class Selection:
    half_life: float | None
    lgb_rounds: int
    lgb_seed_ensemble: bool
    lgb_rounds_2: int
    keep_cat: bool
    cat_rounds: int
    keep_aft: bool
    aft_rounds: int
    aft_distribution: str
    aft_scale: float
    blend_weights: list[float]
    blend_mode: str
    keep_context: bool
    diagnostics: dict


def sample_weights(episodes: pd.DataFrame, half_life: float | None, anchor=None) -> np.ndarray:
    result = episodes["base_weight"].to_numpy(dtype=np.float64).copy()
    if half_life is not None:
        timestamp = pd.to_datetime(episodes["timestamp"])
        endpoint = pd.Timestamp(anchor) if anchor is not None else timestamp.max()
        age = (endpoint - timestamp).dt.total_seconds().to_numpy() / 86_400.0 / 365.25
        result *= np.power(0.5, np.maximum(age, 0.0) / half_life)
    result *= len(result) / max(result.sum(), 1.0)
    return result.astype(np.float32)


def lgb_params(seed: int = 1337) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.03,
        "num_leaves": 63,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "num_threads": 11,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "force_col_wise": True,
    }


def train_lightgbm(X: pd.DataFrame, y: np.ndarray, weight: np.ndarray, rounds: int, validation=None, seed: int = 1337):
    categorical = [column for column in CAT_COLUMNS if column in X.columns]
    train_set = lgb.Dataset(X, label=y, weight=weight, categorical_feature=categorical, free_raw_data=True)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if validation is not None:
        X_valid, y_valid = validation
        valid_sets = [lgb.Dataset(X_valid, label=y_valid, categorical_feature=categorical, reference=train_set, free_raw_data=True)]
        callbacks.append(lgb.early_stopping(70, verbose=False))
    model = lgb.train(lgb_params(seed), train_set, num_boost_round=rounds, valid_sets=valid_sets, callbacks=callbacks)
    return model


def cat_pool(X: pd.DataFrame, y=None, weight=None) -> Pool:
    return Pool(X, label=y, weight=weight, cat_features=[X.columns.get_loc(column) for column in CAT_COLUMNS if column in X.columns])


def train_catboost(X: pd.DataFrame, y: np.ndarray, weight: np.ndarray, rounds: int, validation=None, seed: int = 1337):
    model = CatBoostClassifier(
        iterations=rounds,
        depth=8,
        learning_rate=0.04,
        l2_leaf_reg=5.0,
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        bootstrap_type="Bernoulli",
        subsample=0.8,
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    eval_set = cat_pool(validation[0], validation[1]) if validation is not None else None
    model.fit(cat_pool(X, y, weight), eval_set=eval_set, use_best_model=validation is not None, early_stopping_rounds=60 if validation is not None else None, verbose=False)
    return model


def aft_matrix(X: pd.DataFrame, duration: np.ndarray | None = None, dormant: np.ndarray | None = None, weight: np.ndarray | None = None) -> xgb.DMatrix:
    matrix = xgb.DMatrix(X, weight=weight, nthread=11)
    if duration is not None and dormant is not None:
        lower = np.asarray(duration, dtype=np.float32)
        upper = lower.copy()
        upper[np.asarray(dormant, dtype=bool)] = np.inf
        matrix.set_float_info("label_lower_bound", lower)
        matrix.set_float_info("label_upper_bound", upper)
    return matrix


def aft_params(distribution: str, scale: float, seed: int = 1337) -> dict:
    return {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": distribution,
        "aft_loss_distribution_scale": scale,
        "learning_rate": 0.03,
        "max_depth": 6,
        "min_child_weight": 100.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 5.0,
        "tree_method": "hist",
        "device": "cuda",
        "verbosity": 0,
        "seed": seed,
        "nthread": 11,
    }


def train_aft(X: pd.DataFrame, duration: np.ndarray, dormant: np.ndarray, weight: np.ndarray, rounds: int, distribution: str, scale: float, validation=None, seed: int = 1337):
    train_matrix = aft_matrix(X, duration, dormant, weight)
    evaluations = []
    if validation is not None:
        valid_matrix = aft_matrix(validation[0], validation[1], validation[2])
        evaluations = [(valid_matrix, "valid")]
    model = xgb.train(
        aft_params(distribution, scale, seed),
        train_matrix,
        num_boost_round=rounds,
        evals=evaluations,
        early_stopping_rounds=45 if validation is not None else None,
        verbose_eval=False,
    )
    return model


def predict_aft(model, X: pd.DataFrame, distribution: str, scale: float) -> np.ndarray:
    location_time = np.maximum(model.predict(aft_matrix(X)), 1e-5)
    z = (math.log(365.0) - np.log(location_time)) / scale
    if distribution == "logistic":
        prediction = 1.0 / (1.0 + np.exp(np.clip(z, -30, 30)))
    else:
        prediction = ndtr(-z)
    return np.asarray(prediction, dtype=np.float64)


def folds_for(episodes: pd.DataFrame) -> list[tuple[pd.Timestamp, np.ndarray, np.ndarray]]:
    timestamps = pd.to_datetime(episodes["timestamp"])
    desired = [pd.Timestamp("2015-09-02"), pd.Timestamp("2016-01-01"), pd.Timestamp("2016-09-01"), pd.Timestamp("2017-01-01"), pd.Timestamp("2017-09-01")]
    result = []
    for origin in desired:
        valid = np.flatnonzero(timestamps.to_numpy() == np.datetime64(origin))
        train = np.flatnonzero((timestamps + pd.Timedelta(days=365) <= origin).to_numpy())
        if len(valid) >= 500 and len(train) >= 1000:
            result.append((origin, train, valid))
    return result


def fold_mean(prediction: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray) -> tuple[float, list[float]]:
    values = []
    for fold in np.unique(fold_ids):
        mask = fold_ids == fold
        values.append(float(roc_auc_score(labels[mask], prediction[mask])))
    return float(np.mean(values)), values


def optimize_blend(predictions: list[np.ndarray], labels: np.ndarray, fold_ids: np.ndarray) -> tuple[list[float], float, list[float]]:
    count = len(predictions)
    candidates = []
    if count == 1:
        candidates = [(1.0,)]
    elif count == 2:
        candidates = [(i / 40.0, 1.0 - i / 40.0) for i in range(41)]
    else:
        for i in range(21):
            for j in range(21 - i):
                candidates.append((i / 20.0, j / 20.0, 1.0 - (i + j) / 20.0))
    best_weights = None
    best_mean = -np.inf
    best_folds = []
    for weights in candidates:
        blended = sum(weight * prediction for weight, prediction in zip(weights, predictions))
        mean, scores = fold_mean(blended, labels, fold_ids)
        if mean > best_mean + 1e-12:
            best_weights = list(weights)
            best_mean = mean
            best_folds = scores
    return best_weights, best_mean, best_folds


def transform_oof(predictions: list[np.ndarray], fold_ids: np.ndarray, mode: str) -> list[np.ndarray]:
    if mode == "raw":
        return predictions
    if mode == "logit":
        return [np.log(np.clip(value, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - value, 1e-6, 1.0)) for value in predictions]
    transformed = []
    for value in predictions:
        ranks = np.empty(len(value), dtype=np.float64)
        for fold in np.unique(fold_ids):
            mask = fold_ids == fold
            ranks[mask] = rankdata(value[mask], method="average") / (np.sum(mask) + 1.0)
        transformed.append(ranks)
    return transformed


def optimize_blend_modes(predictions: list[np.ndarray], labels: np.ndarray, fold_ids: np.ndarray) -> tuple[list[float], float, list[float], str]:
    results = []
    for mode in ["raw", "logit", "rank"]:
        weights, mean, scores = optimize_blend(transform_oof(predictions, fold_ids, mode), labels, fold_ids)
        results.append((weights, mean, scores, mode))
    best_mean = max(value[1] for value in results)
    for result in results:
        if result[1] >= best_mean - 0.0001:
            return result
    return max(results, key=lambda value: value[1])


def debug_selection() -> Selection:
    return Selection(
        half_life=5.0,
        lgb_rounds=75,
        lgb_seed_ensemble=False,
        lgb_rounds_2=75,
        keep_cat=True,
        cat_rounds=60,
        keep_aft=True,
        aft_rounds=60,
        aft_distribution="logistic",
        aft_scale=1.0,
        blend_weights=[0.60, 0.25, 0.15],
        blend_mode="raw",
        keep_context=True,
        diagnostics={"mode": "debug", "selection": "fixed pipeline exercise"},
    )


def select_design(X: pd.DataFrame, episodes: pd.DataFrame, debug: bool) -> Selection:
    if debug:
        return debug_selection()
    folds = folds_for(episodes)
    if len(folds) < 4:
        raise RuntimeError(f"insufficient forward folds: {len(folds)}")
    y_all = episodes["dormant"].to_numpy(dtype=np.int8)
    half_lives = [3.0, 5.0, None]
    lgb_results = {}
    for half_life in half_lives:
        predictions = []
        labels = []
        fold_ids = []
        rounds = []
        scores = []
        for fold_number, (origin, train_index, valid_index) in enumerate(folds):
            weights = sample_weights(episodes.iloc[train_index], half_life, anchor=origin)
            model = train_lightgbm(X.iloc[train_index], y_all[train_index], weights, 900, validation=(X.iloc[valid_index], y_all[valid_index]), seed=1337 + fold_number)
            prediction = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
            predictions.append(prediction)
            labels.append(y_all[valid_index])
            fold_ids.append(np.full(len(valid_index), fold_number, dtype=np.int8))
            rounds.append(int(model.best_iteration or model.current_iteration()))
            scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
        lgb_results[half_life] = {
            "prediction": np.concatenate(predictions),
            "labels": np.concatenate(labels),
            "fold_ids": np.concatenate(fold_ids),
            "rounds": rounds,
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
        print(f"[forward] lightgbm half_life={half_life} mean={np.mean(scores):.6f} std={np.std(scores):.6f} folds={scores}", flush=True)
    best_mean = max(item["mean"] for item in lgb_results.values())
    preference = [5.0, 3.0, None]
    selected_half_life = next(value for value in preference if lgb_results[value]["mean"] >= best_mean - 0.002)
    base = lgb_results[selected_half_life]
    keep_context = True
    context_diagnostics = {}
    context_columns = [column for column in X.columns if column.startswith("cx_")]
    if context_columns:
        baseline_X = X.drop(columns=context_columns)
        predictions = []
        labels = []
        fold_ids = []
        rounds = []
        scores = []
        for fold_number, (origin, train_index, valid_index) in enumerate(folds):
            weights = sample_weights(episodes.iloc[train_index], selected_half_life, anchor=origin)
            model = train_lightgbm(baseline_X.iloc[train_index], y_all[train_index], weights, 900, validation=(baseline_X.iloc[valid_index], y_all[valid_index]), seed=5137 + fold_number)
            prediction = model.predict(baseline_X.iloc[valid_index], num_iteration=model.best_iteration)
            predictions.append(prediction)
            labels.append(y_all[valid_index])
            fold_ids.append(np.full(len(valid_index), fold_number, dtype=np.int8))
            rounds.append(int(model.best_iteration or model.current_iteration()))
            scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
        baseline = {"prediction": np.concatenate(predictions), "labels": np.concatenate(labels), "fold_ids": np.concatenate(fold_ids), "rounds": rounds, "scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
        improved = sum(new > old for new, old in zip(base["scores"], baseline["scores"]))
        keep_context = bool(improved >= 2 and base["mean"] > baseline["mean"] + 0.0001)
        context_diagnostics = {"context_full_mean": base["mean"], "context_baseline_mean": baseline["mean"], "context_improved_folds": improved, "context_baseline_scores": scores}
        print(f"[forward] context full={base['mean']:.6f} baseline={baseline['mean']:.6f} improved_folds={improved} keep={keep_context}", flush=True)
        if not keep_context:
            X = baseline_X
            base = baseline
    second_predictions = []
    second_rounds = []
    second_scores = []
    for fold_number, (origin, train_index, valid_index) in enumerate(folds):
        weights = sample_weights(episodes.iloc[train_index], selected_half_life, anchor=origin)
        model = train_lightgbm(X.iloc[train_index], y_all[train_index], weights, 900, validation=(X.iloc[valid_index], y_all[valid_index]), seed=6137 + fold_number)
        prediction = model.predict(X.iloc[valid_index], num_iteration=model.best_iteration)
        second_predictions.append(prediction)
        second_rounds.append(int(model.best_iteration or model.current_iteration()))
        second_scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
    second_prediction = np.concatenate(second_predictions)
    single_lgb_prediction = base["prediction"].copy()
    single_lgb_scores = list(base["scores"])
    single_lgb_mean = float(base["mean"])
    ensemble_prediction = 0.5 * base["prediction"] + 0.5 * second_prediction
    ensemble_mean, ensemble_scores = fold_mean(ensemble_prediction, base["labels"], base["fold_ids"])
    seed_improvements = sum(new > old for new, old in zip(ensemble_scores, base["scores"]))
    lgb_seed_ensemble = bool(seed_improvements >= 2 and ensemble_mean > base["mean"] + 0.0001)
    seed_diagnostics = {"single_mean": base["mean"], "second_mean": float(np.mean(second_scores)), "ensemble_mean": ensemble_mean, "ensemble_scores": ensemble_scores, "improved_folds": seed_improvements}
    print(f"[forward] lightgbm seed ensemble single={base['mean']:.6f} second={np.mean(second_scores):.6f} ensemble={ensemble_mean:.6f} improved_folds={seed_improvements} keep={lgb_seed_ensemble}", flush=True)
    if lgb_seed_ensemble:
        base["prediction"] = ensemble_prediction
        base["scores"] = ensemble_scores
        base["mean"] = ensemble_mean
        base["std"] = float(np.std(ensemble_scores))
    lgb_rounds = int(np.median(base["rounds"]))
    lgb_rounds_2 = int(np.median(second_rounds))
    cat_predictions = []
    cat_round_values = []
    cat_scores = []
    for fold_number, (origin, train_index, valid_index) in enumerate(folds):
        weights = sample_weights(episodes.iloc[train_index], selected_half_life, anchor=origin)
        model = train_catboost(X.iloc[train_index], y_all[train_index], weights, 750, validation=(X.iloc[valid_index], y_all[valid_index]), seed=2137 + fold_number)
        prediction = model.predict_proba(cat_pool(X.iloc[valid_index]))[:, 1]
        cat_predictions.append(prediction)
        cat_round_values.append(int(model.get_best_iteration() + 1 if model.get_best_iteration() >= 0 else 750))
        cat_scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
    cat_prediction = np.concatenate(cat_predictions)
    two_weights, two_mean, two_folds, two_mode = optimize_blend_modes([base["prediction"], cat_prediction], base["labels"], base["fold_ids"])
    cat_improvements = sum(new > old for new, old in zip(two_folds, base["scores"]))
    keep_cat = bool(cat_improvements >= 2 and two_mean > base["mean"] + 0.0001 and two_weights[1] > 0)
    print(f"[forward] catboost mean={np.mean(cat_scores):.6f} blend={two_mean:.6f} mode={two_mode} weights={two_weights} improved_folds={cat_improvements} keep={keep_cat}", flush=True)
    stage_predictions = [base["prediction"]]
    if keep_cat:
        stage_predictions.append(cat_prediction)
    tuning_fold_numbers = [max(0, len(folds) - 2), len(folds) - 1]
    configurations = list(itertools.product(["logistic", "normal"], [0.75, 1.0, 1.5]))
    config_scores = {}
    config_models = {}
    for distribution, scale in configurations:
        scores = []
        for fold_number in tuning_fold_numbers:
            origin, train_index, valid_index = folds[fold_number]
            model = train_aft(
                X.iloc[train_index],
                episodes.iloc[train_index]["duration"].to_numpy(),
                y_all[train_index],
                sample_weights(episodes.iloc[train_index], selected_half_life, anchor=origin),
                500,
                distribution,
                scale,
                validation=(X.iloc[valid_index], episodes.iloc[valid_index]["duration"].to_numpy(), y_all[valid_index]),
                seed=3137 + fold_number,
            )
            prediction = predict_aft(model, X.iloc[valid_index], distribution, scale)
            scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
            config_models[(distribution, scale, fold_number)] = (prediction, int(getattr(model, "best_iteration", 499) + 1))
        config_scores[(distribution, scale)] = float(np.mean(scores))
        print(f"[forward] aft distribution={distribution} scale={scale} tuning_mean={np.mean(scores):.6f}", flush=True)
    aft_distribution, aft_scale = max(configurations, key=lambda value: config_scores[value])
    aft_predictions = []
    aft_round_values = []
    aft_scores = []
    for fold_number, (origin, train_index, valid_index) in enumerate(folds):
        cached = config_models.get((aft_distribution, aft_scale, fold_number))
        if cached is None:
            model = train_aft(
                X.iloc[train_index],
                episodes.iloc[train_index]["duration"].to_numpy(),
                y_all[train_index],
                sample_weights(episodes.iloc[train_index], selected_half_life, anchor=origin),
                500,
                aft_distribution,
                aft_scale,
                validation=(X.iloc[valid_index], episodes.iloc[valid_index]["duration"].to_numpy(), y_all[valid_index]),
                seed=4137 + fold_number,
            )
            prediction = predict_aft(model, X.iloc[valid_index], aft_distribution, aft_scale)
            best_round = int(getattr(model, "best_iteration", 499) + 1)
        else:
            prediction, best_round = cached
        aft_predictions.append(prediction)
        aft_round_values.append(best_round)
        aft_scores.append(float(roc_auc_score(y_all[valid_index], prediction)))
    aft_prediction = np.concatenate(aft_predictions)
    candidate_predictions = stage_predictions + [aft_prediction]
    final_weights, final_mean, final_folds, final_mode = optimize_blend_modes(candidate_predictions, base["labels"], base["fold_ids"])
    previous_weights, previous_mean, previous_folds, previous_mode = optimize_blend_modes(stage_predictions, base["labels"], base["fold_ids"])
    aft_improvements = sum(new > old for new, old in zip(final_folds, previous_folds))
    keep_aft = bool(aft_improvements >= 2 and final_mean > previous_mean + 0.0001 and final_weights[-1] > 0)
    if not keep_aft:
        final_weights, final_mean, final_folds, final_mode = previous_weights, previous_mean, previous_folds, previous_mode
    seed_full_diagnostics = {}
    if lgb_seed_ensemble:
        single_stage_predictions = [single_lgb_prediction]
        if keep_cat:
            single_stage_predictions.append(cat_prediction)
        if keep_aft:
            single_stage_predictions.append(aft_prediction)
        single_weights, single_mean, single_folds, single_mode = optimize_blend_modes(single_stage_predictions, base["labels"], base["fold_ids"])
        seed_full_improvements = sum(new > old for new, old in zip(final_folds, single_folds))
        seed_full_diagnostics = {"seed_ensemble_mean": final_mean, "single_seed_mean": single_mean, "seed_ensemble_improved_folds": seed_full_improvements}
        keep_seed_full = bool(seed_full_improvements >= 2 and final_mean > single_mean + 0.0001)
        print(f"[forward] full seed gate ensemble={final_mean:.6f} single={single_mean:.6f} improved_folds={seed_full_improvements} keep={keep_seed_full}", flush=True)
        if not keep_seed_full:
            lgb_seed_ensemble = False
            base["prediction"] = single_lgb_prediction
            base["scores"] = single_lgb_scores
            base["mean"] = single_lgb_mean
            final_weights, final_mean, final_folds, final_mode = single_weights, single_mean, single_folds, single_mode
    print(f"[forward] aft mean={np.mean(aft_scores):.6f} blend={final_mean:.6f} mode={final_mode} weights={final_weights} improved_folds={aft_improvements} keep={keep_aft}", flush=True)
    diagnostics = {
        "fold_origins": [str(origin) for origin, _, _ in folds],
        "fold_counts": [int(len(valid)) for _, _, valid in folds],
        "lightgbm": {str(key): {k: v for k, v in value.items() if k in ["rounds", "scores", "mean", "std"]} for key, value in lgb_results.items()},
        "selected_half_life": selected_half_life,
        "cat_scores": cat_scores,
        "cat_blend_mean": two_mean,
        "cat_improved_folds": cat_improvements,
        "aft_configuration_scores": {f"{key[0]}_{key[1]}": value for key, value in config_scores.items()},
        "aft_scores": aft_scores,
        "aft_improved_folds": aft_improvements,
        "selected_fold_scores": final_folds,
        "selected_mean": final_mean,
        "selected_blend_mode": final_mode,
        "context_gate": context_diagnostics,
        "lightgbm_seed_gate": seed_diagnostics,
        "full_seed_gate": seed_full_diagnostics,
    }
    return Selection(
        half_life=selected_half_life,
        lgb_rounds=max(100, lgb_rounds),
        lgb_seed_ensemble=lgb_seed_ensemble,
        lgb_rounds_2=max(100, lgb_rounds_2),
        keep_cat=keep_cat,
        cat_rounds=max(100, int(np.median(cat_round_values))),
        keep_aft=keep_aft,
        aft_rounds=max(100, int(np.median(aft_round_values))),
        aft_distribution=aft_distribution,
        aft_scale=float(aft_scale),
        blend_weights=final_weights,
        blend_mode=final_mode,
        keep_context=keep_context,
        diagnostics=diagnostics,
    )


def fit_selected(X: pd.DataFrame, episodes: pd.DataFrame, selection: Selection, seed: int) -> list:
    if not selection.keep_context:
        X = X[[column for column in X.columns if not column.startswith("cx_")]]
    y = episodes["dormant"].to_numpy(dtype=np.int8)
    weight = sample_weights(episodes, selection.half_life)
    models = [("lightgbm_1", train_lightgbm(X, y, weight, selection.lgb_rounds, seed=seed))]
    if selection.lgb_seed_ensemble:
        models.append(("lightgbm_2", train_lightgbm(X, y, weight, selection.lgb_rounds_2, seed=seed + 1000)))
    if selection.keep_cat:
        models.append(("catboost", train_catboost(X, y, weight, selection.cat_rounds, seed=seed + 100)))
    if selection.keep_aft:
        models.append(("aft", train_aft(X, episodes["duration"].to_numpy(), y, weight, selection.aft_rounds, selection.aft_distribution, selection.aft_scale, seed=seed + 200)))
    return models


def predict_selected(models: list, X: pd.DataFrame, selection: Selection) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not selection.keep_context:
        X = X[[column for column in X.columns if not column.startswith("cx_")]]
    predictions = {}
    lightgbm_predictions = []
    for name, model in models:
        if name.startswith("lightgbm"):
            lightgbm_predictions.append(np.asarray(model.predict(X), dtype=np.float64))
        elif name == "catboost":
            predictions[name] = np.asarray(model.predict_proba(cat_pool(X))[:, 1], dtype=np.float64)
        else:
            predictions[name] = predict_aft(model, X, selection.aft_distribution, selection.aft_scale)
    predictions["lightgbm"] = np.mean(lightgbm_predictions, axis=0)
    ordered = [predictions[name] for name in ["lightgbm", "catboost", "aft"] if name in predictions]
    if selection.blend_mode == "logit":
        ordered = [np.log(np.clip(value, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - value, 1e-6, 1.0)) for value in ordered]
    elif selection.blend_mode == "rank":
        ordered = [rankdata(value, method="average") / (len(value) + 1.0) for value in ordered]
    weights = np.asarray(selection.blend_weights, dtype=np.float64)
    if len(weights) != len(ordered):
        weights = np.full(len(ordered), 1.0 / len(ordered))
    weights = weights / weights.sum()
    blended = sum(weight * prediction for weight, prediction in zip(weights, ordered))
    if selection.blend_mode == "logit":
        blended = 1.0 / (1.0 + np.exp(-np.clip(blended, -30, 30)))
    return np.clip(blended, 1e-6, 1.0 - 1e-6), predictions
