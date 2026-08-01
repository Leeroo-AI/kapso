from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from relational_features import CACHE_VERSION, feature_columns


SEEDS = (17, 41)
MAX_ROUNDS = 1800


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "count": int(len(y)),
        "r2": float(r2_score(y, prediction)),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
    }


def recency_weights(origins: pd.Series, prediction_origin: pd.Timestamp, mode: str) -> np.ndarray:
    if mode == "uniform":
        return np.ones(len(origins), dtype=np.float32)
    age = (pd.Timestamp(prediction_origin) - pd.to_datetime(origins)).dt.days.to_numpy(dtype=np.float32)
    weight = np.power(2.0, -age / 730.0)
    return (weight / max(float(weight.mean()), 1e-6)).astype(np.float32)


def target_for_objective(y: np.ndarray, objective: str) -> np.ndarray:
    if objective == "log":
        return np.log1p(np.maximum(y - 1.0, 0.0))
    if objective in {"poisson", "tweedie"}:
        return np.maximum(y - 1.0, 0.0)
    return y


def restore_prediction(prediction: np.ndarray, objective: str) -> np.ndarray:
    if objective == "log":
        return np.expm1(prediction) + 1.0
    if objective in {"poisson", "tweedie"}:
        return prediction + 1.0
    return prediction


def lgb_parameters(objective: str, seed: int) -> dict:
    metric = "l2"
    lgb_objective = "regression"
    if objective == "poisson":
        lgb_objective = "poisson"
        metric = "poisson"
    if objective == "tweedie":
        lgb_objective = "tweedie"
        metric = "tweedie"
    return {
        "objective": lgb_objective,
        "metric": metric,
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "tweedie_variance_power": 1.35,
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "data_random_seed": seed,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "verbosity": -1,
        "force_col_wise": True,
    }


def fit_lgb(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    prediction_x: pd.DataFrame,
    objective: str,
    seed: int,
    rounds: int,
    weights: np.ndarray | None = None,
    validation_x: pd.DataFrame | None = None,
    validation_y: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    transformed = target_for_objective(train_y, objective)
    dataset = lgb.Dataset(train_x, label=transformed, weight=weights, free_raw_data=False)
    callbacks = [lgb.log_evaluation(0)]
    valid_sets = None
    if validation_x is not None and validation_y is not None:
        valid = lgb.Dataset(validation_x, label=target_for_objective(validation_y, objective), reference=dataset, free_raw_data=False)
        valid_sets = [valid]
        callbacks.append(lgb.early_stopping(100, verbose=False))
    model = lgb.train(
        lgb_parameters(objective, seed),
        dataset,
        num_boost_round=int(rounds),
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    best_iteration = int(model.best_iteration or rounds)
    prediction = model.predict(prediction_x, num_iteration=best_iteration)
    return restore_prediction(np.asarray(prediction, dtype=np.float64), objective), best_iteration


def fit_catboost(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    prediction_x: pd.DataFrame,
    seed: int,
    rounds: int,
    weights: np.ndarray | None = None,
    validation_x: pd.DataFrame | None = None,
    validation_y: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    model = CatBoostRegressor(
        iterations=int(rounds),
        learning_rate=0.04,
        depth=8,
        loss_function="RMSE",
        l2_leaf_reg=5.0,
        random_seed=seed,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "1")),
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )
    kwargs = {"sample_weight": weights}
    if validation_x is not None and validation_y is not None:
        kwargs["eval_set"] = (validation_x, validation_y)
        kwargs["use_best_model"] = True
    model.fit(train_x, train_y, **kwargs)
    best_value = model.get_best_iteration()
    best = int(best_value) if best_value is not None else int(rounds)
    if best < 1:
        best = int(rounds)
    return np.asarray(model.predict(prediction_x), dtype=np.float64), best


def fold_prediction(
    data: pd.DataFrame,
    held_origin: pd.Timestamp,
    gap_days: int,
    columns: list[str],
    objective: str,
    weight_mode: str,
    max_rounds: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    held_origin = pd.Timestamp(held_origin)
    train_limit = held_origin - pd.Timedelta(days=gap_days)
    training = data.loc[pd.to_datetime(data["origin"]).le(train_limit)]
    held = data.loc[pd.to_datetime(data["origin"]).eq(held_origin)]
    if len(training) == 0 or len(held) == 0:
        raise RuntimeError(f"empty forward fold for {held_origin}")
    weights = recency_weights(training["origin"], held_origin, weight_mode)
    if objective == "catboost":
        prediction, best = fit_catboost(
            training[columns],
            training["publication_count"].to_numpy(dtype=np.float64),
            held[columns],
            17,
            max_rounds,
            weights,
            held[columns],
            held["publication_count"].to_numpy(dtype=np.float64),
        )
    else:
        prediction, best = fit_lgb(
            training[columns],
            training["publication_count"].to_numpy(dtype=np.float64),
            held[columns],
            objective,
            17,
            max_rounds,
            weights,
            held[columns],
            held["publication_count"].to_numpy(dtype=np.float64),
        )
    return held.index.to_numpy(), held["publication_count"].to_numpy(dtype=np.float64), prediction, best


def run_oof(
    data: pd.DataFrame,
    folds: list[pd.Timestamp],
    fold_gaps: list[int],
    columns: list[str],
    objective: str,
    weight_mode: str,
    max_rounds: int = MAX_ROUNDS,
) -> dict:
    indices = []
    labels = []
    predictions = []
    fold_ids = []
    best_rounds = []
    fold_metrics = []
    for fold_index, (held_origin, gap_days) in enumerate(zip(folds, fold_gaps)):
        index, y, prediction, best = fold_prediction(data, held_origin, gap_days, columns, objective, weight_mode, max_rounds)
        prediction = np.maximum(prediction, 1.0)
        indices.append(index)
        labels.append(y)
        predictions.append(prediction)
        fold_ids.append(np.full(len(y), fold_index, dtype=np.int8))
        best_rounds.append(best)
        result = metrics(y, prediction)
        result["origin"] = str(pd.Timestamp(held_origin).date())
        result["gap_days"] = int(gap_days)
        result["best_round"] = int(best)
        fold_metrics.append(result)
    return {
        "indices": np.concatenate(indices),
        "y": np.concatenate(labels),
        "prediction": np.concatenate(predictions),
        "fold_id": np.concatenate(fold_ids),
        "best_rounds": best_rounds,
        "fold_metrics": fold_metrics,
    }


def stable_improvement(candidate: dict, baseline: dict) -> tuple[bool, float, float]:
    candidate_scores = np.array([fold["r2"] for fold in candidate["fold_metrics"]], dtype=np.float64)
    baseline_scores = np.array([fold["r2"] for fold in baseline["fold_metrics"]], dtype=np.float64)
    differences = candidate_scores - baseline_scores
    standard_error = float(differences.std(ddof=1) / math.sqrt(len(differences))) if len(differences) > 1 else 0.0
    mean = float(differences.mean())
    return bool(mean > standard_error and np.sum(differences > 0) >= len(differences) - 1), mean, standard_error


def crossfit_affine(predictions: np.ndarray, y: np.ndarray, fold_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    crossfit = np.zeros(len(y), dtype=np.float64)
    for fold in np.unique(fold_id):
        train = fold_id != fold
        held = fold_id == fold
        model = LinearRegression(positive=True, fit_intercept=True)
        model.fit(predictions[train], y[train])
        crossfit[held] = model.predict(predictions[held])
    final = LinearRegression(positive=True, fit_intercept=True)
    final.fit(predictions, y)
    return np.maximum(crossfit, 1.0), np.asarray(final.coef_, dtype=np.float64), float(final.intercept_)


def crossfit_segmented_affine(predictions: np.ndarray, y: np.ndarray, fold_id: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, dict]:
    crossfit = np.zeros(len(y), dtype=np.float64)
    parameters = {}
    for segment in np.unique(segments):
        segment_mask = segments == segment
        for fold in np.unique(fold_id):
            train = (fold_id != fold) & segment_mask
            held = (fold_id == fold) & segment_mask
            model = LinearRegression(positive=True, fit_intercept=True)
            model.fit(predictions[train], y[train])
            crossfit[held] = model.predict(predictions[held])
        final = LinearRegression(positive=True, fit_intercept=True)
        final.fit(predictions[segment_mask], y[segment_mask])
        parameters[str(int(segment))] = {"weights": np.asarray(final.coef_, dtype=np.float64).tolist(), "intercept": float(final.intercept_)}
    return np.maximum(crossfit, 1.0), parameters


def score_by_fold(y: np.ndarray, prediction: np.ndarray, fold_id: np.ndarray, folds: list[pd.Timestamp]) -> list[dict]:
    results = []
    for fold, origin in enumerate(folds):
        selected = fold_id == fold
        result = metrics(y[selected], prediction[selected])
        result["origin"] = str(pd.Timestamp(origin).date())
        results.append(result)
    return results


def select_pipeline(data: pd.DataFrame, folds: list[pd.Timestamp], cache_dir: Path) -> dict:
    selection_path = cache_dir / f"selection_{CACHE_VERSION}_v10.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        print(f"[selection] reused {selection_path.name}: {json.dumps(selection, sort_keys=True)}", flush=True)
        return selection
    sample = data.iloc[:1]
    core_columns = feature_columns(sample, "core")
    full_columns = feature_columns(sample, "full")
    configurations = [
        ("core_uniform", core_columns, "uniform"),
        ("full_uniform", full_columns, "uniform"),
        ("full_recency", full_columns, "recency"),
    ]
    fold_gaps = [183, 365, 183, 365]
    if len(folds) != len(fold_gaps):
        raise RuntimeError("selection requires four configured forward gaps")
    raw_results = {}
    for name, columns, weight_mode in configurations:
        started = time.time()
        result = run_oof(data, folds, fold_gaps, columns, "raw", weight_mode)
        raw_results[name] = result
        print(f"[selection] checkpoint={name} features={len(columns)} seconds={time.time()-started:.1f} folds={json.dumps(result['fold_metrics'])}", flush=True)
    chosen_name = "core_uniform"
    block_comparisons = []
    for candidate_name in ("full_uniform", "full_recency"):
        stable, gain, standard_error = stable_improvement(raw_results[candidate_name], raw_results[chosen_name])
        block_comparisons.append({"candidate": candidate_name, "baseline": chosen_name, "gain": gain, "standard_error": standard_error, "stable": stable})
        if stable:
            chosen_name = candidate_name
    chosen_columns = full_columns if chosen_name.startswith("full") else core_columns
    weight_mode = "recency" if chosen_name.endswith("recency") else "uniform"
    objective_results = {"raw": raw_results[chosen_name]}
    for objective in ("log", "poisson", "tweedie"):
        started = time.time()
        result = run_oof(data, folds, fold_gaps, chosen_columns, objective, weight_mode)
        objective_results[objective] = result
        residual_correlation = float(np.corrcoef(result["y"] - result["prediction"], objective_results["raw"]["y"] - objective_results["raw"]["prediction"])[0, 1])
        print(f"[selection] objective={objective} residual_correlation={residual_correlation:.6f} seconds={time.time()-started:.1f} folds={json.dumps(result['fold_metrics'])}", flush=True)
    raw = objective_results["raw"]
    candidate_objectives = ["raw"]
    residual_correlations = {}
    for objective in ("log", "poisson", "tweedie"):
        correlation = float(np.corrcoef(raw["y"] - raw["prediction"], objective_results[objective]["y"] - objective_results[objective]["prediction"])[0, 1])
        residual_correlations[objective] = correlation
        if correlation < 0.995:
            candidate_objectives.append(objective)
    prediction_matrix = np.column_stack([objective_results[objective]["prediction"] for objective in candidate_objectives])
    blend_prediction, blend_weights, blend_intercept = crossfit_affine(prediction_matrix, raw["y"], raw["fold_id"])
    blend_folds = score_by_fold(raw["y"], blend_prediction, raw["fold_id"], folds)
    blend_result = {"fold_metrics": blend_folds}
    stable_blend, blend_gain, blend_standard_error = stable_improvement(blend_result, raw)
    catboost_evaluated = False
    if len(candidate_objectives) > 1 and min(residual_correlations.values()) < 0.97:
        catboost_evaluated = True
        catboost_result = run_oof(data, folds, fold_gaps, chosen_columns, "catboost", weight_mode, 1200)
        catboost_correlation = float(np.corrcoef(raw["y"] - raw["prediction"], catboost_result["y"] - catboost_result["prediction"])[0, 1])
        residual_correlations["catboost"] = catboost_correlation
        print(f"[selection] objective=catboost residual_correlation={catboost_correlation:.6f} folds={json.dumps(catboost_result['fold_metrics'])}", flush=True)
        if catboost_correlation < 0.995:
            objective_results["catboost"] = catboost_result
            candidate_objectives.append("catboost")
            prediction_matrix = np.column_stack([objective_results[objective]["prediction"] for objective in candidate_objectives])
            blend_prediction, blend_weights, blend_intercept = crossfit_affine(prediction_matrix, raw["y"], raw["fold_id"])
            blend_folds = score_by_fold(raw["y"], blend_prediction, raw["fold_id"], folds)
            blend_result = {"fold_metrics": blend_folds}
            stable_blend, blend_gain, blend_standard_error = stable_improvement(blend_result, raw)
    if stable_blend:
        keep = blend_weights > 0.01
        objectives = [objective for objective, selected in zip(candidate_objectives, keep) if selected]
        weights = blend_weights[keep]
        reduced_matrix = np.column_stack([objective_results[objective]["prediction"] for objective in objectives])
        reduced_crossfit, weights, blend_intercept = crossfit_affine(reduced_matrix, raw["y"], raw["fold_id"])
    else:
        objectives = ["raw"]
        weights = np.array([1.0])
        blend_intercept = 0.0
        reduced_matrix = np.column_stack([raw["prediction"]])
        reduced_crossfit = raw["prediction"]
    segments = data.loc[raw["indices"], "author_cold_start"].fillna(1).ge(0.5).to_numpy(dtype=np.int8)
    segmented_prediction, segmented_parameters = crossfit_segmented_affine(reduced_matrix, raw["y"], raw["fold_id"], segments)
    global_result = {"fold_metrics": score_by_fold(raw["y"], reduced_crossfit, raw["fold_id"], folds)}
    segmented_folds = score_by_fold(raw["y"], segmented_prediction, raw["fold_id"], folds)
    segmented_result = {"fold_metrics": segmented_folds}
    stable_segmented, segmented_gain, segmented_standard_error = stable_improvement(segmented_result, global_result)
    rounds = {}
    for objective in objectives:
        result = objective_results[objective]
        rounds[objective] = int(np.clip(np.median(result["best_rounds"]), 100, MAX_ROUNDS if objective != "catboost" else 1200))
    selection = {
        "feature_scope": "full" if chosen_name.startswith("full") else "core",
        "weight_mode": weight_mode,
        "objectives": objectives,
        "weights": weights.tolist(),
        "intercept": float(blend_intercept),
        "rounds": rounds,
        "block_comparisons": block_comparisons,
        "residual_correlations": residual_correlations,
        "blend_gain": float(blend_gain),
        "blend_standard_error": float(blend_standard_error),
        "catboost_evaluated": catboost_evaluated,
        "segmented_calibration": segmented_parameters if stable_segmented else {},
        "segmented_gain": float(segmented_gain),
        "segmented_standard_error": float(segmented_standard_error),
        "raw_fold_metrics": raw["fold_metrics"],
        "blend_fold_metrics": blend_folds,
    }
    selection_path.write_text(json.dumps(selection, indent=2))
    print(f"[selection] chosen={json.dumps(selection, sort_keys=True)}", flush=True)
    report_strata(data, raw["indices"], raw["y"], segmented_prediction if stable_segmented else reduced_crossfit, raw["fold_id"], folds)
    return selection


def report_strata(data: pd.DataFrame, indices: np.ndarray, y: np.ndarray, prediction: np.ndarray, fold_id: np.ndarray, folds: list[pd.Timestamp]) -> None:
    held = data.loc[indices].copy()
    held["_y"] = y
    held["_prediction"] = prediction
    held["_fold"] = fold_id
    strata = []
    for fold, origin in enumerate(folds):
        mask = held["_fold"].eq(fold)
        result = metrics(held.loc[mask, "_y"], held.loc[mask, "_prediction"])
        result.update({"axis": "origin", "stratum": str(pd.Timestamp(origin).date())})
        strata.append(result)
    cold = held["author_cold_start"].fillna(1).ge(0.5)
    for label, mask in (("cold", cold), ("history", ~cold)):
        if mask.sum() > 1:
            result = metrics(held.loc[mask, "_y"], held.loc[mask, "_prediction"])
            result.update({"axis": "history", "stratum": label})
            strata.append(result)
    publications = held["pub_count_lifetime"].fillna(0)
    publication_bands = pd.cut(publications, [-1, 0, 2, 5, np.inf], labels=["0", "1-2", "3-5", "6+"])
    for label in publication_bands.cat.categories:
        mask = publication_bands.eq(label)
        if mask.sum() > 1:
            result = metrics(held.loc[mask, "_y"], held.loc[mask, "_prediction"])
            result.update({"axis": "prior_publications", "stratum": str(label)})
            strata.append(result)
    team = held["team_size_max"].fillna(0)
    team_bands = pd.cut(team, [-1, 2, 10, 50, np.inf], labels=["0-2", "3-10", "11-50", "51+"])
    for label in team_bands.cat.categories:
        mask = team_bands.eq(label)
        if mask.sum() > 1:
            result = metrics(held.loc[mask, "_y"], held.loc[mask, "_prediction"])
            result.update({"axis": "max_team_size", "stratum": str(label)})
            strata.append(result)
    print(f"[internal_strata] {json.dumps(strata)}", flush=True)


def final_predictions(
    training: pd.DataFrame,
    prediction: pd.DataFrame,
    prediction_origin: pd.Timestamp,
    selection: dict,
    debug: bool,
) -> np.ndarray:
    columns = feature_columns(training.iloc[:1], selection["feature_scope"])
    missing = sorted(set(columns) - set(prediction.columns))
    if missing:
        raise RuntimeError(f"prediction features missing: {missing}")
    training_y = training["publication_count"].to_numpy(dtype=np.float64)
    weights = recency_weights(training["origin"], prediction_origin, selection["weight_mode"])
    objective_predictions = []
    seeds = (17,) if debug else SEEDS
    for objective in selection["objectives"]:
        seed_predictions = []
        rounds = min(int(selection["rounds"][objective]), 200) if debug else int(selection["rounds"][objective])
        for seed in seeds:
            if objective == "catboost":
                values, _ = fit_catboost(training[columns], training_y, prediction[columns], seed, rounds, weights)
            else:
                values, _ = fit_lgb(training[columns], training_y, prediction[columns], objective, seed, rounds, weights)
            seed_predictions.append(values)
        objective_predictions.append(np.mean(seed_predictions, axis=0))
    matrix = np.column_stack(objective_predictions)
    segmented = selection.get("segmented_calibration", {})
    if segmented:
        values = np.zeros(len(prediction), dtype=np.float64)
        cold = prediction["author_cold_start"].fillna(1).ge(0.5).to_numpy(dtype=np.int8)
        for segment in (0, 1):
            selected = cold == segment
            parameters = segmented[str(segment)]
            values[selected] = matrix[selected] @ np.asarray(parameters["weights"], dtype=np.float64) + float(parameters["intercept"])
    else:
        blend_weights = np.asarray(selection["weights"], dtype=np.float64)
        values = matrix @ blend_weights + float(selection["intercept"])
    return np.maximum(values, 1.0).astype(np.float64)
