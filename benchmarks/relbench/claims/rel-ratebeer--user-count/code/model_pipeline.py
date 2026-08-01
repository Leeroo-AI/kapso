from __future__ import annotations

import gc
import math
import os
import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DAY_MS = 86_400_000


@dataclass
class Structure:
    leaves: int
    half_life: float | None
    use_wide: bool
    feature_indices: np.ndarray
    tweedie_power: float
    trees: dict[str, int]


@dataclass
class Blend:
    component_names: list[str]
    coefficients: np.ndarray
    intercept: float

    def predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        matrix = np.column_stack([predictions[name] for name in self.component_names])
        return matrix @ self.coefficients + self.intercept


def _params(objective: str, leaves: int, seed: int, tweedie_power: float = 1.5) -> dict:
    parameters = {
        "objective": objective,
        "metric": "l2",
        "learning_rate": 0.035,
        "num_leaves": leaves,
        "min_data_in_leaf": 150,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 20.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "11")),
        "seed": seed,
        "feature_fraction_seed": seed + 1,
        "bagging_seed": seed + 2,
        "data_random_seed": seed + 3,
        "force_col_wise": True,
    }
    if objective == "tweedie":
        parameters["tweedie_variance_power"] = tweedie_power
    return parameters


def _weights(timestamps: np.ndarray, indices: np.ndarray, cutoff_ms: int, half_life: float | None) -> np.ndarray | None:
    if half_life is None:
        return None
    ages = np.maximum(cutoff_ms - (timestamps[indices] + 90 * DAY_MS), 0) / DAY_MS
    values = np.exp2(-ages / half_life).astype(np.float32)
    values /= max(float(values.mean()), 1e-8)
    return values


def _fit_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    cutoff_ms: int,
    leaves: int,
    half_life: float | None,
    component: str,
    max_trees: int,
    seed: int,
    tweedie_power: float = 1.5,
    early_stopping: bool = True,
) -> tuple[lgb.Booster, np.ndarray, int]:
    objective = component
    train_target = target[train_indices]
    if component == "log":
        objective = "regression"
        train_target = np.log1p(train_target)
    train_data = lgb.Dataset(
        matrix[train_indices],
        label=train_target,
        weight=_weights(timestamps, train_indices, cutoff_ms, half_life),
        free_raw_data=True,
    )
    validation_data = None
    callbacks = [lgb.log_evaluation(0)]
    if len(validation_indices):
        validation_target = target[validation_indices]
        if component == "log":
            validation_target = np.log1p(validation_target)
        validation_data = lgb.Dataset(matrix[validation_indices], label=validation_target, reference=train_data, free_raw_data=True)
        if early_stopping:
            callbacks.insert(0, lgb.early_stopping(100, verbose=False))
    started = time.time()
    booster = lgb.train(
        _params(objective, leaves, seed, tweedie_power),
        train_data,
        num_boost_round=max_trees,
        valid_sets=[validation_data] if validation_data is not None else None,
        callbacks=callbacks,
    )
    best_iteration = booster.best_iteration if booster.best_iteration else max_trees
    prediction = booster.predict(matrix[validation_indices], num_iteration=best_iteration) if len(validation_indices) else np.empty(0)
    if component == "log":
        prediction = np.expm1(prediction)
    prediction = np.maximum(prediction, 0.0)
    print(f"[model] {component} leaves={leaves} half={half_life} train={len(train_indices)} valid={len(validation_indices)} trees={best_iteration} seconds={time.time() - started:.1f}")
    return booster, prediction.astype(np.float64), int(best_iteration)


def _fold_indices(timestamps: np.ndarray, allowed: np.ndarray, forecast_ms: int) -> tuple[np.ndarray, np.ndarray]:
    training = np.flatnonzero(allowed & (timestamps + 90 * DAY_MS <= forecast_ms))
    validation = np.flatnonzero(allowed & (timestamps == forecast_ms))
    return training, validation


def _fold_score(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(r2_score(target, prediction))


def _bootstrap_se(target: np.ndarray, prediction: np.ndarray, seed: int, repetitions: int = 64) -> float:
    if len(target) < 2:
        return 0.0
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, len(target), len(target))
        values[index] = r2_score(target[sample], prediction[sample])
    return float(np.std(values, ddof=1))


def _candidate_oof(
    matrix: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    allowed: np.ndarray,
    folds: list[int],
    leaves: int,
    half_life: float | None,
    component: str,
    max_trees: int,
    seed: int,
    tweedie_power: float = 1.5,
    early_stopping: bool = True,
) -> dict:
    predictions = []
    targets = []
    rows = []
    fold_ids = []
    scores = []
    bootstrap = []
    trees = []
    for fold_number, forecast_ms in enumerate(folds):
        train_indices, validation_indices = _fold_indices(timestamps, allowed, forecast_ms)
        if not len(train_indices) or not len(validation_indices):
            continue
        booster, prediction, best_iteration = _fit_predict(
            matrix,
            target,
            timestamps,
            train_indices,
            validation_indices,
            forecast_ms,
            leaves,
            half_life,
            component,
            max_trees,
            seed + fold_number * 17,
            tweedie_power,
            early_stopping,
        )
        score = _fold_score(target[validation_indices], prediction)
        scores.append(score)
        bootstrap.append(_bootstrap_se(target[validation_indices], prediction, seed + fold_number))
        trees.append(best_iteration)
        predictions.append(prediction)
        targets.append(target[validation_indices])
        rows.append(validation_indices)
        fold_ids.append(np.full(len(validation_indices), fold_number, dtype=np.int8))
        del booster
        gc.collect()
    if not predictions:
        raise RuntimeError(f"no usable forward folds for component={component}")
    print(f"[folds] {component} mean_r2={np.mean(scores):.6f} std={np.std(scores):.6f} bootstrap_se_mean={np.mean(bootstrap):.6f} per_origin={','.join(f'{value:.6f}' for value in scores)}")
    return {
        "prediction": np.concatenate(predictions),
        "target": np.concatenate(targets),
        "rows": np.concatenate(rows),
        "fold": np.concatenate(fold_ids),
        "scores": np.asarray(scores),
        "bootstrap": np.asarray(bootstrap),
        "trees": np.asarray(trees),
    }


def _stable_choice(results: dict[tuple, dict]) -> tuple:
    utilities = {}
    for key, result in results.items():
        scores = result["scores"]
        utilities[key] = float(np.mean(scores) - 0.25 * np.std(scores))
    best_utility = max(utilities.values())
    eligible = [key for key, utility in utilities.items() if utility >= best_utility - 0.002]
    eligible.sort(key=lambda key: (key[0], key[1] is not None, -(key[1] or 1e12)))
    choice = eligible[0]
    print(f"[selection] stable choice={choice} utility={utilities[choice]:.6f} mean={np.mean(results[choice]['scores']):.6f}")
    return choice


def _fit_blend(component_oof: dict[str, dict], folds_for_fit: np.ndarray | None = None) -> Blend:
    names = list(component_oof)
    base = component_oof[names[0]]
    matrix = np.column_stack([component_oof[name]["prediction"] for name in names])
    target = base["target"]
    if folds_for_fit is not None:
        mask = np.isin(base["fold"], folds_for_fit)
        matrix = matrix[mask]
        target = target[mask]
    ridge = Ridge(alpha=20.0, positive=True, fit_intercept=True)
    ridge.fit(matrix, target)
    coefficients = np.asarray(ridge.coef_, dtype=np.float64)
    print(f"[blend] names={names} coefficients={coefficients.tolist()} intercept={float(ridge.intercept_):.6f}")
    return Blend(names, coefficients, float(ridge.intercept_))


def _blend_oof(blend: Blend, component_oof: dict[str, dict]) -> np.ndarray:
    return blend.predict({name: component_oof[name]["prediction"] for name in blend.component_names})


def select_and_build_a(
    matrix: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    kinds: np.ndarray,
    names: list[str],
    groups: list[str],
    validation_indices: np.ndarray,
    debug: bool,
) -> tuple[Structure, Blend, np.ndarray, dict]:
    cutoff_v = np.datetime64("2018-09-01", "ms").astype(np.int64)
    allowed = (kinds != 2) & (timestamps + 90 * DAY_MS <= cutoff_v)
    addition_groups = ("site_drift_addition", "user_trajectory_addition", "cadence_distribution_addition", "weekly_trajectory_addition")
    base_core_indices = np.asarray([index for index, group in enumerate(groups) if not group.startswith("wide_") and group not in addition_groups], dtype=np.int32)
    if debug:
        folds = [np.datetime64("2018-06-01", "ms").astype(np.int64)]
        candidates = [(63, 1096.0)]
        max_trees = 100
    else:
        folds = [
            np.datetime64("2017-09-01", "ms").astype(np.int64),
            np.datetime64("2018-03-01", "ms").astype(np.int64),
            np.datetime64("2018-06-01", "ms").astype(np.int64),
        ]
        candidates = [(63, 1096.0)]
        max_trees = 2000
    core_matrix = np.ascontiguousarray(matrix[:, base_core_indices])
    raw_results = {}
    for leaves, half_life in candidates:
        raw_results[(leaves, half_life)] = _candidate_oof(core_matrix, target, timestamps, allowed, folds, leaves, half_life, "regression", max_trees, 1337)
    chosen = _stable_choice(raw_results)
    leaves, half_life = chosen
    use_wide = False
    selected_matrix = core_matrix
    raw_oof = raw_results[chosen]
    selected_indices = base_core_indices
    for addition_group in addition_groups:
        group_indices = [index for index, group in enumerate(groups) if group == addition_group]
        if not group_indices:
            continue
        extended_indices = np.asarray(list(selected_indices) + group_indices, dtype=np.int32)
        extended_matrix = np.ascontiguousarray(matrix[:, extended_indices])
        extended_oof = _candidate_oof(extended_matrix, target, timestamps, allowed, folds, leaves, half_life, "regression", max_trees, 1337)
        difference = extended_oof["scores"] - raw_oof["scores"]
        uncertainty = float(np.std(difference) / math.sqrt(max(len(difference), 1)))
        retained = float(np.mean(difference)) > max(0.0, uncertainty)
        if retained:
            selected_matrix = extended_matrix
            selected_indices = extended_indices
            raw_oof = extended_oof
        else:
            del extended_matrix
        print(f"[selection] {addition_group}_delta_mean={np.mean(difference):.6f} uncertainty={uncertainty:.6f} retained={retained}")
    removal_specs = {
        "favorite_aux": lambda index: names[index].startswith("favorite_") or names[index].startswith("availability_"),
        "place_aux": lambda index: names[index].startswith("place_rating_"),
    }
    for removal_name, predicate in removal_specs.items():
        reduced_indices = np.asarray([index for index in selected_indices if not predicate(int(index))], dtype=np.int32)
        reduced_matrix = np.ascontiguousarray(matrix[:, reduced_indices])
        reduced_oof = _candidate_oof(reduced_matrix, target, timestamps, allowed, folds, leaves, half_life, "regression", max_trees, 1337)
        difference = reduced_oof["scores"] - raw_oof["scores"]
        uncertainty = float(np.std(difference) / math.sqrt(max(len(difference), 1)))
        remove = float(np.mean(difference)) >= -uncertainty
        if remove:
            selected_matrix = reduced_matrix
            selected_indices = reduced_indices
            raw_oof = reduced_oof
        else:
            del reduced_matrix
        print(f"[selection] remove_{removal_name}_delta_mean={np.mean(difference):.6f} uncertainty={uncertainty:.6f} removed={remove}")
    wide_indices = np.asarray(list(selected_indices) + [index for index, group in enumerate(groups) if group.startswith("wide_")], dtype=np.int32)
    if not debug and len(wide_indices) > len(selected_indices):
        full_matrix = np.ascontiguousarray(matrix[:, wide_indices])
        full_oof = _candidate_oof(full_matrix, target, timestamps, allowed, folds, leaves, half_life, "regression", max_trees, 1337)
        difference = full_oof["scores"] - raw_oof["scores"]
        uncertainty = float(np.std(difference) / math.sqrt(max(len(difference), 1)))
        if float(np.mean(difference)) > max(0.0, uncertainty):
            use_wide = True
            selected_matrix = full_matrix
            selected_indices = wide_indices
            raw_oof = full_oof
        else:
            del full_matrix
        print(f"[selection] wide_delta_mean={np.mean(difference):.6f} uncertainty={uncertainty:.6f} retained={use_wide}")
    feature_indices = selected_indices
    component_oof = {"regression": raw_oof}
    for component in ("poisson",):
        component_oof[component] = _candidate_oof(selected_matrix, target, timestamps, allowed, folds, leaves, half_life, component, max_trees, 3337)
    tweedie_results = {}
    powers = (1.5,) if debug else (1.3, 1.5)
    for power in powers:
        tweedie_results[power] = _candidate_oof(selected_matrix, target, timestamps, allowed, folds, leaves, half_life, "tweedie", max_trees, 4337 + int(power * 100), power)
    tweedie_power = max(powers, key=lambda power: float(np.mean(tweedie_results[power]["scores"]) - 0.25 * np.std(tweedie_results[power]["scores"])))
    component_oof["tweedie"] = tweedie_results[tweedie_power]
    log_oof = _candidate_oof(selected_matrix, target, timestamps, allowed, folds, leaves, half_life, "log", max_trees, 5337)
    component_oof["log"] = log_oof
    fold_values = np.unique(raw_oof["fold"])
    meta_passed_outer = True
    if len(fold_values) > 1:
        outer_mask = raw_oof["fold"] == fold_values[-1]
        raw_outer = r2_score(raw_oof["target"][outer_mask], raw_oof["prediction"][outer_mask])
        tolerance = max(0.002, float(raw_oof["bootstrap"][-1]))
        outer_blend = _fit_blend(component_oof, fold_values[:-1])
        outer_predictions = outer_blend.predict({name: component_oof[name]["prediction"][outer_mask] for name in outer_blend.component_names})
        outer_score = r2_score(raw_oof["target"][outer_mask], outer_predictions)
        without_log = {name: values for name, values in component_oof.items() if name != "log"}
        outer_without_log = _fit_blend(without_log, fold_values[:-1])
        without_log_prediction = outer_without_log.predict({name: without_log[name]["prediction"][outer_mask] for name in outer_without_log.component_names})
        without_log_score = r2_score(raw_oof["target"][outer_mask], without_log_prediction)
        if outer_score <= without_log_score + tolerance:
            component_oof.pop("log")
            outer_score = without_log_score
            print(f"[blend] log component excluded by later-outer uncertainty gate tolerance={tolerance:.6f}")
        meta_passed_outer = outer_score > raw_outer + tolerance
        print(f"[blend] later_outer_r2={outer_score:.6f} raw_r2={raw_outer:.6f} tolerance={tolerance:.6f} meta_retained={meta_passed_outer}")
    preliminary = _fit_blend(component_oof)
    if "log" in preliminary.component_names and preliminary.coefficients[preliminary.component_names.index("log")] <= 0.01:
        component_oof.pop("log")
        print("[blend] log component excluded by stable OOF weight gate")
    blend = _fit_blend(component_oof)
    if not meta_passed_outer:
        coefficients = np.zeros(len(blend.component_names), dtype=np.float64)
        coefficients[blend.component_names.index("regression")] = 1.0
        blend = Blend(blend.component_names, coefficients, 0.0)
        print("[blend] conservative raw-L2 outer-fold tie break applied for model A")
    oof_prediction = np.maximum(_blend_oof(blend, component_oof), 0.0)
    diagnostics = diagnostics_by_stratum(raw_oof["target"], oof_prediction, raw_oof["fold"])
    trees = {name: int(np.median(component_oof[name]["trees"])) for name in component_oof}
    structure = Structure(leaves, half_life, use_wide, feature_indices, float(tweedie_power), trees)
    final_predictions = fit_final_predictions(
        selected_matrix,
        target,
        timestamps,
        np.flatnonzero(allowed),
        matrix[validation_indices][:, feature_indices],
        cutoff_v,
        structure,
        blend.component_names,
        6337,
    )
    validation_prediction = np.maximum(blend.predict(final_predictions), 0.0)
    return structure, blend, validation_prediction, diagnostics


def fit_final_predictions(
    training_matrix: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    train_indices: np.ndarray,
    prediction_matrix: np.ndarray,
    cutoff_ms: int,
    structure: Structure,
    component_names: list[str],
    seed: int,
) -> dict[str, np.ndarray]:
    result = {}
    for component_number, component in enumerate(component_names):
        objective = component
        training_target = target[train_indices]
        if component == "log":
            objective = "regression"
            training_target = np.log1p(training_target)
        train_data = lgb.Dataset(
            training_matrix[train_indices],
            label=training_target,
            weight=_weights(timestamps, train_indices, cutoff_ms, structure.half_life),
            free_raw_data=True,
        )
        started = time.time()
        booster = lgb.train(
            _params(objective, structure.leaves, seed + component_number * 101, structure.tweedie_power),
            train_data,
            num_boost_round=structure.trees[component],
            callbacks=[lgb.log_evaluation(0)],
        )
        prediction = booster.predict(prediction_matrix, num_iteration=structure.trees[component])
        if component == "log":
            prediction = np.expm1(prediction)
        result[component] = np.maximum(prediction, 0.0)
        print(f"[final] {component} train={len(train_indices)} trees={structure.trees[component]} seconds={time.time() - started:.1f}")
        del booster
        gc.collect()
    return result


def build_b_blend_and_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    timestamps: np.ndarray,
    structure: Structure,
    blend_a: Blend,
    test_matrix: np.ndarray,
    debug: bool,
) -> tuple[Blend, np.ndarray, dict]:
    cutoff_t = np.datetime64("2020-01-01", "ms").astype(np.int64)
    allowed = timestamps + 90 * DAY_MS <= cutoff_t
    selected_matrix = np.ascontiguousarray(matrix[:, structure.feature_indices])
    component_names = blend_a.component_names
    if debug:
        blend_b = blend_a
        diagnostics = {}
    else:
        folds = [
            np.datetime64("2018-09-01", "ms").astype(np.int64),
            np.datetime64("2019-04-01", "ms").astype(np.int64),
            np.datetime64("2019-10-01", "ms").astype(np.int64),
        ]
        component_oof = {}
        for component_number, component in enumerate(component_names):
            component_oof[component] = _candidate_oof(
                selected_matrix,
                target,
                timestamps,
                allowed,
                folds,
                structure.leaves,
                structure.half_life,
                component,
                structure.trees[component],
                7337 + component_number * 101,
                structure.tweedie_power,
                False,
            )
        blend_b = _fit_blend(component_oof)
        oof_prediction = np.maximum(_blend_oof(blend_b, component_oof), 0.0)
        diagnostics = diagnostics_by_stratum(component_oof[component_names[0]]["target"], oof_prediction, component_oof[component_names[0]]["fold"])
    component_predictions = fit_final_predictions(
        selected_matrix,
        target,
        timestamps,
        np.flatnonzero(allowed),
        np.ascontiguousarray(test_matrix[:, structure.feature_indices]),
        cutoff_t,
        structure,
        component_names,
        8337,
    )
    prediction = np.maximum(blend_b.predict(component_predictions), 0.0)
    return blend_b, prediction, diagnostics


def diagnostics_by_stratum(target: np.ndarray, prediction: np.ndarray, fold: np.ndarray) -> dict:
    output = {"origins": {}, "target_strata": {}}
    for value in np.unique(fold):
        mask = fold == value
        output["origins"][str(int(value))] = {
            "count": int(mask.sum()),
            "r2": float(r2_score(target[mask], prediction[mask])),
            "mae": float(mean_absolute_error(target[mask], prediction[mask])),
            "rmse": float(math.sqrt(mean_squared_error(target[mask], prediction[mask]))),
        }
    strata = [
        ("zero", target == 0),
        ("1-2", (target >= 1) & (target <= 2)),
        ("3-15", (target >= 3) & (target <= 15)),
        ("16-70", (target >= 16) & (target <= 70)),
        ("71-332", (target >= 71) & (target <= 332)),
        ("333+", target >= 333),
    ]
    for name, mask in strata:
        if not mask.any():
            continue
        variance = float(np.var(target[mask]))
        output["target_strata"][name] = {
            "count": int(mask.sum()),
            "r2": float(r2_score(target[mask], prediction[mask])) if variance > 0 else None,
            "mae": float(mean_absolute_error(target[mask], prediction[mask])),
            "rmse": float(math.sqrt(mean_squared_error(target[mask], prediction[mask]))),
        }
    return output
