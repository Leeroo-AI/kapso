from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Configuration

MEMBERS = ("raw_l1", "raw_quantile", "log_quantile", "rate_quantile")
SEEDS = (17, 43)


@dataclass
class Selection:
    pseudo_count: int
    use_text: bool
    use_text_neighbors: bool
    use_profiles: bool
    blend_weights: dict[str, float]
    rounds: dict[str, int]
    upper_cap: float | None
    diagnostics: dict


# Feature matrices

def select_pseudo_count(rows: pd.DataFrame, priors: pd.DataFrame) -> tuple[int, dict]:
    mask = (
        (rows["split"] == "train")
        & rows["timestamp"].dt.year.isin([2016, 2017, 2018, 2019])
    )
    y = rows.loc[mask, "target"].to_numpy(float)
    indices = rows.loc[mask, "row_idx"].to_numpy()
    results = {}
    relations = ("sponsor", "agency", "condition", "intervention", "country", "phase", "study_type")
    for pseudo_count in (5, 15, 50):
        columns = [
            f"prior_{relation}__k{pseudo_count}__median_y_mean"
            for relation in relations
            if f"prior_{relation}__k{pseudo_count}__median_y_mean" in priors
        ]
        values = priors.loc[indices, columns].to_numpy(float)
        prediction = np.nanmedian(values, axis=1)
        fallback = priors.loc[indices, "prior_global_median_y"].to_numpy(float)
        prediction = np.where(np.isfinite(prediction), prediction, fallback)
        results[str(pseudo_count)] = float(mean_absolute_error(y, prediction))
    selected = int(min(results, key=results.get))
    return selected, results


def structured_frame(
    local: pd.DataFrame,
    priors: pd.DataFrame,
    pseudo_count: int,
    use_profiles: bool,
) -> pd.DataFrame:
    keep = []
    token = f"__k{pseudo_count}__"
    for column in priors.columns:
        if "__k" in column and token not in column:
            continue
        if not use_profiles and "_profile_" in column:
            continue
        keep.append(column)
    frame = pd.concat([local, priors[keep]], axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    return frame.replace([np.inf, -np.inf], np.nan).astype(np.float32)


def make_matrix(
    structured: pd.DataFrame,
    text: np.ndarray | None,
    text_neighbors: np.ndarray | None = None,
) -> np.ndarray:
    values = structured.to_numpy(dtype=np.float32, copy=False)
    if text is not None:
        values = np.hstack([values, np.asarray(text, dtype=np.float32)])
    if text_neighbors is not None:
        values = np.hstack(
            [values, np.asarray(text_neighbors, dtype=np.float32)]
        )
    return values


# LightGBM

def _parameters(member: str, seed: int) -> dict:
    parameters = {
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_data_in_leaf": 64,
        "feature_fraction": 0.8,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "seed": seed,
        "feature_fraction_seed": seed,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "force_col_wise": True,
        "deterministic": True,
    }
    if member == "raw_l1":
        parameters.update(objective="regression_l1", metric="l1")
    else:
        parameters.update(objective="quantile", metric="quantile", alpha=0.5)
    return parameters


def _target(member: str, y: np.ndarray, enrollment: np.ndarray) -> np.ndarray:
    if member == "log_quantile":
        return np.log1p(np.maximum(y, 0))
    if member == "rate_quantile":
        return np.log((np.maximum(y, 0) + 1.0) / (np.maximum(enrollment, 0) + 1.0))
    return y


def _inverse(member: str, prediction: np.ndarray, enrollment: np.ndarray) -> np.ndarray:
    if member == "log_quantile":
        return np.expm1(prediction)
    if member == "rate_quantile":
        return np.exp(prediction) * (np.maximum(enrollment, 0) + 1.0) - 1.0
    return prediction


def fit_fold(
    member: str,
    matrix: np.ndarray,
    y: np.ndarray,
    enrollment: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    debug: bool,
    seed: int = 17,
) -> tuple[np.ndarray, int]:
    transformed = _target(member, y, enrollment)
    train_set = lgb.Dataset(
        matrix[train_indices],
        label=transformed[train_indices],
        free_raw_data=True,
    )
    valid_set = lgb.Dataset(
        matrix[valid_indices],
        label=transformed[valid_indices],
        reference=train_set,
        free_raw_data=True,
    )
    rounds = 100 if debug else 2000
    callbacks = [lgb.log_evaluation(0)]
    if not debug:
        callbacks.append(lgb.early_stopping(120, verbose=False))
    model = lgb.train(
        _parameters(member, seed),
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=callbacks,
    )
    best = model.best_iteration if model.best_iteration else rounds
    prediction = model.predict(matrix[valid_indices], num_iteration=best)
    return np.maximum(
        _inverse(member, prediction, enrollment[valid_indices]), 0
    ), int(best)


def fit_final_member(
    member: str,
    matrix: np.ndarray,
    y: np.ndarray,
    enrollment: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    rounds: int,
    debug: bool,
) -> np.ndarray:
    if debug and len(train_indices) > 5000:
        train_indices = train_indices[-5000:]
    predictions = []
    seeds = (17,) if debug else SEEDS
    transformed = _target(member, y, enrollment)
    for seed in seeds:
        train_set = lgb.Dataset(
            matrix[train_indices],
            label=transformed[train_indices],
            free_raw_data=True,
        )
        model = lgb.train(
            _parameters(member, seed),
            train_set,
            num_boost_round=100 if debug else min(2000, max(100, rounds)),
            callbacks=[lgb.log_evaluation(0)],
        )
        raw = model.predict(matrix[predict_indices])
        predictions.append(
            np.maximum(_inverse(member, raw, enrollment[predict_indices]), 0)
        )
    return np.mean(predictions, axis=0)


# Rolling selection

def _rolling_folds(rows: pd.DataFrame, debug: bool):
    years = [2019] if debug else [2016, 2017, 2018, 2019]
    folds = []
    for year in years:
        valid_mask = (rows["split"] == "train") & (rows["timestamp"].dt.year == year)
        if not valid_mask.any():
            continue
        cutoff = rows.loc[valid_mask, "timestamp"].iloc[0]
        train_mask = (rows["split"] == "train") & (rows["available_time"] <= cutoff)
        train_indices = np.flatnonzero(train_mask.to_numpy())
        if debug and len(train_indices) > 5000:
            train_indices = train_indices[-5000:]
        valid_indices = np.flatnonzero(valid_mask.to_numpy())
        folds.append((year, train_indices, valid_indices))
    return folds


def _metric_bundle(y: np.ndarray, prediction: np.ndarray) -> dict:
    correlation = spearmanr(y, prediction).statistic
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "r2": float(r2_score(y, prediction)),
        "spearman": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def _blend_grid(predictions: dict[str, np.ndarray], y: np.ndarray) -> tuple[dict, float]:
    names = list(predictions)
    matrix = np.column_stack([predictions[name] for name in names])
    best_score = np.inf
    best_weights = None
    grid = range(41)
    for first in grid:
        for second in range(41 - first):
            for third in range(41 - first - second):
                fourth = 40 - first - second - third
                weights = np.array([first, second, third, fourth], dtype=float) / 40.0
                prediction = matrix @ weights
                score = np.mean(np.abs(y - prediction))
                if score < best_score:
                    best_score = float(score)
                    best_weights = weights
    return {
        name: float(weight) for name, weight in zip(names, best_weights)
    }, best_score


def _upper_cap_ablation(y: np.ndarray, prediction: np.ndarray) -> tuple[float | None, dict]:
    candidates = [None, 250.0, 500.0, 1000.0, 2000.0]
    scores = {}
    for candidate in candidates:
        current = prediction if candidate is None else np.minimum(prediction, candidate)
        scores["none" if candidate is None else str(int(candidate))] = float(
            mean_absolute_error(y, current)
        )
    selected_name = min(scores, key=scores.get)
    selected = None if selected_name == "none" else float(selected_name)
    return selected, scores


def rolling_selection(
    rows: pd.DataFrame,
    local: pd.DataFrame,
    priors: pd.DataFrame,
    text_oof: np.ndarray,
    text_neighbors_oof: np.ndarray,
    debug: bool,
    phase_logger,
) -> Selection:
    pseudo_count, pseudo_scores = select_pseudo_count(rows, priors)
    y = rows["target"].fillna(0).to_numpy(float)
    enrollment = local["study_enrollment"].fillna(0).to_numpy(float)
    folds = _rolling_folds(rows, debug)
    variants = {
        "structured": make_matrix(
            structured_frame(local, priors, pseudo_count, False), None
        ),
        "profiles": make_matrix(
            structured_frame(local, priors, pseudo_count, True), None
        ),
        "profiles_text": make_matrix(
            structured_frame(local, priors, pseudo_count, True), text_oof
        ),
        "text_neighbors": make_matrix(
            structured_frame(local, priors, pseudo_count, False),
            None,
            text_neighbors_oof,
        ),
        "profiles_neighbors": make_matrix(
            structured_frame(local, priors, pseudo_count, True),
            None,
            text_neighbors_oof,
        ),
        "profiles_text_neighbors": make_matrix(
            structured_frame(local, priors, pseudo_count, True),
            text_oof,
            text_neighbors_oof,
        ),
    }
    ablation_predictions = {
        name: np.full(len(rows), np.nan, dtype=float) for name in variants
    }
    ablation_rounds = {name: [] for name in variants}
    for year, train_indices, valid_indices in folds:
        for name, matrix in variants.items():
            prediction, best = fit_fold(
                "raw_l1",
                matrix,
                y,
                enrollment,
                train_indices,
                valid_indices,
                debug,
            )
            ablation_predictions[name][valid_indices] = prediction
            ablation_rounds[name].append(best)
        phase_logger(
            "rolling_ablation_fold",
            year=year,
            train_rows=len(train_indices),
            valid_rows=len(valid_indices),
        )
    oof_indices = np.concatenate([valid for _, _, valid in folds])
    ablation_scores = {
        name: float(mean_absolute_error(y[oof_indices], prediction[oof_indices]))
        for name, prediction in ablation_predictions.items()
    }
    selected_variant = min(ablation_scores, key=ablation_scores.get)
    use_profiles = selected_variant in {
        "profiles",
        "profiles_text",
        "profiles_neighbors",
        "profiles_text_neighbors",
    }
    use_text = selected_variant in {
        "profiles_text",
        "profiles_text_neighbors",
    }
    use_text_neighbors = selected_variant in {
        "text_neighbors",
        "profiles_neighbors",
        "profiles_text_neighbors",
    }
    selected_structured = structured_frame(
        local, priors, pseudo_count, use_profiles
    )
    selected_matrix = make_matrix(
        selected_structured,
        text_oof if use_text else None,
        text_neighbors_oof if use_text_neighbors else None,
    )
    predictions = {
        member: np.full(len(rows), np.nan, dtype=float) for member in MEMBERS
    }
    predictions["raw_l1"][:] = ablation_predictions[selected_variant]
    best_rounds = {
        "raw_l1": list(ablation_rounds[selected_variant]),
        "raw_quantile": [],
        "log_quantile": [],
        "rate_quantile": [],
    }
    for year, train_indices, valid_indices in folds:
        for member in MEMBERS[1:]:
            prediction, best = fit_fold(
                member,
                selected_matrix,
                y,
                enrollment,
                train_indices,
                valid_indices,
                debug,
            )
            predictions[member][valid_indices] = prediction
            best_rounds[member].append(best)
        phase_logger("rolling_member_fold", year=year)
    compact_predictions = {
        member: values[oof_indices] for member, values in predictions.items()
    }
    weights, blend_mae = _blend_grid(compact_predictions, y[oof_indices])
    blend = sum(
        weights[member] * compact_predictions[member] for member in MEMBERS
    )
    upper_cap, cap_scores = _upper_cap_ablation(y[oof_indices], blend)
    if upper_cap is not None:
        blend = np.minimum(blend, upper_cap)
    rounds = {}
    for member, values in best_rounds.items():
        median = np.median(values) if values else 100
        rounds[member] = int(np.clip(round(median * 1.1), 100, 2000))
    diagnostics = {
        "pseudo_count_mae": pseudo_scores,
        "ablation_mae": ablation_scores,
        "member_metrics": {
            member: _metric_bundle(y[oof_indices], prediction)
            for member, prediction in compact_predictions.items()
        },
        "blend_metrics": _metric_bundle(y[oof_indices], blend),
        "blend_mae_before_cap": blend_mae,
        "upper_cap_mae": cap_scores,
        "fold_rows": {
            str(year): {
                "train": int(len(train_indices)),
                "valid": int(len(valid_indices)),
            }
            for year, train_indices, valid_indices in folds
        },
        "oof_indices": oof_indices.tolist(),
        "oof_prediction": blend.tolist(),
    }
    return Selection(
        pseudo_count=pseudo_count,
        use_text=use_text,
        use_text_neighbors=use_text_neighbors,
        use_profiles=use_profiles,
        blend_weights=weights,
        rounds=rounds,
        upper_cap=upper_cap,
        diagnostics=diagnostics,
    )


# Diagnostics

def stratified_diagnostics(
    rows: pd.DataFrame,
    local: pd.DataFrame,
    priors: pd.DataFrame,
    selection: Selection,
) -> dict:
    indices = np.asarray(selection.diagnostics["oof_indices"], dtype=int)
    prediction = np.asarray(selection.diagnostics["oof_prediction"], dtype=float)
    y = rows.loc[indices, "target"].to_numpy(float)
    enrollment = local.loc[indices, "study_enrollment"].fillna(0).to_numpy(float)
    frame = pd.DataFrame(
        {
            "y": y,
            "prediction": prediction,
            "enrollment": enrollment,
            "support": priors.loc[
                indices, "prior_sponsor_support_sum"
            ].fillna(0).to_numpy(float),
            "sponsor_seen": priors.loc[
                indices, "prior_sponsor_known_fraction"
            ].fillna(0).to_numpy(float),
            "condition_seen": priors.loc[
                indices, "prior_condition_known_fraction"
            ].fillna(0).to_numpy(float),
            "dispersion": priors.loc[
                indices,
                f"prior_condition__k{selection.pseudo_count}__median_y_dispersion",
            ].fillna(0).to_numpy(float),
        }
    )
    frame["absolute_error"] = np.abs(frame["y"] - frame["prediction"])
    frame["support_bucket"] = pd.cut(
        frame["support"],
        bins=[-1, 0, 5, 25, 100, np.inf],
        labels=["unseen", "1-5", "6-25", "26-100", "100+"],
    )
    frame["enrollment_bucket"] = pd.cut(
        frame["enrollment"],
        bins=[-1, 50, 200, 1000, np.inf],
        labels=["0-50", "51-200", "201-1000", "1000+"],
    )
    frame["sponsor_visibility"] = np.where(
        frame["sponsor_seen"] > 0, "seen", "unseen"
    )
    frame["condition_visibility"] = np.where(
        frame["condition_seen"] > 0, "seen", "unseen"
    )
    nonzero_dispersion = frame["dispersion"].replace(0, np.nan)
    try:
        frame["dispersion_bucket"] = pd.qcut(
            nonzero_dispersion,
            q=4,
            labels=["q1", "q2", "q3", "q4"],
            duplicates="drop",
        ).astype(object)
    except ValueError:
        frame["dispersion_bucket"] = "single"
    frame["dispersion_bucket"] = frame["dispersion_bucket"].fillna("none")
    rate = np.log((frame["y"] + 1.0) / (frame["enrollment"] + 1.0))
    center = priors.loc[
        indices,
        f"prior_condition__k{selection.pseudo_count}__median_log_rate_mean",
    ].fillna(
        priors.loc[indices, "prior_global_median_log_rate"]
    ).to_numpy(float)
    spread = priors.loc[
        indices,
        f"prior_condition__k{selection.pseudo_count}__std_log_y_mean",
    ].fillna(1.0).clip(lower=0.25).to_numpy(float)
    frame["regime"] = np.where(
        (rate < center - 1.5 * spread) | (rate > center + 1.5 * spread),
        "shifted",
        "predictable",
    )
    output = {}
    for axis in (
        "support_bucket",
        "sponsor_visibility",
        "condition_visibility",
        "enrollment_bucket",
        "dispersion_bucket",
        "regime",
    ):
        output[axis] = {}
        for value, group in frame.groupby(axis, observed=True):
            output[axis][str(value)] = {
                "count": int(len(group)),
                "mae": float(group["absolute_error"].mean()),
            }
    output["overall"] = _metric_bundle(y, prediction)
    return output


# Final refits

def final_predictions(
    rows: pd.DataFrame,
    local: pd.DataFrame,
    priors: pd.DataFrame,
    text_a: np.ndarray,
    text_b: np.ndarray,
    text_neighbors_a: np.ndarray,
    text_neighbors_b: np.ndarray,
    selection: Selection,
    debug: bool,
    phase_logger,
) -> tuple[np.ndarray, np.ndarray]:
    structured = structured_frame(
        local, priors, selection.pseudo_count, selection.use_profiles
    )
    matrix_a = make_matrix(
        structured,
        text_a if selection.use_text else None,
        text_neighbors_a if selection.use_text_neighbors else None,
    )
    matrix_b = make_matrix(
        structured,
        text_b if selection.use_text else None,
        text_neighbors_b if selection.use_text_neighbors else None,
    )
    y = rows["target"].fillna(0).to_numpy(float)
    enrollment = local["study_enrollment"].fillna(0).to_numpy(float)
    train_a = np.flatnonzero((rows["split"] == "train").to_numpy())
    train_b = np.flatnonzero(rows["split"].isin(["train", "val"]).to_numpy())
    val_indices = np.flatnonzero((rows["split"] == "val").to_numpy())
    test_indices = np.flatnonzero((rows["split"] == "test").to_numpy())
    val_members = {}
    test_members = {}
    for member in MEMBERS:
        val_members[member] = fit_final_member(
            member,
            matrix_a,
            y,
            enrollment,
            train_a,
            val_indices,
            selection.rounds[member],
            debug,
        )
        test_members[member] = fit_final_member(
            member,
            matrix_b,
            y,
            enrollment,
            train_b,
            test_indices,
            selection.rounds[member],
            debug,
        )
        phase_logger(
            "final_member",
            member=member,
            rounds=selection.rounds[member],
        )
    val_prediction = sum(
        selection.blend_weights[member] * val_members[member] for member in MEMBERS
    )
    test_prediction = sum(
        selection.blend_weights[member] * test_members[member] for member in MEMBERS
    )
    val_prediction = np.maximum(val_prediction, 0)
    test_prediction = np.maximum(test_prediction, 0)
    if selection.upper_cap is not None:
        val_prediction = np.minimum(val_prediction, selection.upper_cap)
        test_prediction = np.minimum(test_prediction, selection.upper_cap)
    return val_prediction.astype(np.float64), test_prediction.astype(np.float64)
