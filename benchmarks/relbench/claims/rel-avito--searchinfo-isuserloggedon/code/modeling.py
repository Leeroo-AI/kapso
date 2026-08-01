from __future__ import annotations

import time
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), -30, 30)
    return 1 / (1 + np.exp(-values))


def safe_auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(labels) == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, predictions))


def feature_sets(base_columns: list[str], label_columns: list[str]) -> tuple[list[str], list[str]]:
    repeat_columns = base_columns + label_columns
    cold_label_columns = [
        column
        for column in label_columns
        if not column.startswith("user_") and column != "has_user_history"
    ]
    return repeat_columns, base_columns + cold_label_columns


def assemble_matrix(
    rows: pd.DataFrame,
    label_features: pd.DataFrame,
    indices: np.ndarray,
    base_columns: list[str],
    selected_columns: list[str],
) -> pd.DataFrame:
    index = pd.Index(indices)
    base_selected = [column for column in selected_columns if column in base_columns]
    label_selected = [column for column in selected_columns if column not in base_columns]
    left = rows.loc[index, base_selected].reset_index(drop=True)
    right = label_features.loc[index, label_selected].reset_index(drop=True)
    matrix = pd.concat([left, right], axis=1)
    matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
    return matrix


def episode_weights(rows: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    selected = rows.loc[pd.Index(indices), ["SearchDate", "UserID", "SearchID"]]
    day = selected["SearchDate"].dt.floor("D").astype("int64").to_numpy()
    users = selected["UserID"].to_numpy(dtype=np.float64)
    search_ids = selected["SearchID"].to_numpy(dtype=np.int64)
    episode_user = np.where(np.isfinite(users), users, -search_ids).astype(np.int64)
    keys = pd.DataFrame({"day": day, "user": episode_user})
    sizes = keys.groupby(["day", "user"], sort=False)["user"].transform("size").to_numpy(dtype=np.float64)
    return (1 / np.sqrt(sizes)).astype(np.float32)


def _sample_indices(indices: np.ndarray, labels: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    selected_labels = labels[indices]
    positive = indices[selected_labels > 0.5]
    negative = indices[selected_labels <= 0.5]
    positive_n = max(1, int(limit * len(positive) / len(indices)))
    negative_n = limit - positive_n
    sampled = np.concatenate(
        [
            rng.choice(positive, min(positive_n, len(positive)), replace=False),
            rng.choice(negative, min(negative_n, len(negative)), replace=False),
        ]
    )
    if len(sampled) < limit:
        remaining = np.setdiff1d(indices, sampled, assume_unique=False)
        sampled = np.concatenate([sampled, rng.choice(remaining, limit - len(sampled), replace=False)])
    return np.sort(sampled)


def _fit_lightgbm(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    sample_weight: np.ndarray | None,
    categorical_columns: list[str],
    expert: str,
    trees: int,
    seed: int,
) -> lgb.LGBMClassifier:
    common = {
        "objective": "binary",
        "learning_rate": 0.04,
        "n_estimators": trees,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_lambda": 2.0,
        "reg_alpha": 0.05,
        "max_bin": 127,
        "n_jobs": 11,
        "random_state": seed,
        "verbosity": -1,
    }
    if expert == "repeat":
        common.update({"num_leaves": 191, "min_child_samples": 250})
    else:
        common.update({"num_leaves": 63, "min_child_samples": 500})
    model = lgb.LGBMClassifier(**common)
    active_categorical = [column for column in categorical_columns if column in matrix.columns]
    model.fit(
        matrix,
        labels,
        sample_weight=sample_weight,
        categorical_feature=active_categorical,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


@dataclass
class ExpertPair:
    repeat: lgb.LGBMClassifier
    cold: lgb.LGBMClassifier
    repeat_columns: list[str]
    cold_columns: list[str]


def train_experts(
    rows: pd.DataFrame,
    label_features: pd.DataFrame,
    train_indices: np.ndarray,
    labels: np.ndarray,
    base_columns: list[str],
    categorical_columns: list[str],
    debug: bool,
    seed: int,
    origin_model: bool = False,
) -> ExpertPair:
    start = time.time()
    label_columns = list(label_features.columns)
    repeat_columns, cold_columns = feature_sets(base_columns, label_columns)
    history = label_features.loc[pd.Index(train_indices), "user_history_count"].to_numpy()
    repeat_indices = train_indices[history > 0]
    cold_indices = train_indices[history <= 0]
    if debug:
        repeat_indices = _sample_indices(repeat_indices, labels, 35_000, seed)
        cold_indices = _sample_indices(cold_indices, labels, 15_000, seed + 1)
    if len(repeat_indices) < 100 or len(cold_indices) < 100:
        raise RuntimeError(f"Insufficient specialist rows: repeat={len(repeat_indices)} cold={len(cold_indices)}")
    repeat_matrix = assemble_matrix(rows, label_features, repeat_indices, base_columns, repeat_columns)
    repeat_trees = 100 if debug else (300 if origin_model else 520)
    repeat_model = _fit_lightgbm(
        repeat_matrix,
        labels[repeat_indices],
        None,
        categorical_columns + ["user_gap_bucket"],
        "repeat",
        repeat_trees,
        seed,
    )
    del repeat_matrix
    cold_matrix = assemble_matrix(rows, label_features, cold_indices, base_columns, cold_columns)
    weights = episode_weights(rows, cold_indices)
    cold_trees = 100 if debug else (260 if origin_model else 440)
    cold_model = _fit_lightgbm(
        cold_matrix,
        labels[cold_indices],
        weights,
        categorical_columns,
        "cold",
        cold_trees,
        seed + 17,
    )
    del cold_matrix
    print(
        f"[models] trained experts repeat_rows={len(repeat_indices)} cold_rows={len(cold_indices)} "
        f"trees={repeat_trees}/{cold_trees} elapsed={time.time() - start:.1f}s",
        flush=True,
    )
    return ExpertPair(repeat_model, cold_model, repeat_columns, cold_columns)


def predict_experts(
    experts: ExpertPair,
    rows: pd.DataFrame,
    label_features: pd.DataFrame,
    indices: np.ndarray,
    base_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    repeat_matrix = assemble_matrix(rows, label_features, indices, base_columns, experts.repeat_columns)
    repeat_predictions = experts.repeat.predict_proba(repeat_matrix)[:, 1]
    del repeat_matrix
    cold_matrix = assemble_matrix(rows, label_features, indices, base_columns, experts.cold_columns)
    cold_predictions = experts.cold.predict_proba(cold_matrix)[:, 1]
    del cold_matrix
    return repeat_predictions.astype(np.float32), cold_predictions.astype(np.float32)


def router_matrix(repeat: np.ndarray, cold: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    repeat_logit = _logit(repeat)
    cold_logit = _logit(cold)
    covered = (features["user_history_count"].to_numpy(dtype=np.float64) > 0).astype(np.float64)
    count = features["user_history_count"].to_numpy(dtype=np.float64)
    age = features["user_last_label_age"].to_numpy(dtype=np.float64)
    age = np.log1p(np.nan_to_num(age, nan=86400 * 30, posinf=86400 * 30) / 3600)
    return np.column_stack(
        [
            repeat_logit,
            cold_logit,
            covered,
            np.log1p(count),
            age,
            covered * repeat_logit,
            covered * cold_logit,
        ]
    )


def gated_predictions(
    repeat: np.ndarray,
    cold: np.ndarray,
    counts: np.ndarray,
    gate: tuple[float, float],
) -> np.ndarray:
    weights = np.where(counts <= 0, 0.0, np.where(counts == 1, gate[0], np.where(counts == 2, gate[1], 1.0)))
    return _sigmoid(weights * _logit(repeat) + (1 - weights) * _logit(cold))


def _origin_scores(labels: np.ndarray, predictions: np.ndarray, origins: np.ndarray) -> list[float]:
    return [safe_auc(labels[origins == origin], predictions[origins == origin]) for origin in np.unique(origins)]


def _score_pair(scores: list[float]) -> tuple[float, float]:
    finite = np.asarray([score for score in scores if np.isfinite(score)])
    return (float(finite.mean()), float(finite.min())) if len(finite) else (float("nan"), float("nan"))


def _select_gate(
    labels: np.ndarray,
    repeat: np.ndarray,
    cold: np.ndarray,
    counts: np.ndarray,
    origins: np.ndarray,
) -> tuple[tuple[float, float], dict]:
    best_gate = (0.35, 0.7)
    best_key = (-np.inf, -np.inf)
    best_scores = []
    for first in (0.0, 0.25, 0.5, 0.75, 1.0):
        for second in (0.0, 0.25, 0.5, 0.75, 1.0):
            predictions = gated_predictions(repeat, cold, counts, (first, second))
            scores = _origin_scores(labels, predictions, origins)
            mean, worst = _score_pair(scores)
            key = (worst, mean)
            if key > best_key:
                best_key = key
                best_gate = (first, second)
                best_scores = scores
    return best_gate, {"origin_auc": best_scores, "mean_auc": best_key[1], "worst_auc": best_key[0]}


@dataclass
class Router:
    kind: str
    gate: tuple[float, float]
    model: LogisticRegression | None
    diagnostics: dict


def fit_router(
    labels: np.ndarray,
    repeat: np.ndarray,
    cold: np.ndarray,
    label_features: pd.DataFrame,
    origins: np.ndarray,
) -> Router:
    counts = label_features["user_history_count"].to_numpy(dtype=np.float64)
    hard = np.where(counts > 0, repeat, cold)
    hard_scores = _origin_scores(labels, hard, origins)
    hard_mean, hard_worst = _score_pair(hard_scores)
    gate, gate_metrics = _select_gate(labels, repeat, cold, counts, origins)
    matrix = router_matrix(repeat, cold, label_features)
    best_c = 0.03
    best_key = (-np.inf, -np.inf)
    best_scores = []
    unique_origins = np.unique(origins)
    for c_value in (0.03, 0.1, 0.3):
        cross_predictions = np.empty(len(labels), dtype=np.float64)
        for origin in unique_origins:
            validation_mask = origins == origin
            training_mask = ~validation_mask
            if training_mask.sum() == 0:
                training_mask = validation_mask
            model = LogisticRegression(C=c_value, penalty="l2", solver="lbfgs", max_iter=500)
            model.fit(matrix[training_mask], labels[training_mask])
            cross_predictions[validation_mask] = model.predict_proba(matrix[validation_mask])[:, 1]
        scores = _origin_scores(labels, cross_predictions, origins)
        mean, worst = _score_pair(scores)
        key = (worst, mean)
        if key > best_key:
            best_key = key
            best_c = c_value
            best_scores = scores
    final_model = LogisticRegression(C=best_c, penalty="l2", solver="lbfgs", max_iter=500)
    final_model.fit(matrix, labels)
    coefficients = final_model.coef_[0]
    cold_dominant = coefficients[1] > coefficients[0]
    repeat_dominant = coefficients[0] + coefficients[5] > coefficients[1] + coefficients[6]
    improved = best_key[1] > hard_mean and best_key[0] > hard_worst
    accepted = bool(improved and cold_dominant and repeat_dominant)
    diagnostics = {
        "hard_origin_auc": hard_scores,
        "hard_mean_auc": hard_mean,
        "hard_worst_auc": hard_worst,
        "gate": list(gate),
        "gate_metrics": gate_metrics,
        "logistic_c": best_c,
        "logistic_origin_auc": best_scores,
        "logistic_mean_auc": best_key[1],
        "logistic_worst_auc": best_key[0],
        "logistic_coefficients": coefficients.tolist(),
        "cold_dominant": bool(cold_dominant),
        "repeat_dominant": bool(repeat_dominant),
        "accepted": accepted,
    }
    print(
        f"[router] hard mean/worst={hard_mean:.6f}/{hard_worst:.6f} "
        f"logistic={best_key[1]:.6f}/{best_key[0]:.6f} C={best_c} accepted={accepted} gate={gate}",
        flush=True,
    )
    return Router("logistic" if accepted else "gate", gate, final_model if accepted else None, diagnostics)


def apply_router(
    router: Router,
    repeat: np.ndarray,
    cold: np.ndarray,
    label_features: pd.DataFrame,
) -> np.ndarray:
    if router.kind == "logistic" and router.model is not None:
        return router.model.predict_proba(router_matrix(repeat, cold, label_features))[:, 1]
    counts = label_features["user_history_count"].to_numpy(dtype=np.float64)
    return gated_predictions(repeat, cold, counts, router.gate)


def slice_diagnostics(
    labels: np.ndarray,
    predictions: np.ndarray,
    label_features: pd.DataFrame,
) -> dict:
    counts = label_features["user_history_count"].to_numpy(dtype=np.float64)
    masks = {
        "cold": counts == 0,
        "history_1": counts == 1,
        "history_2": counts == 2,
        "history_3_plus": counts >= 3,
        "covered": counts > 0,
    }
    return {
        name: {"count": int(mask.sum()), "roc_auc": safe_auc(labels[mask], predictions[mask])}
        for name, mask in masks.items()
    }
