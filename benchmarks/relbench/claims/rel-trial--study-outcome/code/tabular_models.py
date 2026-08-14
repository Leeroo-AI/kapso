# Imports

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from text_models import forward_folds


# Selection

@dataclass
class TabularSelection:
    strength: float
    config: dict[str, int]
    rounds: int
    oof: np.ndarray
    mask: np.ndarray
    search_scores: dict[str, object]


def _parameters(config: dict[str, int], seed: int) -> dict[str, object]:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.025,
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "feature_fraction": 0.75,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbosity": -1,
        "seed": seed,
        "num_threads": 22,
    }


def _fold_predictions(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    timestamps: pd.Series,
    categorical: list[str],
    config: dict[str, int],
    seed: int,
    debug: bool,
) -> tuple[np.ndarray, list[int], list[float]]:
    prediction = np.full(len(labels), np.nan, dtype=np.float32)
    iterations = []
    fold_scores = []
    for train, validation in forward_folds(timestamps, debug):
        train_set = lgb.Dataset(
            matrix.iloc[train], label=labels[train], categorical_feature=categorical,
            free_raw_data=False,
        )
        validation_set = lgb.Dataset(
            matrix.iloc[validation], label=labels[validation], categorical_feature=categorical,
            reference=train_set, free_raw_data=False,
        )
        model = lgb.train(
            _parameters(config, seed), train_set,
            num_boost_round=80 if debug else 1400,
            valid_sets=[validation_set],
            callbacks=[lgb.early_stopping(60 if debug else 100, verbose=False), lgb.log_evaluation(0)],
        )
        current = model.predict(matrix.iloc[validation], num_iteration=model.best_iteration)
        prediction[validation] = current
        iterations.append(int(model.best_iteration))
        fold_scores.append(float(roc_auc_score(labels[validation], current)))
    return prediction, iterations, fold_scores


def select_tabular_model(
    matrices_by_strength: dict[float, pd.DataFrame],
    labels: np.ndarray,
    timestamps: pd.Series,
    categorical: list[str],
    debug: bool,
) -> TabularSelection:
    configurations = [
        {"num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 150},
        {"num_leaves": 31, "max_depth": 7, "min_data_in_leaf": 80},
    ]
    strengths = [next(iter(matrices_by_strength))] if debug else [10.0, 30.0, 100.0]
    records = []
    score_map = {}
    for strength in strengths:
        for config in configurations[:1] if debug else configurations:
            prediction, iterations, fold_scores = _fold_predictions(
                matrices_by_strength[strength], labels, timestamps, categorical,
                config, 17, debug,
            )
            mask = np.isfinite(prediction)
            pooled = float(roc_auc_score(labels[mask], prediction[mask]))
            stable = float(np.mean(fold_scores) - 0.25 * np.std(fold_scores))
            key = f"eb={int(strength)}|leaves={config['num_leaves']}|depth={config['max_depth']}|leaf={config['min_data_in_leaf']}"
            score_map[key] = {"pooled_auc": pooled, "stable_score": stable, "fold_auc": fold_scores, "iterations": iterations}
            records.append((stable, pooled, -config["num_leaves"], strength, config, prediction, iterations))
    _, _, _, strength, config, _, _ = max(records, key=lambda item: (item[0], item[1], item[2]))
    seed_predictions = []
    all_iterations = []
    selected_fold_scores = {}
    for seed in ([17] if debug else [17, 29, 43]):
        prediction, iterations, fold_scores = _fold_predictions(
            matrices_by_strength[strength], labels, timestamps, categorical,
            config, seed, debug,
        )
        seed_predictions.append(prediction)
        all_iterations.extend(iterations)
        selected_fold_scores[str(seed)] = fold_scores
    stacked = np.vstack(seed_predictions)
    oof = np.full(len(labels), np.nan, dtype=np.float32)
    available = np.isfinite(stacked).any(axis=0)
    oof[available] = np.nanmean(stacked[:, available], axis=0)
    mask = np.isfinite(oof)
    rounds = int(np.median(all_iterations)) if all_iterations else (50 if debug else 300)
    score_map["selected_seed_folds"] = selected_fold_scores
    return TabularSelection(float(strength), config, rounds, oof.astype(np.float32), mask, score_map)


# Final fit

def fit_tabular_predict(
    train_matrix: pd.DataFrame,
    labels: np.ndarray,
    predict_matrix: pd.DataFrame,
    categorical: list[str],
    config: dict[str, int],
    rounds: int,
    debug: bool,
) -> np.ndarray:
    predictions = []
    for seed in ([17] if debug else [17, 29, 43]):
        train_set = lgb.Dataset(
            train_matrix, label=labels, categorical_feature=categorical,
            free_raw_data=False,
        )
        model = lgb.train(
            _parameters(config, seed), train_set,
            num_boost_round=max(20, rounds),
            callbacks=[lgb.log_evaluation(0)],
        )
        predictions.append(model.predict(predict_matrix))
    return np.mean(predictions, axis=0).clip(1e-6, 1 - 1e-6).astype(np.float32)
