from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score


def lightgbm_params(seed: int = 1337) -> dict:
    return {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.04,
        "num_leaves": 127,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        "seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }


def train_fold(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    score_features: pd.DataFrame,
    score_labels: np.ndarray,
    rounds: int = 1600,
    seed: int = 1337,
) -> tuple[lgb.Booster, np.ndarray]:
    train_set = lgb.Dataset(train_features, label=train_labels, free_raw_data=True)
    score_set = lgb.Dataset(score_features, label=score_labels, reference=train_set, free_raw_data=True)
    model = lgb.train(
        lightgbm_params(seed),
        train_set,
        num_boost_round=rounds,
        valid_sets=[score_set],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    prediction = model.predict(score_features, num_iteration=model.best_iteration)
    return model, prediction


def train_fixed(
    features: pd.DataFrame,
    labels: np.ndarray,
    rounds: int,
    seed: int = 1337,
) -> lgb.Booster:
    dataset = lgb.Dataset(features, label=labels, free_raw_data=True)
    return lgb.train(
        lightgbm_params(seed),
        dataset,
        num_boost_round=int(rounds),
        callbacks=[lgb.log_evaluation(0)],
    )


def catboost_categories(features: pd.DataFrame) -> list[str]:
    candidates = [
        "position",
        "objecttype",
        "hour",
        "weekday",
        "adid",
        "userid",
        "ipid",
        "searchlocationid",
        "searchcategoryid",
        "adlocationid",
        "adcategoryid",
        "useragentid",
        "useragentosid",
        "userdeviceid",
        "useragentfamilyid",
        "searchparentcategoryid",
        "searchsubcategoryid",
        "adparentcategoryid",
        "adsubcategoryid",
        "searchregionid",
        "searchcityid",
        "adregionid",
        "adcityid",
    ]
    return [column for column in candidates if column in features.columns]


def prepare_catboost(features: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    prepared = features.copy(deep=False)
    for column in categories:
        prepared[column] = prepared[column].fillna(-1).astype(np.int64)
    return prepared


def train_catboost(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    score_features: pd.DataFrame,
) -> tuple[CatBoostClassifier, np.ndarray]:
    categories = catboost_categories(train_features)
    train_prepared = prepare_catboost(train_features, categories)
    score_prepared = prepare_catboost(score_features, categories)
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        task_type="GPU",
        devices="0",
        depth=8,
        learning_rate=0.05,
        iterations=800,
        l2_leaf_reg=10,
        random_seed=1337,
        random_strength=1.0,
        border_count=128,
        allow_writing_files=False,
        verbose=False,
        thread_count=int(os.environ.get("OMP_NUM_THREADS", "8")),
    )
    model.fit(train_prepared, train_labels, cat_features=categories, verbose=False)
    prediction = model.predict_proba(score_prepared)[:, 1]
    return model, prediction


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    output = np.empty_like(values, dtype=np.float64)
    output[positive] = 1 / (1 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1 + exponential)
    return output


def blend_prediction(model_prediction: np.ndarray, hist_ctr: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return np.asarray(model_prediction, dtype=np.float64)
    return sigmoid((1 - weight) * logit(model_prediction) + weight * logit(hist_ctr))


def auc(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(labels, prediction))


def slice_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    frame: pd.DataFrame,
    label_features: pd.DataFrame,
) -> dict:
    output: dict[str, dict] = {}
    strata = {
        "position": frame["Position"].fillna(-1).astype(str),
        "hist_available": frame["HistCTR"].notna().astype(str),
        "object_type": frame["ObjectType"].fillna(-1).astype(str),
        "ad_warm": (label_features["label_ad_count"] > 0).astype(str),
        "day": frame["SearchDate"].dt.strftime("%Y-%m-%d"),
    }
    for axis, values in strata.items():
        axis_result: dict[str, dict] = {}
        for value in sorted(values.unique()):
            mask = values.eq(value).to_numpy()
            positives = int(labels[mask].sum())
            negatives = int(mask.sum() - positives)
            score = auc(labels[mask], prediction[mask]) if positives > 0 and negatives > 0 else None
            axis_result[str(value)] = {"count": int(mask.sum()), "auc": score}
        output[axis] = axis_result
    return output


def write_diagnostics(path: Path, diagnostics: dict) -> None:
    path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True))
