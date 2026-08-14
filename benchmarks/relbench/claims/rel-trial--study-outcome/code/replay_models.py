# Imports

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# Replay channel

def replay_channel(
    matrix: pd.DataFrame,
    labels: np.ndarray,
    timestamps: pd.Series,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    seeds: tuple[int, ...] = (17, 29, 43),
) -> np.ndarray:
    predictions = []
    parameters = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.025,
        "num_leaves": 15, "max_depth": 5, "min_data_in_leaf": 150,
        "feature_fraction": 0.75, "bagging_fraction": 0.8, "bagging_freq": 1,
        "lambda_l2": 5.0, "verbosity": -1, "num_threads": 22,
    }
    for seed in seeds:
        train_set = lgb.Dataset(matrix.iloc[train_indices], label=labels[train_indices], free_raw_data=False)
        model = lgb.train(
            {**parameters, "seed": seed}, train_set, num_boost_round=300,
            callbacks=[lgb.log_evaluation(0)],
        )
        predictions.append(model.predict(matrix.iloc[predict_indices]))
    return np.mean(predictions, axis=0).astype(np.float32)


def replay_forward_selection(
    matrix: pd.DataFrame, labels: np.ndarray, timestamps: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    validation = np.flatnonzero((pd.to_datetime(timestamps) >= pd.Timestamp("2017-07-01")).to_numpy())
    train = np.flatnonzero((pd.to_datetime(timestamps) <= pd.Timestamp("2015-07-01")).to_numpy())
    prediction = replay_channel(matrix, labels, timestamps, train, validation, seeds=(17,))
    return validation, prediction, float(roc_auc_score(labels[validation], prediction))
