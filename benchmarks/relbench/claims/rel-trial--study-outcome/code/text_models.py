# Imports

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Text models

@dataclass
class TextSelection:
    channel: str
    selected_c: float
    oof: np.ndarray
    mask: np.ndarray
    scores: dict[str, float]


def forward_folds(timestamps: pd.Series, debug: bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    unique_times = sorted(pd.to_datetime(timestamps).unique())
    validation_times = unique_times[-(2 if debug else 4):]
    folds = []
    values = pd.to_datetime(timestamps)
    for timestamp in validation_times:
        train = np.flatnonzero((values + pd.Timedelta(days=365) <= timestamp).to_numpy())
        validation = np.flatnonzero((values == timestamp).to_numpy())
        if len(train) >= 50 and len(validation) >= 2:
            folds.append((train, validation))
    return folds


def _vectorizer(channel: str, debug: bool) -> TfidfVectorizer:
    if channel == "word":
        return TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), max_features=12000 if debug else 120000,
            min_df=2, max_df=0.995, sublinear_tf=True, strip_accents="unicode",
            dtype=np.float32,
        )
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), max_features=8000 if debug else 80000,
        min_df=2, max_df=0.995, sublinear_tf=True, dtype=np.float32,
    )


def _logistic(c_value: float, seed: int, debug: bool) -> LogisticRegression:
    return LogisticRegression(
        C=c_value, penalty="elasticnet", l1_ratio=0.05, solver="saga",
        max_iter=50 if debug else 180, tol=2e-3 if debug else 8e-4,
        random_state=seed, n_jobs=22,
    )


def select_text_channel(
    documents: list[str], labels: np.ndarray, timestamps: pd.Series, channel: str, debug: bool
) -> TextSelection:
    folds = forward_folds(timestamps, debug)
    c_values = [0.1] if debug else [0.03, 0.1, 0.3]
    predictions = {value: np.full(len(labels), np.nan, dtype=np.float32) for value in c_values}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold_number, (train, validation) in enumerate(folds):
            vectorizer = _vectorizer(channel, debug)
            train_matrix = vectorizer.fit_transform([documents[index] for index in train])
            validation_matrix = vectorizer.transform([documents[index] for index in validation])
            for c_value in c_values:
                model = _logistic(c_value, 17 + fold_number, debug)
                model.fit(train_matrix, labels[train])
                predictions[c_value][validation] = model.predict_proba(validation_matrix)[:, 1]
    mask = np.isfinite(next(iter(predictions.values())))
    scores = {str(value): float(roc_auc_score(labels[mask], prediction[mask])) for value, prediction in predictions.items()}
    selected = max(c_values, key=lambda value: scores[str(value)])
    return TextSelection(channel=channel, selected_c=float(selected), oof=predictions[selected], mask=mask, scores=scores)


def fit_text_predict(
    train_documents: list[str], labels: np.ndarray, predict_documents: list[str],
    channel: str, c_value: float, debug: bool, seed: int,
) -> np.ndarray:
    vectorizer = _vectorizer(channel, debug)
    train_matrix = vectorizer.fit_transform(train_documents)
    predict_matrix = vectorizer.transform(predict_documents)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = _logistic(c_value, seed, debug)
        model.fit(train_matrix, labels)
    return model.predict_proba(predict_matrix)[:, 1].astype(np.float32)


# Judgment models

@dataclass
class JudgmentSelection:
    selected_c: float
    oof: np.ndarray
    mask: np.ndarray
    scores: dict[str, float]
    copied_dimensions: list[int]
    r2_by_dimension: list[float]


def _residual_model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0))


def _judgment_logistic(c_value: float, seed: int, debug: bool) -> object:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=c_value, penalty="elasticnet", l1_ratio=0.1, solver="saga",
            max_iter=80 if debug else 400, tol=1e-3, random_state=seed,
        ),
    )


def select_judgment_channel(
    judgments: np.ndarray,
    embeddings: np.ndarray,
    no_llm_matrix: np.ndarray,
    labels: np.ndarray,
    timestamps: pd.Series,
    debug: bool,
) -> JudgmentSelection:
    folds = forward_folds(timestamps, debug)
    ridge_oof = np.full_like(judgments, np.nan, dtype=np.float32)
    for train, validation in folds:
        model = _residual_model()
        model.fit(no_llm_matrix[train], judgments[train])
        ridge_oof[validation] = model.predict(no_llm_matrix[validation])
    mask = np.isfinite(ridge_oof).all(axis=1)
    r2_values = [float(r2_score(judgments[mask, column], ridge_oof[mask, column])) for column in range(judgments.shape[1])]
    copied = [column for column, score in enumerate(r2_values) if score >= 0.8]
    kept = [column for column in range(judgments.shape[1]) if column not in copied]
    if not kept:
        kept = [int(np.argmin(r2_values))]
    c_values = [0.1] if debug else [0.03, 0.1, 0.3]
    predictions = {value: np.full(len(labels), np.nan, dtype=np.float32) for value in c_values}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for fold_number, (train, validation) in enumerate(folds):
            residual = _residual_model()
            residual.fit(no_llm_matrix[train], judgments[train])
            train_residual = judgments[train][:, kept] - residual.predict(no_llm_matrix[train])[:, kept]
            validation_residual = judgments[validation][:, kept] - residual.predict(no_llm_matrix[validation])[:, kept]
            train_matrix = np.column_stack([train_residual, embeddings[train]])
            validation_matrix = np.column_stack([validation_residual, embeddings[validation]])
            for c_value in c_values:
                model = _judgment_logistic(c_value, 41 + fold_number, debug)
                model.fit(train_matrix, labels[train])
                predictions[c_value][validation] = model.predict_proba(validation_matrix)[:, 1]
    prediction_mask = np.isfinite(next(iter(predictions.values())))
    scores = {str(value): float(roc_auc_score(labels[prediction_mask], prediction[prediction_mask])) for value, prediction in predictions.items()}
    selected = max(c_values, key=lambda value: scores[str(value)])
    return JudgmentSelection(float(selected), predictions[selected], prediction_mask, scores, copied, r2_values)


def fit_judgment_predict(
    train_judgments: np.ndarray,
    train_embeddings: np.ndarray,
    train_no_llm: np.ndarray,
    labels: np.ndarray,
    predict_judgments: np.ndarray,
    predict_embeddings: np.ndarray,
    predict_no_llm: np.ndarray,
    c_value: float,
    copied_dimensions: list[int],
    debug: bool,
) -> np.ndarray:
    kept = [column for column in range(train_judgments.shape[1]) if column not in copied_dimensions]
    if not kept:
        kept = [0]
    residual = _residual_model()
    residual.fit(train_no_llm, train_judgments)
    train_residual = train_judgments[:, kept] - residual.predict(train_no_llm)[:, kept]
    predict_residual = predict_judgments[:, kept] - residual.predict(predict_no_llm)[:, kept]
    model = _judgment_logistic(c_value, 73, debug)
    model.fit(np.column_stack([train_residual, train_embeddings]), labels)
    return model.predict_proba(np.column_stack([predict_residual, predict_embeddings]))[:, 1].astype(np.float32)


# Rank blending

def _within_time_rank(prediction: np.ndarray, timestamps: pd.Series) -> np.ndarray:
    result = np.zeros(len(prediction), dtype=np.float64)
    values = pd.to_datetime(timestamps)
    for timestamp in values.unique():
        index = np.flatnonzero((values == timestamp).to_numpy())
        result[index] = rankdata(prediction[index], method="average") / len(index)
    return result


def _simplex_weights(count: int, units: int = 10) -> list[np.ndarray]:
    values = []
    for dividers in itertools.combinations(range(units + count - 1), count - 1):
        points = (-1,) + dividers + (units + count - 1,)
        weights = np.array([points[index + 1] - points[index] - 1 for index in range(count)], dtype=float) / units
        values.append(weights)
    return values


def select_rank_blend(
    channels: dict[str, np.ndarray], labels: np.ndarray, timestamps: pd.Series, mask: np.ndarray
) -> tuple[dict[str, float], dict[str, object]]:
    names = list(channels)
    ranked = np.column_stack([_within_time_rank(channels[name], timestamps) for name in names])
    fold_times = pd.to_datetime(timestamps[mask]).unique()
    candidates = []
    for weights in _simplex_weights(len(names)):
        prediction = ranked @ weights
        fold_scores = []
        for timestamp in fold_times:
            fold_mask = mask & (pd.to_datetime(timestamps).to_numpy() == timestamp)
            if fold_mask.sum() > 1 and len(np.unique(labels[fold_mask])) > 1:
                fold_scores.append(roc_auc_score(labels[fold_mask], prediction[fold_mask]))
        pooled = roc_auc_score(labels[mask], prediction[mask])
        stable = float(np.mean(fold_scores) - 0.25 * np.std(fold_scores))
        candidates.append((stable, pooled, weights, fold_scores))
    stable, pooled, selected, fold_scores = max(candidates, key=lambda item: (item[0], item[1], -np.count_nonzero(item[2])))
    correlations = {}
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            correlations[f"{left}|{right}"] = float(spearmanr(channels[left][mask], channels[right][mask]).statistic)
    weights = {name: float(value) for name, value in zip(names, selected)}
    diagnostics = {"pooled_auc": float(pooled), "stable_score": float(stable), "fold_auc": [float(value) for value in fold_scores], "correlations": correlations}
    return weights, diagnostics


def apply_rank_blend(channels: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(next(iter(channels.values()))), dtype=np.float64)
    for name, prediction in channels.items():
        result += float(weights.get(name, 0.0)) * rankdata(prediction, method="average") / len(prediction)
    return result.clip(1e-6, 1 - 1e-6).astype(np.float64)
