from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class SidecarBundle:
    logistic: object
    lightgbm: object
    median: np.ndarray


@dataclass
class BlendChoice:
    weights: np.ndarray
    score: float
    fold_scores: list[float]


@dataclass
class CalibrationChoice:
    model: object | None
    enabled: bool
    brier_raw: float
    brier_calibrated: float


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    output = np.asarray(matrix, dtype=np.float64).copy()
    output[~np.isfinite(output)] = np.nan
    return output


def make_forward_folds(dates: pd.Series, debug: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    origins = np.unique(values)
    fold_count = 2 if debug else 3
    initial = max(8, int(len(origins) * 0.52))
    remaining = max(len(origins) - initial, fold_count)
    width = max(1, remaining // fold_count)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in range(fold_count):
        start = initial + fold * width
        stop = len(origins) if fold == fold_count - 1 else min(len(origins), start + width)
        if start >= len(origins) or stop <= start:
            continue
        valid_origins = origins[start:stop]
        valid_start = valid_origins[0]
        train_mask = values + np.timedelta64(30, "D") <= valid_start
        valid_mask = np.isin(values, valid_origins)
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(valid_mask)
        if len(train_idx) and len(valid_idx):
            folds.append((train_idx, valid_idx))
    return folds


def fit_sidecars(matrix: np.ndarray, labels: np.ndarray, debug: bool, seed: int) -> SidecarBundle:
    features = clean_matrix(matrix)
    median = np.nanmedian(features, axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(np.isnan(features), median, features)
    logistic = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.3, max_iter=1500, class_weight="balanced", random_state=seed),
    )
    lightgbm = LGBMClassifier(
        num_leaves=15,
        max_depth=5,
        min_child_samples=60,
        learning_rate=0.03,
        n_estimators=120 if debug else 400,
        reg_lambda=8.0,
        objective="binary",
        verbosity=-1,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
    )
    logistic.fit(filled, labels)
    lightgbm.fit(filled, labels)
    return SidecarBundle(logistic=logistic, lightgbm=lightgbm, median=median)


def predict_sidecars(bundle: SidecarBundle, matrix: np.ndarray) -> np.ndarray:
    features = clean_matrix(matrix)
    filled = np.where(np.isnan(features), bundle.median, features)
    logistic = bundle.logistic.predict_proba(filled)[:, 1]
    lightgbm = bundle.lightgbm.predict_proba(filled)[:, 1]
    return np.column_stack([logistic, lightgbm]).astype(np.float64)


def sidecar_oof(
    matrix: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    debug: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    folds = make_forward_folds(dates, debug)
    predictions = np.full((len(labels), 2), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int32)
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        bundle = fit_sidecars(matrix[train_idx], labels[train_idx], debug, seed + fold_id)
        predictions[valid_idx] = predict_sidecars(bundle, matrix[valid_idx])
        fold_ids[valid_idx] = fold_id
    return predictions, fold_ids, folds


def _weight_grid(columns: int) -> list[np.ndarray]:
    units = 5
    weights: list[np.ndarray] = []
    for values in itertools.product(range(units + 1), repeat=columns):
        if sum(values) == units:
            weights.append(np.asarray(values, dtype=np.float64) / units)
    return weights


def select_blend(predictions: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray) -> BlendChoice:
    valid = (fold_ids >= 0) & np.all(np.isfinite(predictions), axis=1)
    if not valid.any():
        weights = np.full(predictions.shape[1], 1.0 / predictions.shape[1])
        return BlendChoice(weights=weights, score=0.5, fold_scores=[])
    best: BlendChoice | None = None
    for weights in _weight_grid(predictions.shape[1]):
        blended = predictions @ weights
        scores: list[float] = []
        for fold in sorted(np.unique(fold_ids[valid])):
            mask = valid & (fold_ids == fold)
            if len(np.unique(labels[mask])) < 2:
                continue
            scores.append(float(roc_auc_score(labels[mask], blended[mask])))
        if not scores:
            continue
        objective = float(np.mean(scores) - np.var(scores))
        candidate = BlendChoice(weights=weights, score=objective, fold_scores=scores)
        if best is None or candidate.score > best.score + 1e-12:
            best = candidate
    if best is None:
        weights = np.full(predictions.shape[1], 1.0 / predictions.shape[1])
        return BlendChoice(weights=weights, score=0.5, fold_scores=[])
    return best


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def fit_oof_calibrator(blended: np.ndarray, labels: np.ndarray, fold_ids: np.ndarray) -> CalibrationChoice:
    valid_folds = sorted(int(value) for value in np.unique(fold_ids) if value >= 0)
    calibrated = np.full(len(labels), np.nan, dtype=np.float64)
    for fold in valid_folds[1:]:
        train_mask = (fold_ids >= 0) & (fold_ids < fold) & np.isfinite(blended)
        valid_mask = (fold_ids == fold) & np.isfinite(blended)
        if train_mask.sum() < 30 or len(np.unique(labels[train_mask])) < 2:
            continue
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(_logit(blended[train_mask]), labels[train_mask])
        calibrated[valid_mask] = model.predict_proba(_logit(blended[valid_mask]))[:, 1]
    compare = np.isfinite(calibrated)
    if compare.sum() < 30:
        return CalibrationChoice(None, False, np.nan, np.nan)
    brier_raw = float(brier_score_loss(labels[compare], blended[compare]))
    brier_calibrated = float(brier_score_loss(labels[compare], calibrated[compare]))
    auc_raw = float(roc_auc_score(labels[compare], blended[compare]))
    auc_calibrated = float(roc_auc_score(labels[compare], calibrated[compare]))
    enabled = brier_calibrated < brier_raw - 1e-5 and auc_calibrated >= auc_raw - 1e-4
    if not enabled:
        return CalibrationChoice(None, False, brier_raw, brier_calibrated)
    training = (fold_ids >= 0) & np.isfinite(blended)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(_logit(blended[training]), labels[training])
    if float(model.coef_[0, 0]) <= 0.0:
        return CalibrationChoice(None, False, brier_raw, brier_calibrated)
    return CalibrationChoice(model, True, brier_raw, brier_calibrated)


def apply_calibrator(choice: CalibrationChoice, values: np.ndarray) -> np.ndarray:
    if not choice.enabled or choice.model is None:
        return np.asarray(values, dtype=np.float64)
    return choice.model.predict_proba(_logit(values))[:, 1]


def context_indices(
    labels: np.ndarray,
    dates: pd.Series,
    maximum: int,
    policy: str,
    seed: int,
    debug: bool,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32)
    date_values = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    if len(labels) <= maximum:
        return np.arange(len(labels), dtype=np.int64)
    order = np.argsort(date_values, kind="stable")
    if policy == "last":
        return np.sort(order[-maximum:])
    pool = order[-min(len(order), 6000):] if debug else order
    negatives = pool[labels[pool] == 0]
    if len(negatives) >= maximum:
        return np.sort(negatives[-maximum:])
    positives = pool[labels[pool] == 1]
    capacity = maximum - len(negatives)
    if len(positives) <= capacity:
        return np.sort(np.concatenate([negatives, positives]))
    origins = pd.Series(pd.to_datetime(date_values[positives])).astype(str).to_numpy()
    years = pd.DatetimeIndex(date_values[positives]).year.to_numpy()
    eras = (years // 10) * 10
    origin_counts = pd.Series(origins).value_counts().to_dict()
    era_counts = pd.Series(eras).value_counts().to_dict()
    recency = (years - years.min() + 10.0) / (years.max() - years.min() + 10.0)
    weights = np.asarray(
        [recency[i] / math.sqrt(origin_counts[origins[i]] * era_counts[eras[i]]) for i in range(len(positives))],
        dtype=np.float64,
    )
    weights = weights / weights.sum()
    generator = np.random.default_rng(seed)
    chosen = generator.choice(positives, size=capacity, replace=False, p=weights)
    return np.sort(np.concatenate([negatives, chosen]))


def save_diagnostics(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
