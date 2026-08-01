from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score

from f1_transport import TransportConfig, clip_by_race, decode_transport, rank_fractions


@dataclass(frozen=True)
class ModelSettings:
    debug: bool
    thresholds: tuple[int, ...]
    seeds: tuple[int, ...]
    sinkhorn_iterations: int

    @property
    def ordinal_trees(self) -> int:
        return 100 if self.debug else 700

    @property
    def ranker_trees(self) -> int:
        return 100 if self.debug else 800

    @property
    def l2_trees(self) -> int:
        return 100 if self.debug else 1500


@dataclass
class Heads:
    l2_models: list
    ordinal_models: list[list]
    ranker_models: list


@dataclass
class HeadPredictions:
    l2: np.ndarray
    ordinal: np.ndarray
    ranker: np.ndarray


@dataclass
class Calibrators:
    l2: IsotonicRegression
    ordinal: list[IsotonicRegression]
    ranker: IsotonicRegression


@dataclass
class Selection:
    config: TransportConfig
    baseline_r2: float
    selected_r2: float
    raw_transport_r2: float
    marginal_error: float
    fold_metrics: list[dict]


def _common(seed: int) -> dict:
    return {
        "num_leaves": 31,
        "verbosity": -1,
        "n_jobs": 11,
        "random_state": seed,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "deterministic": True,
        "force_col_wise": True,
    }


def _fit_ranker(
    features: pd.DataFrame,
    labels: np.ndarray,
    race_ids: np.ndarray,
    categorical: list[str],
    seed: int,
    trees: int,
) -> lgb.LGBMRanker:
    order = np.argsort(race_ids, kind="stable")
    ordered_races = race_ids[order]
    _, groups = np.unique(ordered_races, return_counts=True)
    sizes = pd.Series(race_ids).map(pd.Series(race_ids).value_counts()).to_numpy()
    relevance = np.maximum(np.rint(sizes - labels), 0).astype(int)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        learning_rate=0.04,
        min_child_samples=30,
        n_estimators=trees,
        label_gain=list(range(64)),
        **_common(seed),
    )
    model.fit(
        features.iloc[order],
        relevance[order],
        group=groups.tolist(),
        categorical_feature=categorical,
        callbacks=[lgb.log_evaluation(0)],
    )
    return model


def fit_heads(
    features: pd.DataFrame,
    labels: np.ndarray,
    race_ids: np.ndarray,
    categorical: list[str],
    settings: ModelSettings,
) -> Heads:
    l2_models = []
    ordinal_models = [[] for _ in settings.thresholds]
    ranker_models = []
    for seed in settings.seeds:
        l2_model = lgb.LGBMRegressor(
            objective="regression_l2",
            learning_rate=0.025,
            min_child_samples=40,
            reg_lambda=8,
            n_estimators=settings.l2_trees,
            **_common(seed),
        )
        l2_model.fit(
            features,
            labels,
            categorical_feature=categorical,
            callbacks=[lgb.log_evaluation(0)],
        )
        l2_models.append(l2_model)
        for index, threshold in enumerate(settings.thresholds):
            ordinal_model = lgb.LGBMClassifier(
                objective="binary",
                learning_rate=0.04,
                min_child_samples=40,
                reg_lambda=8,
                n_estimators=settings.ordinal_trees,
                **_common(seed),
            )
            ordinal_model.fit(
                features,
                (labels <= threshold).astype(int),
                categorical_feature=categorical,
                callbacks=[lgb.log_evaluation(0)],
            )
            ordinal_models[index].append(ordinal_model)
        ranker_models.append(
            _fit_ranker(
                features,
                labels,
                race_ids,
                categorical,
                seed,
                settings.ranker_trees,
            )
        )
    return Heads(l2_models, ordinal_models, ranker_models)


def predict_heads(heads: Heads, features: pd.DataFrame) -> HeadPredictions:
    l2 = np.mean([model.predict(features) for model in heads.l2_models], axis=0)
    ordinal = np.column_stack([
        np.mean([model.predict_proba(features)[:, 1] for model in models], axis=0)
        for models in heads.ordinal_models
    ])
    ranker = np.mean([model.predict(features) for model in heads.ranker_models], axis=0)
    return HeadPredictions(l2, ordinal, ranker)


def make_forward_folds(dates: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    years = dates.dt.year.to_numpy()
    maximum = int(years.max())
    start = maximum - 5
    folds = []
    for lower in [start, start + 2, start + 4]:
        train_indices = np.flatnonzero(years < lower)
        valid_indices = np.flatnonzero((years >= lower) & (years <= lower + 1))
        if len(train_indices) == 0 or len(valid_indices) == 0:
            raise RuntimeError(f"empty forward fold at {lower}")
        folds.append((train_indices, valid_indices))
    return folds


def forward_fold_years(dates: pd.Series) -> list[int]:
    maximum = int(dates.dt.year.max())
    start = maximum - 5
    return [start, start + 2, start + 4]


def generate_oof(
    features: pd.DataFrame,
    labels: np.ndarray,
    race_ids: np.ndarray,
    dates: pd.Series,
    categorical: list[str],
    settings: ModelSettings,
    fold_features: list[pd.DataFrame] | None = None,
    fold_categorical: list[list[str]] | None = None,
) -> tuple[HeadPredictions, np.ndarray]:
    l2 = np.full(len(labels), np.nan, dtype=float)
    ordinal = np.full((len(labels), len(settings.thresholds)), np.nan, dtype=float)
    ranker = np.full(len(labels), np.nan, dtype=float)
    fold_ids = np.full(len(labels), -1, dtype=int)
    for fold_id, (train_indices, valid_indices) in enumerate(make_forward_folds(dates)):
        current_features = features if fold_features is None else fold_features[fold_id]
        current_categorical = categorical if fold_categorical is None else fold_categorical[fold_id]
        heads = fit_heads(
            current_features.iloc[train_indices],
            labels[train_indices],
            race_ids[train_indices],
            current_categorical,
            settings,
        )
        predictions = predict_heads(heads, current_features.iloc[valid_indices])
        l2[valid_indices] = predictions.l2
        ordinal[valid_indices] = predictions.ordinal
        ranker[valid_indices] = predictions.ranker
        fold_ids[valid_indices] = fold_id
    return HeadPredictions(l2, ordinal, ranker), fold_ids


def fit_calibrators(
    predictions: HeadPredictions,
    labels: np.ndarray,
    race_ids: np.ndarray,
    thresholds: tuple[int, ...],
) -> Calibrators:
    mask = np.isfinite(predictions.l2)
    l2 = IsotonicRegression(out_of_bounds="clip").fit(predictions.l2[mask], labels[mask])
    ordinal = []
    for index, threshold in enumerate(thresholds):
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(
            predictions.ordinal[mask, index],
            (labels[mask] <= threshold).astype(float),
        )
        ordinal.append(calibrator)
    fractions = rank_fractions(predictions.ranker[mask], race_ids[mask])
    sizes = pd.Series(race_ids[mask]).map(pd.Series(race_ids[mask]).value_counts()).to_numpy()
    target = np.clip((labels[mask] - 1) / np.maximum(sizes - 1, 1), 0.0, 1.0)
    ranker = IsotonicRegression(out_of_bounds="clip").fit(fractions, target)
    return Calibrators(l2, ordinal, ranker)


def apply_calibration(predictions: HeadPredictions, calibrators: Calibrators) -> HeadPredictions:
    l2 = calibrators.l2.predict(predictions.l2)
    ordinal = np.column_stack([
        calibrator.predict(predictions.ordinal[:, index])
        for index, calibrator in enumerate(calibrators.ordinal)
    ])
    return HeadPredictions(l2, ordinal, predictions.ranker)


def select_transport(
    predictions: HeadPredictions,
    labels: np.ndarray,
    race_ids: np.ndarray,
    fold_ids: np.ndarray,
    thresholds: tuple[int, ...],
    calibrators: Calibrators,
    iterations: int,
) -> Selection:
    mask = fold_ids >= 0
    filtered = HeadPredictions(
        predictions.l2[mask],
        predictions.ordinal[mask],
        predictions.ranker[mask],
    )
    filtered = apply_calibration(filtered, calibrators)
    filtered_labels = labels[mask]
    filtered_races = race_ids[mask]
    filtered_folds = fold_ids[mask]
    baseline = clip_by_race(filtered.l2, filtered_races)
    baseline_score = float(r2_score(filtered_labels, baseline))
    best_score = -np.inf
    best_transport_score = -np.inf
    best_error = np.inf
    best_config = TransportConfig(0.5, 0.2, 2.0, 0.0)
    best_prediction = baseline
    for mixture in [0.3, 0.5, 0.7]:
        for entropy in [0.05, 0.1, 0.2, 0.5, 1.0]:
            for width in [1.0, 2.0, 3.0]:
                transport, error = decode_transport(
                    filtered.ordinal,
                    np.asarray(thresholds),
                    filtered.ranker,
                    filtered_races,
                    calibrators.ranker,
                    TransportConfig(mixture, entropy, width, 1.0),
                    iterations,
                )
                transport_score = float(r2_score(filtered_labels, transport))
                for blend in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
                    combined = blend * transport + (1.0 - blend) * baseline
                    score = float(r2_score(filtered_labels, combined))
                    if score > best_score:
                        best_score = score
                        best_transport_score = transport_score
                        best_error = error
                        best_config = TransportConfig(mixture, entropy, width, blend)
                        best_prediction = combined
    raw_fold_metrics = []
    catastrophic = False
    for fold_id in sorted(np.unique(filtered_folds)):
        fold_mask = filtered_folds == fold_id
        base_score = float(r2_score(filtered_labels[fold_mask], baseline[fold_mask]))
        selected_score = float(r2_score(filtered_labels[fold_mask], best_prediction[fold_mask]))
        raw_fold_metrics.append({
            "fold": int(fold_id),
            "count": int(fold_mask.sum()),
            "baseline_r2": base_score,
            "transport_blend_r2": selected_score,
            "delta_r2": selected_score - base_score,
        })
        catastrophic = catastrophic or selected_score - base_score < -0.02
    if best_score - baseline_score < 0.005 or catastrophic:
        best_config = TransportConfig(
            best_config.pmf_mixture,
            best_config.entropy,
            best_config.rank_width,
            0.0,
        )
        best_score = baseline_score
        for metric in raw_fold_metrics:
            metric["transport_blend_r2"] = metric["baseline_r2"]
            metric["delta_r2"] = 0.0
    return Selection(
        best_config,
        baseline_score,
        best_score,
        best_transport_score,
        best_error,
        raw_fold_metrics,
    )


def predict_chain(
    heads: Heads,
    features: pd.DataFrame,
    race_ids: np.ndarray,
    thresholds: tuple[int, ...],
    calibrators: Calibrators,
    config: TransportConfig,
    iterations: int,
) -> tuple[np.ndarray, float]:
    raw = predict_heads(heads, features)
    calibrated = apply_calibration(raw, calibrators)
    baseline = clip_by_race(calibrated.l2, race_ids)
    if config.blend == 0:
        return baseline, 0.0
    transport, error = decode_transport(
        calibrated.ordinal,
        np.asarray(thresholds),
        calibrated.ranker,
        race_ids,
        calibrators.ranker,
        config,
        iterations,
    )
    return config.blend * transport + (1.0 - config.blend) * baseline, error
