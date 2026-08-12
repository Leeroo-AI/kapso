from __future__ import annotations

import json
import warnings
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_FEATURES = [
    "phase_number",
    "is_phase_2",
    "is_phase_3",
    "is_phase_4",
    "is_phase_na",
    "is_interventional",
    "log_enrollment",
    "log_arms",
    "trial_age_years",
    "is_randomized",
    "is_blinded",
    "eligibility_log_words",
    "eligibility_truncated",
    "condition_count",
    "intervention_count",
    "facility_count",
    "sponsor_industry",
    "sponsor_government",
    "sponsor_hist_log_count",
    "sponsor_hist_eb_rate",
    "sponsor_expected_analyses",
    "condition_hist_log_count",
    "condition_hist_eb_rate",
    "condition_expected_analyses",
    "global_history_rate",
    "expected_primary_analysis_count",
    "enrollment_origin_percentile",
    "sponsor_rate_origin_percentile",
    "condition_rate_origin_percentile",
]
LLM_FEATURES = [
    "llm_design_quality",
    "llm_endpoint_hardness",
    "llm_recruitment_burden",
    "llm_powering_plausibility",
    "llm_multiplicity",
    "llm_protocol_risk",
    "llm_area_respiratory_infectious",
    "llm_area_oncology",
]


@dataclass
class CompactConfig:
    family: str
    include_llm: bool
    blend_weight: float


def feature_columns(include_llm: bool) -> list[str]:
    columns = list(BASE_FEATURES)
    if include_llm:
        columns.extend(LLM_FEATURES)
    columns.append("encoder_probability")
    return columns


def _fit(frame: pd.DataFrame, labels: np.ndarray, family: str, include_llm: bool, seed: int):
    columns = feature_columns(include_llm)
    matrix = frame[columns].astype(float)
    if family == "logistic":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=0.2,
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    else:
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=100,
            max_depth=-1,
            reg_lambda=2.0,
            reg_alpha=0.1,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=seed,
            n_jobs=11,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(matrix, labels)
    return model


def predict_compact(model, frame: pd.DataFrame, include_llm: bool) -> np.ndarray:
    columns = feature_columns(include_llm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict_proba(frame[columns].astype(float))[:, 1], dtype=float)


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values)).rank(method="average", pct=True).to_numpy(dtype=float)


def rank_blend(
    encoder: np.ndarray,
    compact: np.ndarray,
    timestamps: np.ndarray,
    compact_weight: float,
) -> np.ndarray:
    if compact_weight <= 0.0:
        return np.asarray(encoder, dtype=float)
    if compact_weight >= 1.0:
        return np.asarray(compact, dtype=float)
    output = np.zeros(len(encoder), dtype=float)
    timestamp_series = pd.Series(pd.to_datetime(timestamps))
    for timestamp in timestamp_series.unique():
        indices = np.flatnonzero(timestamp_series.eq(timestamp).to_numpy())
        output[indices] = (
            (1.0 - compact_weight) * _rank(np.asarray(encoder)[indices])
            + compact_weight * _rank(np.asarray(compact)[indices])
        )
    return output


def paired_auc_se(
    labels: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    seed: int,
    draws: int = 300,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    differences = []
    count = len(labels)
    for _ in range(draws):
        indices = rng.integers(0, count, count)
        if len(np.unique(labels[indices])) < 2:
            continue
        differences.append(
            roc_auc_score(labels[indices], candidate[indices])
            - roc_auc_score(labels[indices], reference[indices])
        )
    improvement = roc_auc_score(labels, candidate) - roc_auc_score(labels, reference)
    return float(improvement), float(np.std(differences, ddof=1))


def select_compact(oof: pd.DataFrame, seed: int = 1337) -> tuple[CompactConfig, dict]:
    evaluation_years = [2018, 2019]
    candidates = {}
    prediction_store = {}
    for family in ["logistic", "lightgbm"]:
        for include_llm in [False, True]:
            name = f"{family}_{'llm' if include_llm else 'no_llm'}"
            fold_scores = []
            fold_predictions = {}
            for year in evaluation_years:
                train_mask = oof["origin_year"].between(2017, year - 1)
                valid_mask = oof["origin_year"].eq(year)
                train = oof[train_mask]
                valid = oof[valid_mask]
                if len(train) < 100 or valid["outcome"].nunique() < 2:
                    continue
                model = _fit(
                    train,
                    train["outcome"].to_numpy(dtype=float),
                    family,
                    include_llm,
                    seed + year,
                )
                predictions = predict_compact(model, valid, include_llm)
                score = roc_auc_score(valid["outcome"], predictions)
                fold_scores.append(float(score))
                fold_predictions[year] = predictions
            mean = float(np.mean(fold_scores)) if fold_scores else 0.5
            stability = float(np.std(fold_scores)) if fold_scores else 1.0
            candidates[name] = {
                "family": family,
                "include_llm": include_llm,
                "fold_auc": fold_scores,
                "mean_auc": mean,
                "stability": stability,
                "selection_value": mean - 0.1 * stability,
            }
            prediction_store[name] = fold_predictions
    selected_name = max(candidates, key=lambda name: candidates[name]["selection_value"])
    selected = candidates[selected_name]
    labels_all = []
    encoder_all = []
    compact_all = []
    fold_positions = []
    for year in evaluation_years:
        valid = oof[oof["origin_year"].eq(year)]
        prediction = prediction_store[selected_name].get(year)
        if prediction is None:
            continue
        labels_all.append(valid["outcome"].to_numpy(dtype=float))
        encoder_all.append(valid["encoder_probability"].to_numpy(dtype=float))
        compact_all.append(prediction)
        fold_positions.append((year, len(prediction)))
    labels = np.concatenate(labels_all)
    encoder = np.concatenate(encoder_all)
    compact = np.concatenate(compact_all)
    encoder_fold_scores = []
    weight_diagnostics = {}
    offset = 0
    fold_chunks = []
    for year, size in fold_positions:
        fold_chunks.append((year, slice(offset, offset + size)))
        offset += size
    for _, chunk in fold_chunks:
        encoder_fold_scores.append(float(roc_auc_score(labels[chunk], encoder[chunk])))
    for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        scores = []
        combined = np.zeros(len(labels), dtype=float)
        for _, chunk in fold_chunks:
            blended = rank_blend(
                encoder[chunk],
                compact[chunk],
                np.zeros(chunk.stop - chunk.start),
                weight,
            )
            combined[chunk] = blended
            scores.append(float(roc_auc_score(labels[chunk], blended)))
        improvement, se = paired_auc_se(labels, combined, encoder, seed + int(weight * 100))
        weight_diagnostics[str(weight)] = {
            "fold_auc": scores,
            "mean_auc": float(np.mean(scores)),
            "stability": float(np.std(scores)),
            "selection_value": float(np.mean(scores) - 0.1 * np.std(scores)),
            "pooled_improvement": improvement,
            "paired_bootstrap_se": se,
        }
    best_weight = max(
        weight_diagnostics,
        key=lambda weight: weight_diagnostics[weight]["selection_value"],
    )
    best = weight_diagnostics[best_weight]
    encoder_selection = float(np.mean(encoder_fold_scores) - 0.1 * np.std(encoder_fold_scores))
    if (
        best["selection_value"] <= encoder_selection
        or best["pooled_improvement"] < 0.8 * best["paired_bootstrap_se"]
    ):
        ship_weight = 0.0
    else:
        ship_weight = float(best_weight)
    without_llm = max(
        (value for value in candidates.values() if not value["include_llm"]),
        key=lambda value: value["selection_value"],
    )
    with_llm = max(
        (value for value in candidates.values() if value["include_llm"]),
        key=lambda value: value["selection_value"],
    )
    diagnostics = {
        "encoder_fold_auc": encoder_fold_scores,
        "encoder_mean_auc": float(np.mean(encoder_fold_scores)),
        "candidates": candidates,
        "selected_compact": selected_name,
        "llm_gate_delta": float(with_llm["selection_value"] - without_llm["selection_value"]),
        "blend_weights": weight_diagnostics,
        "selected_blend_weight": ship_weight,
    }
    print(f"[compact] selection={json.dumps(diagnostics, separators=(',', ':'))}")
    return CompactConfig(selected["family"], bool(selected["include_llm"]), ship_weight), diagnostics


def fit_final_compact(frame: pd.DataFrame, config: CompactConfig, seed: int):
    return _fit(
        frame,
        frame["outcome"].to_numpy(dtype=float),
        config.family,
        config.include_llm,
        seed,
    )


def bootstrap_auc_se(labels: np.ndarray, predictions: np.ndarray, seed: int, draws: int = 300) -> float:
    rng = np.random.default_rng(seed)
    values = []
    labels = np.asarray(labels)
    for _ in range(draws):
        indices = rng.integers(0, len(labels), len(labels))
        if len(np.unique(labels[indices])) == 2:
            values.append(roc_auc_score(labels[indices], predictions[indices]))
    return float(np.std(values, ddof=1))
