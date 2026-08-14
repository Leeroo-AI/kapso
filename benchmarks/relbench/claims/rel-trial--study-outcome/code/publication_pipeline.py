# Imports

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from campaign_io import locked_append, register_artifact
from kapso_datasets.common import load_task, run_data_dir, shared_cache_dir
from publication_evidence import (
    HOSTED_MODEL,
    PROMPT_VERSION,
    adjudicate_candidates,
    build_trial_contexts,
    prefilter_candidates,
    publication_features,
    retrieve_origin,
)
from registry_model import fit_registry_predict


# Configuration

START = time.time()
ORIGINS = {
    "official_2018": pd.Timestamp("2018-01-01"),
    "official_2019": pd.Timestamp("2019-01-01"),
    "validation_2020": pd.Timestamp("2020-01-01"),
    "test_2021": pd.Timestamp("2021-01-01"),
}
INVARIANT_WEIGHTS = {
    "tabular": 0.3,
    "word": 0.0,
    "char": 0.0,
    "judgment": 0.2,
    "judgment_v2": 0.3,
    "structural": 0.1,
    "external_compact": 0.1,
}
LITERATURE_COLUMNS = [
    "exact_si_count", "publication_count", "primary_report_count", "endpoint_match_confidence",
    "met_count", "not_met_count", "mixed_count", "explicit_p_significant_count",
    "explicit_p_nonsignificant_count", "final_count", "interim_count", "months_since_newest",
    "source_agreement", "evidence_confidence",
]


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[publication] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _auc(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        labels = labels[mask]
        prediction = prediction[mask]
    return float(roc_auc_score(labels, prediction)) if len(labels) >= 2 and len(np.unique(labels)) == 2 else float("nan")


def _ap(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        labels = labels[mask]
        prediction = prediction[mask]
    return float(average_precision_score(labels, prediction)) if len(labels) else float("nan")


def _ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values, method="average") / (len(values) + 1.0)


def _invariant_blend(matrix: dict[str, np.ndarray]) -> np.ndarray:
    result = np.zeros(len(next(iter(matrix.values()))), dtype=np.float64)
    for name, weight in INVARIANT_WEIGHTS.items():
        result += weight * _ranks(matrix[name])
    return np.clip(result, 1e-6, 1 - 1e-6)


def _routed_registry(incumbent: np.ndarray, registry: np.ndarray, linked: np.ndarray) -> np.ndarray:
    result = _ranks(incumbent)
    mask = np.asarray(linked, dtype=bool)
    result[mask] = 0.8 * result[mask] + 0.2 * _ranks(registry)[mask]
    return np.clip(result, 1e-6, 1 - 1e-6)


def _fit_nonnegative(matrix: np.ndarray, labels: np.ndarray, c_value: float) -> tuple[float, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(labels) < 10 or len(np.unique(labels)) < 2:
        return 0.0, np.zeros(matrix.shape[1], dtype=np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        scores = intercept + matrix @ coefficients
        probabilities = expit(scores).clip(1e-8, 1 - 1e-8)
        loss = -np.sum(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
        loss += 0.5 * np.sum(coefficients ** 2) / c_value
        residual = probabilities - labels
        gradient = np.concatenate([[residual.sum()], matrix.T @ residual + coefficients / c_value])
        return float(loss), gradient

    initial = np.concatenate([[float(np.log((labels.mean() + 1e-4) / (1 - labels.mean() + 1e-4)))], np.full(matrix.shape[1], 0.1)])
    bounds = [(None, None)] + [(0.0, None)] * matrix.shape[1]
    fitted = minimize(lambda parameters: objective(parameters), initial, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1000, "ftol": 1e-11})
    if not fitted.success:
        raise RuntimeError(f"Nonnegative stacker failed: {fitted.message}")
    return float(fitted.x[0]), fitted.x[1:].astype(np.float64)


def _predict_nonnegative(matrix: np.ndarray, model: tuple[float, np.ndarray]) -> np.ndarray:
    return expit(model[0] + np.asarray(matrix, dtype=np.float64) @ model[1]).clip(1e-6, 1 - 1e-6)


def _crossfit_nonnegative(matrix: np.ndarray, labels: np.ndarray, c_value: float) -> np.ndarray:
    prediction = np.full(len(labels), float(labels.mean()), dtype=np.float64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    for train, validation in folds.split(matrix, labels):
        model = _fit_nonnegative(matrix[train], labels[train], c_value)
        prediction[validation] = _predict_nonnegative(matrix[validation], model)
    return prediction


def _bootstrap(labels_by_origin: list[np.ndarray], incumbents: list[np.ndarray], candidates: list[np.ndarray], draws: int = 2000) -> dict[str, float]:
    random = np.random.default_rng(1337)
    deltas = []
    for _ in range(draws):
        fold_delta = []
        fold_weight = []
        for labels, incumbent, candidate in zip(labels_by_origin, incumbents, candidates):
            sampled = random.choice(np.arange(len(labels)), size=len(labels), replace=True)
            if len(np.unique(labels[sampled])) < 2:
                continue
            fold_delta.append(roc_auc_score(labels[sampled], candidate[sampled]) - roc_auc_score(labels[sampled], incumbent[sampled]))
            fold_weight.append(len(labels))
        if fold_delta:
            deltas.append(float(np.average(fold_delta, weights=fold_weight)))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "draws": int(len(values)),
        "mean_delta": float(values.mean()),
        "standard_error": float(values.std(ddof=1)),
        "probability_positive": float((values > 0).mean()),
        "lower_10": float(np.quantile(values, 0.10)),
        "upper_90": float(np.quantile(values, 0.90)),
    }


def _slice_metrics(labels: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray, covered: np.ndarray) -> dict[str, Any]:
    result = {}
    for name, mask in {"overall": np.ones(len(labels), dtype=bool), "covered": covered, "uncovered": ~covered}.items():
        result[name] = {
            "count": int(mask.sum()),
            "label_rate": float(labels[mask].mean()) if mask.any() else float("nan"),
            "incumbent_auc": _auc(labels, incumbent, mask),
            "candidate_auc": _auc(labels, candidate, mask),
            "incumbent_ap": _ap(labels, incumbent, mask),
            "candidate_ap": _ap(labels, candidate, mask),
        }
    result["prediction_correlation"] = float(spearmanr(incumbent, candidate).statistic)
    return result


# Artifact alignment

def _load_artifacts(cache: Path) -> dict[str, Any]:
    registry_root = cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2"
    invariant = np.load(cache / "predictions" / "generic_exp_3_invariant_channels_v1.npz", allow_pickle=False)
    registry = np.load(cache / "predictions" / "generic_exp_2_registry_clock_v1.npz", allow_pickle=False)
    return {
        "invariant": invariant,
        "registry": registry,
        "seeds": pd.read_parquet(registry_root / "seeds.parquet"),
        "linkage": pd.read_parquet(registry_root / "linkage.parquet"),
        "registry_features": pd.read_parquet(registry_root / "features_strength_100.parquet"),
        "projected_root": cache / "registry_clock_lane0" / "projected",
        "run0009_val": np.load(os.environ["RELBENCH_WORK_DIR"] + "/runs/run_0009/val_predictions.npy", allow_pickle=False),
        "run0009_test": np.load(os.environ["RELBENCH_WORK_DIR"] + "/runs/run_0009/test_predictions.npy", allow_pickle=False),
    }


def _align_invariant(artifacts: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    invariant = artifacts["invariant"]
    timestamp = ORIGINS[split]
    if split.startswith("official"):
        source_mask = pd.to_datetime(invariant["train_timestamp"]).to_numpy() == timestamp.to_datetime64()
        ids = invariant["train_nct_id"][source_mask].astype(np.int64)
        labels = invariant["train_labels"][source_mask].astype(np.int32)
        channels = {str(name): invariant[f"oof_{name}"][source_mask].astype(np.float64) for name in invariant["channel_names"]}
    elif split == "validation_2020":
        ids = invariant["validation_nct_id"].astype(np.int64)
        labels = np.asarray([], dtype=np.int32)
        channels = {str(name): invariant[f"validation_{name}"].astype(np.float64) for name in invariant["channel_names"]}
    else:
        ids = invariant["test_nct_id"].astype(np.int64)
        labels = np.asarray([], dtype=np.int32)
        channels = {str(name): invariant[f"test_{name}"].astype(np.float64) for name in invariant["channel_names"]}
    return ids, labels, channels


def _align_registry_rows(artifacts: dict[str, Any], split: str, ids: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    seeds = artifacts["seeds"].reset_index(drop=True)
    linkage = artifacts["linkage"].set_index("row_id")
    features = artifacts["registry_features"].set_index("row_id")
    rows = seeds[seeds["split"].eq(split)].copy()
    row_map = dict(zip(rows["nct_id"].astype(np.int64), rows["row_id"].astype(np.int64)))
    row_ids = np.asarray([row_map[int(value)] for value in ids], dtype=np.int64)
    aligned_features = features.loc[row_ids].reset_index()
    linked = linkage.loc[row_ids, "linked"].to_numpy(dtype=bool)
    return row_ids, aligned_features, linked


def _safe_registry_proxy(features: pd.DataFrame) -> np.ndarray:
    rate_columns = [column for column in features.columns if re_match_neighbor_rate(column)]
    count_columns = [column.replace("_rate", "_count") for column in rate_columns]
    rates = features[rate_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    counts = features[count_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    weights = np.log1p(np.nan_to_num(counts, nan=0.0)).clip(0.0, 8.0)
    numerator = np.nansum(np.nan_to_num(rates, nan=0.5) * weights, axis=1)
    denominator = weights.sum(axis=1)
    global_rate = pd.to_numeric(features["neighborhood_global_rate"], errors="coerce").fillna(0.5).to_numpy(dtype=np.float64)
    return np.where(denominator > 0, numerator / np.maximum(denominator, 1e-8), global_rate)


def re_match_neighbor_rate(column: str) -> bool:
    return column.startswith("neighbor_") and column.endswith("_rate") and not column.startswith("origin_")


def _registry_predictions_and_ablations(artifacts: dict[str, Any], run_ablations: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    ids_2018, labels_2018, _ = _align_invariant(artifacts, "official_2018")
    ids_2019, labels_2019, _ = _align_invariant(artifacts, "official_2019")
    _, features_2018, _ = _align_registry_rows(artifacts, "official_2018", ids_2018)
    _, features_2019, _ = _align_registry_rows(artifacts, "official_2019", ids_2019)
    proxy_2018 = _safe_registry_proxy(features_2018)
    external_2019 = fit_registry_predict(features_2018, labels_2018, pd.Series(ids_2018), features_2019, density_weighting=True)
    diagnostics = {
        "official_2018_safe_proxy_auc": _auc(labels_2018, proxy_2018),
        "official_2019_full_auc": _auc(labels_2019, external_2019),
        "ablations": {},
        "official_2018_supervised_registry_unavailable": "No complete pre-2018 label cohort has compatible point-in-time registry features; the earlier-origin vector uses only public-result neighborhood rates.",
    }
    if run_ablations:
        specifications = {
            "sponsor": lambda column: "sponsor" in column,
            "condition": lambda column: "condition" in column,
            "facility": lambda column: "facility" in column or "country" in column,
            "registry_evolution": lambda column: column in {"enrollment_revision", "actual_to_planned_enrollment_ratio", "completion_date_slippage_days", "primary_completion_date_slippage_days", "site_count_change", "primary_outcome_changed"} or any(token in column for token in ["enrollment_revision", "slippage", "site_count_change", "primary_outcome_changed"]),
            "snapshot_age": lambda column: column.startswith("days_since_") or column.startswith("origin_rank_days_since_") or column.startswith("origin_z_days_since_") or column.startswith("origin_gap_days_since_"),
        }
        full_auc = _auc(labels_2019, external_2019)
        for name, selector in specifications.items():
            keep = [column for column in features_2018.columns if column == "row_id" or not selector(column)]
            prediction = fit_registry_predict(features_2018[keep], labels_2018, pd.Series(ids_2018), features_2019[keep], density_weighting=True)
            diagnostics["ablations"][name] = {"auc": _auc(labels_2019, prediction), "delta_vs_full": _auc(labels_2019, prediction) - full_auc, "removed_columns": len(features_2018.columns) - len(keep)}
        no_density = fit_registry_predict(features_2018, labels_2018, pd.Series(ids_2018), features_2019, density_weighting=False)
        diagnostics["ablations"]["density_weighting"] = {"auc": _auc(labels_2019, no_density), "delta_vs_full": _auc(labels_2019, no_density) - full_auc, "removed_columns": 0}
    return {"official_2018": proxy_2018, "official_2019": external_2019, "validation_2020": artifacts["registry"]["external_val"].astype(np.float64), "test_2021": artifacts["registry"]["external_test"].astype(np.float64)}, diagnostics


def build_aligned_stacker(artifacts: dict[str, Any], registry_predictions: dict[str, np.ndarray]) -> dict[str, Any]:
    ids_2018, labels_2018, channels_2018 = _align_invariant(artifacts, "official_2018")
    ids_2019, labels_2019, channels_2019 = _align_invariant(artifacts, "official_2019")
    _, _, linked_2018 = _align_registry_rows(artifacts, "official_2018", ids_2018)
    _, _, linked_2019 = _align_registry_rows(artifacts, "official_2019", ids_2019)
    incumbent_2018 = _invariant_blend(channels_2018)
    incumbent_2019 = _invariant_blend(channels_2019)
    baseline_2018 = _routed_registry(incumbent_2018, registry_predictions["official_2018"], linked_2018)
    baseline_2019 = _routed_registry(incumbent_2019, registry_predictions["official_2019"], linked_2019)
    invariant = artifacts["invariant"]
    matrix_2018_seven = np.column_stack([_ranks(channels_2018[str(name)]) for name in invariant["channel_names"]])
    matrix_2018_eight = np.column_stack([matrix_2018_seven, _ranks(registry_predictions["official_2018"])])
    c_scores = {}
    predictions_2018 = {}
    for c_value in [0.03, 0.1, 0.3]:
        prediction = _crossfit_nonnegative(matrix_2018_eight, labels_2018, c_value)
        predictions_2018[c_value] = prediction
        c_scores[str(c_value)] = _auc(labels_2018, prediction)
    selected_c = max([0.03, 0.1, 0.3], key=lambda value: c_scores[str(value)])
    candidate_2018 = predictions_2018[selected_c]
    matrix_2019_eight = np.column_stack([*[_ranks(channels_2019[str(name)]) for name in invariant["channel_names"]], _ranks(registry_predictions["official_2019"])])
    model_2019 = _fit_nonnegative(matrix_2018_eight, labels_2018, selected_c)
    candidate_2019 = _predict_nonnegative(matrix_2019_eight, model_2019)
    bootstrap = _bootstrap([labels_2018, labels_2019], [baseline_2018, baseline_2019], [candidate_2018, candidate_2019])
    delta_2018 = _auc(labels_2018, candidate_2018) - _auc(labels_2018, baseline_2018)
    delta_2019 = _auc(labels_2019, candidate_2019) - _auc(labels_2019, baseline_2019)
    accepted = bool(delta_2018 >= 0 and delta_2019 >= 0 and bootstrap["mean_delta"] >= bootstrap["standard_error"] and bootstrap["probability_positive"] >= 0.8)
    val_ids, _, val_channels = _align_invariant(artifacts, "validation_2020")
    test_ids, _, test_channels = _align_invariant(artifacts, "test_2021")
    matrix_train = np.vstack([matrix_2018_eight, matrix_2019_eight])
    labels_train = np.concatenate([labels_2018, labels_2019])
    val_matrix = np.column_stack([*[_ranks(val_channels[str(name)]) for name in invariant["channel_names"]], _ranks(registry_predictions["validation_2020"])])
    test_matrix = np.column_stack([*[_ranks(test_channels[str(name)]) for name in invariant["channel_names"]], _ranks(registry_predictions["test_2021"])])
    model_a = _fit_nonnegative(matrix_train, labels_train, selected_c)
    validation_prediction = _predict_nonnegative(val_matrix, model_a)
    diagnostics = {
        "selected_c": selected_c,
        "regularization_scores_2018": c_scores,
        "coefficients_2019": model_2019[1].tolist(),
        "auc_2018": {"baseline": _auc(labels_2018, baseline_2018), "candidate": _auc(labels_2018, candidate_2018), "delta": delta_2018},
        "auc_2019": {"baseline": _auc(labels_2019, baseline_2019), "candidate": _auc(labels_2019, candidate_2019), "delta": delta_2019},
        "bootstrap": bootstrap,
        "accepted": accepted,
        "alignment": {"official_2018": len(ids_2018), "official_2019": len(ids_2019), "validation": len(val_ids), "test": len(test_ids)},
    }
    return {
        "ids_2018": ids_2018,
        "ids_2019": ids_2019,
        "labels_2018": labels_2018,
        "labels_2019": labels_2019,
        "baseline_2018": baseline_2018,
        "baseline_2019": baseline_2019,
        "candidate_2018": candidate_2018,
        "candidate_2019": candidate_2019,
        "validation_prediction": validation_prediction,
        "test_matrix": test_matrix,
        "train_matrix": matrix_train,
        "labels_train": labels_train,
        "selected_c": selected_c,
        "diagnostics": diagnostics,
    }


# Publication gate

def _oracle_prediction(labels: np.ndarray, incumbent: np.ndarray, covered: np.ndarray) -> np.ndarray:
    result = np.asarray(incumbent, dtype=np.float64).copy()
    result[covered & (labels == 1)] = 2.0
    result[covered & (labels == 0)] = -1.0
    return result


def _publication_type_profile(records: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if records.empty:
        return counts
    for values in records.loc[records["date_eligible"].astype(bool), "publication_types"]:
        for value in values if isinstance(values, list) else []:
            counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20])


def retrieve_gate_origins(artifacts: dict[str, Any], cache: Path) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, dict[str, Any]], dict[str, Any]]:
    records = {}
    candidates = {}
    contexts = {}
    diagnostics = {}
    for split in ["official_2018", "official_2019"]:
        origin = ORIGINS[split]
        current_records, retrieval = retrieve_origin(artifacts["linkage"], origin, cache)
        current_contexts = build_trial_contexts(artifacts["linkage"], origin, artifacts["projected_root"])
        current_candidates = prefilter_candidates(current_records, current_contexts, maximum=3)
        records[split] = current_records
        candidates[split] = current_candidates
        contexts[split] = current_contexts
        diagnostics[split] = {
            "retrieval": retrieval,
            "candidate_trials": int(current_candidates["queried_nct_id"].nunique()) if len(current_candidates) else 0,
            "candidate_records": int(len(current_candidates)),
            "publication_types": _publication_type_profile(current_records),
            "exact_si_share": float(current_records["exact_si"].mean()) if len(current_records) else 0.0,
        }
        report("retrieval", split=split, trials=retrieval["queried_trials"], coverage=f"{retrieval['coverage']:.4f}", candidates=len(current_candidates), rate=f"{retrieval['trials_per_minute']:.2f}")
    return records, candidates, contexts, diagnostics


def oracle_gate(aligned: dict[str, Any], candidates: dict[str, pd.DataFrame]) -> dict[str, Any]:
    result = {}
    deltas = []
    labels_list = []
    incumbent_list = []
    oracle_list = []
    for split, year in [("official_2018", "2018"), ("official_2019", "2019")]:
        ids = aligned[f"ids_{year}"]
        labels = aligned[f"labels_{year}"]
        incumbent = aligned[f"baseline_{year}"]
        evidence_ids = set(candidates[split]["queried_nct_id"].astype(str)) if len(candidates[split]) else set()
        linkage = artifacts_global["linkage"]
        current = linkage[linkage["split"].eq(split)].set_index("nct_id")
        external = np.asarray([str(current.loc[int(value), "external_nct_id"]) if int(value) in current.index else "" for value in ids])
        covered = np.asarray([value in evidence_ids for value in external], dtype=bool)
        oracle = _oracle_prediction(labels, incumbent, covered)
        delta = _auc(labels, oracle) - _auc(labels, incumbent)
        result[split] = {"covered": int(covered.sum()), "coverage": float(covered.mean()), "incumbent_auc": _auc(labels, incumbent), "oracle_auc": _auc(labels, oracle), "oracle_delta": delta}
        deltas.append(delta)
        labels_list.append(labels)
        incumbent_list.append(incumbent)
        oracle_list.append(oracle)
    bootstrap = _bootstrap(labels_list, incumbent_list, oracle_list)
    result["bootstrap"] = bootstrap
    result["passed"] = bool(min(deltas) > 0 and bootstrap["mean_delta"] >= bootstrap["standard_error"])
    return result


# Literature expert

def _feature_matrix(features: pd.DataFrame) -> np.ndarray:
    return features[LITERATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)


def _crossfit_literature(features: pd.DataFrame, labels: np.ndarray, c_value: float) -> np.ndarray:
    matrix = _feature_matrix(features)
    prediction = np.full(len(labels), float(labels.mean()), dtype=np.float64)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    for train, validation in folds.split(matrix, labels):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=1000, random_state=1337))
        model.fit(matrix[train], labels[train])
        prediction[validation] = model.predict_proba(matrix[validation])[:, 1]
    return prediction


def _fit_literature(features: pd.DataFrame, labels: np.ndarray, predict: pd.DataFrame, c_value: float) -> np.ndarray:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, penalty="l2", solver="liblinear", max_iter=1000, random_state=1337))
    model.fit(_feature_matrix(features), labels)
    return model.predict_proba(_feature_matrix(predict))[:, 1]


def _routing_matrix(incumbent: np.ndarray, registry: np.ndarray, literature: np.ndarray, confidence: np.ndarray) -> np.ndarray:
    return np.column_stack([_ranks(incumbent), _ranks(registry), _ranks(literature), np.asarray(confidence, dtype=np.float64) / 5.0])


def _crossfit_route(matrix: np.ndarray, labels: np.ndarray, covered: np.ndarray, incumbent: np.ndarray, c_value: float) -> np.ndarray:
    prediction = np.asarray(incumbent, dtype=np.float64).copy()
    covered_indices = np.flatnonzero(covered)
    if len(covered_indices) < 20 or len(np.unique(labels[covered_indices])) < 2:
        return prediction
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    for train_local, validation_local in folds.split(matrix[covered_indices], labels[covered_indices]):
        train = covered_indices[train_local]
        validation = covered_indices[validation_local]
        model = _fit_nonnegative(matrix[train], labels[train], c_value)
        prediction[validation] = _predict_nonnegative(matrix[validation], model)
    return prediction


def _fit_route(train_matrix: np.ndarray, train_labels: np.ndarray, train_covered: np.ndarray, predict_matrix: np.ndarray, predict_covered: np.ndarray, incumbent: np.ndarray, c_value: float) -> tuple[np.ndarray, tuple[float, np.ndarray]]:
    result = np.asarray(incumbent, dtype=np.float64).copy()
    model = _fit_nonnegative(train_matrix[train_covered], train_labels[train_covered], c_value)
    result[predict_covered] = _predict_nonnegative(predict_matrix[predict_covered], model)
    return result, model


def _features_for_split(artifacts: dict[str, Any], split: str, records: pd.DataFrame, adjudications: pd.DataFrame, ids: np.ndarray) -> pd.DataFrame:
    current_linkage = artifacts["linkage"][artifacts["linkage"]["split"].eq(split)].copy()
    features = publication_features(current_linkage, records, adjudications)
    mapping = features.set_index("nct_id")
    return mapping.loc[ids.astype(np.int64)].reset_index()


def fit_publication_gate(artifacts: dict[str, Any], aligned: dict[str, Any], registry_predictions: dict[str, np.ndarray], records: dict[str, pd.DataFrame], adjudications: dict[str, pd.DataFrame]) -> dict[str, Any]:
    features_2018 = _features_for_split(artifacts, "official_2018", records["official_2018"], adjudications["official_2018"], aligned["ids_2018"])
    features_2019 = _features_for_split(artifacts, "official_2019", records["official_2019"], adjudications["official_2019"], aligned["ids_2019"])
    labels_2018 = aligned["labels_2018"]
    labels_2019 = aligned["labels_2019"]
    c_scores = {}
    oof_literature = {}
    for c_value in [0.03, 0.1, 0.3]:
        prediction = _crossfit_literature(features_2018, labels_2018, c_value)
        oof_literature[c_value] = prediction
        mask = features_2018["usable_evidence"].to_numpy(dtype=bool)
        c_scores[str(c_value)] = _auc(labels_2018, prediction, mask)
    selected_c = max([0.03, 0.1, 0.3], key=lambda value: c_scores[str(value)])
    literature_2018 = oof_literature[selected_c]
    literature_2019 = _fit_literature(features_2018, labels_2018, features_2019, selected_c)
    covered_2018 = features_2018["usable_evidence"].to_numpy(dtype=bool)
    covered_2019 = features_2019["usable_evidence"].to_numpy(dtype=bool)
    confidence_2018 = features_2018["evidence_confidence"].to_numpy(dtype=float)
    confidence_2019 = features_2019["evidence_confidence"].to_numpy(dtype=float)
    matrix_2018 = _routing_matrix(aligned["baseline_2018"], registry_predictions["official_2018"], literature_2018, confidence_2018)
    matrix_2019 = _routing_matrix(aligned["baseline_2019"], registry_predictions["official_2019"], literature_2019, confidence_2019)
    routing_scores = {}
    candidate_2018_by_c = {}
    for c_value in [0.03, 0.1, 0.3]:
        prediction = _crossfit_route(matrix_2018, labels_2018, covered_2018, aligned["baseline_2018"], c_value)
        candidate_2018_by_c[c_value] = prediction
        routing_scores[str(c_value)] = _auc(labels_2018, prediction)
    routing_c = max([0.03, 0.1, 0.3], key=lambda value: routing_scores[str(value)])
    candidate_2018 = candidate_2018_by_c[routing_c]
    candidate_2019, routing_model = _fit_route(matrix_2018, labels_2018, covered_2018, matrix_2019, covered_2019, aligned["baseline_2019"], routing_c)
    delta_2018 = _auc(labels_2018, candidate_2018) - _auc(labels_2018, aligned["baseline_2018"])
    delta_2019 = _auc(labels_2019, candidate_2019) - _auc(labels_2019, aligned["baseline_2019"])
    bootstrap = _bootstrap([labels_2018, labels_2019], [aligned["baseline_2018"], aligned["baseline_2019"]], [candidate_2018, candidate_2019])
    accepted = bool(delta_2018 >= 0 and delta_2019 >= 0 and bootstrap["mean_delta"] >= bootstrap["standard_error"] and bootstrap["probability_positive"] >= 0.8)
    diagnostics = {
        "literature_c": selected_c,
        "literature_c_scores_earlier_origin": c_scores,
        "routing_c": routing_c,
        "routing_scores_earlier_origin": routing_scores,
        "routing_coefficients_sealed_later_origin": routing_model[1].tolist(),
        "bootstrap": bootstrap,
        "accepted": accepted,
        "official_2018": _slice_metrics(labels_2018, aligned["baseline_2018"], candidate_2018, covered_2018),
        "official_2019": _slice_metrics(labels_2019, aligned["baseline_2019"], candidate_2019, covered_2019),
        "delta_2018": delta_2018,
        "delta_2019": delta_2019,
    }
    return {
        "features_2018": features_2018,
        "features_2019": features_2019,
        "literature_2018": literature_2018,
        "literature_2019": literature_2019,
        "matrix_2018": matrix_2018,
        "matrix_2019": matrix_2019,
        "covered_2018": covered_2018,
        "covered_2019": covered_2019,
        "selected_c": selected_c,
        "routing_c": routing_c,
        "diagnostics": diagnostics,
    }


# Model A and Model B

def _retrieve_and_adjudicate_split(artifacts: dict[str, Any], split: str, cache: Path, debug: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    origin = ORIGINS[split]
    records, retrieval = retrieve_origin(artifacts["linkage"], origin, cache)
    contexts = build_trial_contexts(artifacts["linkage"], origin, artifacts["projected_root"])
    candidates = prefilter_candidates(records, contexts, maximum=3)
    if debug:
        available = []
        adjudication_root = cache / "literature_v2" / "adjudications"
        for index, row in candidates.iterrows():
            from publication_evidence import _adjudication_key
            if (adjudication_root / f"{_adjudication_key(row, contexts[str(row['queried_nct_id'])])}.json").exists():
                available.append(index)
            if len(available) >= 8:
                break
        selected = candidates.loc[available] if available else candidates.head(1)
        adjudications, hosted = adjudicate_candidates(selected, contexts, cache, concurrency=4)
    else:
        adjudications, hosted = adjudicate_candidates(candidates, contexts, cache, concurrency=32)
    return records, adjudications, {"retrieval": retrieval, "candidates": len(candidates), "hosted": hosted}


def build_final_candidate(artifacts: dict[str, Any], aligned: dict[str, Any], registry_predictions: dict[str, np.ndarray], publication_gate: dict[str, Any], records: dict[str, pd.DataFrame], adjudications: dict[str, pd.DataFrame], cache: Path, debug: bool) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    val_records, val_adjudications, val_diagnostics = _retrieve_and_adjudicate_split(artifacts, "validation_2020", cache, debug)
    val_ids, _, _ = _align_invariant(artifacts, "validation_2020")
    val_features = _features_for_split(artifacts, "validation_2020", val_records, val_adjudications, val_ids)
    train_features = pd.concat([publication_gate["features_2018"], publication_gate["features_2019"]], ignore_index=True)
    train_labels = np.concatenate([aligned["labels_2018"], aligned["labels_2019"]])
    literature_val = _fit_literature(train_features, train_labels, val_features, publication_gate["selected_c"])
    train_matrix = np.vstack([publication_gate["matrix_2018"], publication_gate["matrix_2019"]])
    train_covered = np.concatenate([publication_gate["covered_2018"], publication_gate["covered_2019"]])
    val_covered = val_features["usable_evidence"].to_numpy(dtype=bool)
    val_matrix = _routing_matrix(artifacts["run0009_val"], registry_predictions["validation_2020"], literature_val, val_features["evidence_confidence"].to_numpy(dtype=float))
    validation_prediction, model_a = _fit_route(train_matrix, train_labels, train_covered, val_matrix, val_covered, artifacts["run0009_val"], publication_gate["routing_c"])
    freeze_root = cache / "literature_v2" / "model_a"
    freeze_root.mkdir(parents=True, exist_ok=True)
    validation_path = freeze_root / "validation_predictions.npy"
    np.save(validation_path, validation_prediction.astype(np.float64))
    validation_checksum = sha256_file(validation_path)
    report("model_a_saved", checksum=validation_checksum, covered=int(val_covered.sum()), validation_labels_loaded=False)
    context = load_task()
    validation_labels = context.val.df[context.target_col].to_numpy(dtype=np.int32)
    test_records, test_adjudications, test_diagnostics = _retrieve_and_adjudicate_split(artifacts, "test_2021", cache, debug)
    test_ids, _, _ = _align_invariant(artifacts, "test_2021")
    test_features = _features_for_split(artifacts, "test_2021", test_records, test_adjudications, test_ids)
    combined_features = pd.concat([train_features, val_features], ignore_index=True)
    combined_labels = np.concatenate([train_labels, validation_labels])
    literature_test = _fit_literature(combined_features, combined_labels, test_features, publication_gate["selected_c"])
    combined_matrix = np.vstack([train_matrix, val_matrix])
    combined_covered = np.concatenate([train_covered, val_covered])
    test_covered = test_features["usable_evidence"].to_numpy(dtype=bool)
    test_matrix = _routing_matrix(artifacts["run0009_test"], registry_predictions["test_2021"], literature_test, test_features["evidence_confidence"].to_numpy(dtype=float))
    test_prediction, model_b = _fit_route(combined_matrix, combined_labels, combined_covered, test_matrix, test_covered, artifacts["run0009_test"], publication_gate["routing_c"])
    if sha256_file(validation_path) != validation_checksum:
        raise RuntimeError("Model A validation checksum changed after validation labels were loaded")
    diagnostics = {
        "validation_checksum": validation_checksum,
        "model_a_coefficients": model_a[1].tolist(),
        "model_b_coefficients": model_b[1].tolist(),
        "validation_covered": int(val_covered.sum()),
        "test_covered": int(test_covered.sum()),
        "validation": val_diagnostics,
        "test": test_diagnostics,
        "validation_prediction_label_fit": "official_2018_and_official_2019_training_labels_only",
        "test_prediction_label_fit": "official_2018_official_2019_plus_validation_labels",
    }
    return validation_prediction, test_prediction, diagnostics


def persist_candidate(cache: Path, validation: np.ndarray, test: np.ndarray, diagnostics: dict[str, Any]) -> Path:
    path = cache / "predictions" / "generic_exp_4_publication_evidence_v2.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, val=np.asarray(validation, dtype=np.float64), test=np.asarray(test, dtype=np.float64), diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, path)
    register_artifact(cache, {
        "name": "generic_exp_4 pre-origin publication evidence candidate",
        "path": "predictions/generic_exp_4_publication_evidence_v2.npz",
        "description": "Model-A validation and Model-B test vectors from the gated PubMed/Europe PMC primary-result expert over the exact run_0009 fallback.",
        "content_key": "rel-trial-study-outcome:generic-exp-4:publication-evidence-v2",
        "rebuild_hint": "Run publication_pipeline.py after populating literature_v2; validation is physically frozen before Model B.",
    })
    return path


# Orchestration

artifacts_global: dict[str, Any] = {}


def run(stage: str, debug: bool) -> None:
    global artifacts_global
    cache = shared_cache_dir()
    artifacts = _load_artifacts(cache)
    artifacts_global = artifacts
    run0009_val_hash = sha256_file(Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0009" / "val_predictions.npy")
    run0009_test_hash = sha256_file(Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0009" / "test_predictions.npy")
    if run0009_val_hash != "ede536cf338713b2d7e9f64a6bcb03a57c55d537f97393826a1e7724b16bd852" or run0009_test_hash != "348bb8e8f9b894189a8c00ccb0f76d638e73c9c77c78aa180dff1c10a565931a":
        raise RuntimeError("run_0009 incumbent checksum mismatch")
    report("incumbent", validation_hash=run0009_val_hash, test_hash=run0009_test_hash)
    registry_predictions, registry_diagnostics = _registry_predictions_and_ablations(artifacts, run_ablations=stage == "full" and not debug)
    aligned = build_aligned_stacker(artifacts, registry_predictions)
    report("aligned_stacker", diagnostics=json.dumps(aligned["diagnostics"], sort_keys=True))
    aligned_path = cache / "predictions" / "generic_exp_4_aligned_registry_stacker_v2.npz"
    with aligned_path.with_suffix(".npz.part").open("wb") as stream:
        np.savez_compressed(stream, official_2018=aligned["candidate_2018"], official_2019=aligned["candidate_2019"], validation=aligned["validation_prediction"], diagnostics_json=np.asarray([json.dumps(aligned["diagnostics"], sort_keys=True)]))
    os.replace(aligned_path.with_suffix(".npz.part"), aligned_path)
    records, candidates, contexts, retrieval_diagnostics = retrieve_gate_origins(artifacts, cache)
    oracle = oracle_gate(aligned, candidates)
    report("oracle_gate", diagnostics=json.dumps(oracle, sort_keys=True))
    partial_diagnostics = {"registry": registry_diagnostics, "aligned_stacker": aligned["diagnostics"], "retrieval": retrieval_diagnostics, "oracle": oracle}
    diagnostics_root = cache / "literature_v2"
    diagnostics_root.mkdir(parents=True, exist_ok=True)
    (diagnostics_root / "gate_diagnostics.json").write_text(json.dumps(partial_diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    if stage == "retrieve":
        return
    if not oracle["passed"]:
        diagnostics = {**partial_diagnostics, "hosted_killed_by_oracle": True, "final_source": "aligned_stacker" if aligned["diagnostics"]["accepted"] else "run_0009"}
        validation = aligned["validation_prediction"] if aligned["diagnostics"]["accepted"] else artifacts["run0009_val"]
        test = artifacts["run0009_test"]
        persist_candidate(cache, validation, test, diagnostics)
        return
    combined_candidates = pd.concat([candidates["official_2018"], candidates["official_2019"]], ignore_index=True)
    combined_contexts = {**contexts["official_2018"], **contexts["official_2019"]}
    probe, probe_diagnostics = adjudicate_candidates(combined_candidates, combined_contexts, cache, concurrency=1, probe_only=True)
    report("hosted_probe", diagnostics=json.dumps(probe_diagnostics, sort_keys=True), rows=len(probe))
    adjudications = {}
    hosted_diagnostics = {"probe": probe_diagnostics}
    for split in ["official_2018", "official_2019"]:
        adjudications[split], current = adjudicate_candidates(candidates[split], contexts[split], cache, concurrency=8 if debug else 32)
        hosted_diagnostics[split] = current
        report("hosted", split=split, calls=current["calls"], cache_hits=current["cache_hits"], usable=current["usable_rows"])
    publication_gate = fit_publication_gate(artifacts, aligned, registry_predictions, records, adjudications)
    report("publication_gate", diagnostics=json.dumps(publication_gate["diagnostics"], sort_keys=True))
    diagnostics = {**partial_diagnostics, "hosted": hosted_diagnostics, "publication_gate": publication_gate["diagnostics"], "hosted_model": HOSTED_MODEL, "prompt_version": PROMPT_VERSION}
    if publication_gate["diagnostics"]["accepted"]:
        validation, test, model_diagnostics = build_final_candidate(artifacts, aligned, registry_predictions, publication_gate, records, adjudications, cache, debug)
        diagnostics["model_a_b"] = model_diagnostics
        diagnostics["final_source"] = "publication_evidence"
    elif aligned["diagnostics"]["accepted"]:
        validation = aligned["validation_prediction"]
        test = artifacts["run0009_test"]
        diagnostics["final_source"] = "aligned_stacker_model_a_only_test_fallback"
    else:
        validation = artifacts["run0009_val"]
        test = artifacts["run0009_test"]
        diagnostics["final_source"] = "run_0009"
    candidate_path = persist_candidate(cache, validation, test, diagnostics)
    output = run_data_dir()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "val_predictions.npy", validation.astype(np.float64))
    np.save(output / "test_predictions.npy", test.astype(np.float64))
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    feature_status = "TESTED-KEPT" if publication_gate["diagnostics"]["accepted"] else "TESTED-REJECTED"
    locked_append(cache / "features_history.md", f'''\n### Pre-origin PubMed and Europe PMC publication evidence expert — measurement\n- run/experiment: generic_exp_4 lane 0 | status: {feature_status}\n- what: Exact accession PubMed/Europe PMC retrieval with strict origin dates, deterministic three-paper prefilter, hosted primary-endpoint adjudication, regularized logistic literature expert, and evidence-confident nonnegative routing.\n- outcome: retrieval {json.dumps(retrieval_diagnostics, sort_keys=True)}; oracle {json.dumps(oracle, sort_keys=True)}; gate {json.dumps(publication_gate["diagnostics"], sort_keys=True)}.\n- takeaway: uncovered rows retain run_0009 exactly; acceptance requires nonnegative deltas at both origins and a pooled improvement of at least one paired SE.\n''')
    for name, values in registry_diagnostics["ablations"].items():
        locked_append(cache / "features_history.md", f'''\n### Registry block ablation: {name}\n- run/experiment: generic_exp_4 lane 0 | status: TESTED-{"KEPT" if values["delta_vs_full"] < 0 else "REJECTED"}\n- what: Drop the {name} block from the warm strength-100 registry expert and refit the official-2018 to official-2019 gate.\n- outcome: AUC {values["auc"]:.9f}; delta versus full {values["delta_vs_full"]:+.9f}; removed columns {values["removed_columns"]}.\n- takeaway: Negative drop delta supports retaining the block; nonnegative drop delta indicates no measured contribution on the sealed gate.\n''')
    locked_append(cache / "table_information.md", f'''\n### 2026-08-13 pre-origin publication retrieval\n- PubMed query template: exact `<NCT>[si]` with server-side publication-date maximum origin minus one day; Europe PMC query template: exact `ACCESSION_ID:<NCT>` with `FIRST_PDATE` maximum origin minus one day.\n- Independent date policy accepts complete electronic/first-publication dates only when strictly before the row origin, or year-only dates only when strictly before the origin year. Citation counts, corrections, later registry state, and post-origin documents are not used.\n- Gate retrieval diagnostics: {json.dumps(retrieval_diagnostics, sort_keys=True)}.\n''')
    register_artifact(cache, {"name": "generic_exp_4 aligned registry stacker", "path": "predictions/generic_exp_4_aligned_registry_stacker_v2.npz", "description": "Aligned official-2018/2019 invariant-channel plus registry predictions and sealed nonnegative stacker diagnostics.", "content_key": "rel-trial-study-outcome:generic-exp-4:aligned-registry-stacker-v2", "rebuild_hint": "Run publication_pipeline.py from the warm strength-100 registry feature cache."})
    report("complete", candidate=candidate_path, source=diagnostics["final_source"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["retrieve", "full"], default="full")
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    run(arguments.stage, arguments.debug)


if __name__ == "__main__":
    main()
