from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from registry_clock import RegistryFeatureBundle


MODEL_VERSION = "registry_expert_v1"


@dataclass
class RegistryModelAResult:
    strength: float
    blend_weight: float
    literature_c: float
    literature_weight: float
    external_validation: np.ndarray
    validation_prediction: np.ndarray
    external_forward_index: np.ndarray
    external_forward_prediction: np.ndarray
    diagnostics: dict[str, Any]


def _encode_features(train: pd.DataFrame, predict: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_columns = {}
    predict_columns = {}
    excluded = {"row_id"}
    for column in train.columns:
        if column in excluded:
            continue
        train_values = train[column]
        predict_values = predict[column]
        if train_values.dtype == object or str(train_values.dtype).startswith("string") or str(train_values.dtype) == "category":
            vocabulary = {value: index for index, value in enumerate(sorted(train_values.fillna("__missing__").astype(str).unique()))}
            train_columns[column] = train_values.fillna("__missing__").astype(str).map(vocabulary).fillna(-1).astype(np.int32)
            predict_columns[column] = predict_values.fillna("__missing__").astype(str).map(vocabulary).fillna(-1).astype(np.int32)
        else:
            numeric_train = pd.to_numeric(train_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            numeric_predict = pd.to_numeric(predict_values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            median = float(numeric_train.median()) if numeric_train.notna().any() else 0.0
            train_columns[column] = numeric_train.fillna(median).astype(np.float32)
            predict_columns[column] = numeric_predict.fillna(median).astype(np.float32)
    return pd.DataFrame(train_columns, index=train.index), pd.DataFrame(predict_columns, index=predict.index)


def _density_ratio_weights(train: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    train_encoded, target_encoded = _encode_features(train, target)
    combined = pd.concat([train_encoded, target_encoded], ignore_index=True)
    labels = np.concatenate([np.zeros(len(train_encoded), dtype=np.int32), np.ones(len(target_encoded), dtype=np.int32)])
    if len(train_encoded) < 20 or len(target_encoded) < 20 or combined.shape[1] == 0:
        return np.ones(len(train_encoded), dtype=np.float64)
    model = LogisticRegression(C=0.1, max_iter=300, solver="liblinear", random_state=1337)
    model.fit(combined, labels)
    probability = model.predict_proba(train_encoded)[:, 1].clip(1e-4, 1 - 1e-4)
    ratio = probability / (1.0 - probability) * (len(train_encoded) / len(target_encoded))
    return np.clip(ratio, 0.25, 4.0)


def _group_balance_weights(nct_ids: pd.Series) -> np.ndarray:
    counts = nct_ids.astype(str).map(nct_ids.astype(str).value_counts()).to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def fit_registry_predict(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    train_nct_ids: pd.Series,
    predict_features: pd.DataFrame,
    density_weighting: bool = True,
) -> np.ndarray:
    train_encoded, predict_encoded = _encode_features(train_features, predict_features)
    weights = _group_balance_weights(train_nct_ids)
    if density_weighting:
        weights *= _density_ratio_weights(train_features, predict_features)
    weights *= len(weights) / weights.sum()
    dataset = lgb.Dataset(train_encoded, label=train_labels, weight=weights, free_raw_data=False)
    model = lgb.train(
        {
            "objective": "binary", "metric": "auc", "num_leaves": 15, "max_depth": 4,
            "min_data_in_leaf": 80, "learning_rate": 0.025, "lambda_l2": 10.0,
            "verbosity": -1, "seed": 1337, "num_threads": 22,
        },
        dataset,
        num_boost_round=500,
        callbacks=[lgb.log_evaluation(0)],
    )
    return np.asarray(model.predict(predict_encoded), dtype=np.float64)


def literature_columns(features: pd.DataFrame) -> list[str]:
    return [column for column in features.columns if column.startswith("registry_result_reference_")]


def registry_columns(features: pd.DataFrame) -> list[str]:
    literature = set(literature_columns(features))
    return [column for column in features.columns if column not in literature]


def fit_literature_predict(
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    train_nct_ids: pd.Series,
    predict_features: pd.DataFrame,
    c_value: float,
) -> np.ndarray:
    columns = literature_columns(train_features)
    train_encoded, predict_encoded = _encode_features(train_features[columns], predict_features[columns])
    weights = _group_balance_weights(train_nct_ids)
    weights *= len(weights) / weights.sum()
    model = LogisticRegression(
        C=c_value, penalty="elasticnet", solver="saga", l1_ratio=0.1,
        max_iter=1000, random_state=1337, tol=1e-4,
    )
    model.fit(train_encoded, train_labels, sample_weight=weights)
    return np.asarray(model.predict_proba(predict_encoded)[:, 1], dtype=np.float64)


def normalized_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values, method="average") / (len(values) + 1.0)


def routed_rank_blend(incumbent: np.ndarray, external: np.ndarray, linked: np.ndarray, weight: float) -> np.ndarray:
    if weight <= 0:
        return np.asarray(incumbent, dtype=np.float64).copy()
    incumbent_rank = normalized_rank(incumbent)
    external_rank = normalized_rank(external)
    routed_weight = np.asarray(linked, dtype=np.float64) * weight
    return np.clip((1.0 - routed_weight) * incumbent_rank + routed_weight * external_rank, 1e-6, 1 - 1e-6)


def _paired_bootstrap(labels: np.ndarray, incumbent: np.ndarray, candidate: np.ndarray, draws: int = 2000) -> dict[str, float]:
    random = np.random.default_rng(1337)
    deltas = []
    indices = np.arange(len(labels))
    for _ in range(draws):
        sampled = random.choice(indices, size=len(indices), replace=True)
        if len(np.unique(labels[sampled])) < 2:
            continue
        deltas.append(roc_auc_score(labels[sampled], candidate[sampled]) - roc_auc_score(labels[sampled], incumbent[sampled]))
    values = np.asarray(deltas, dtype=float)
    return {
        "draws": int(len(values)),
        "mean_delta": float(values.mean()),
        "standard_error": float(values.std(ddof=1)),
        "probability_positive": float((values > 0).mean()),
        "lower_10pct": float(np.quantile(values, 0.10)),
        "upper_90pct": float(np.quantile(values, 0.90)),
    }


def _slice_auc(labels: np.ndarray, predictions: np.ndarray, mask: np.ndarray) -> float:
    return float(roc_auc_score(labels[mask], predictions[mask])) if mask.sum() >= 2 and len(np.unique(labels[mask])) == 2 else float("nan")


def _labels_for_indices(label_map: pd.Series, seeds: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    labels = seeds.iloc[indices]["row_id"].map(label_map)
    if labels.isna().any():
        raise RuntimeError(f"Missing labels for {int(labels.isna().sum())} registry rows")
    return labels.to_numpy(dtype=np.int32)


def select_registry_model_a(
    bundle: RegistryFeatureBundle,
    label_map: pd.Series,
    incumbent_cache: Path,
    incumbent_validation: np.ndarray,
    official_train_nct_ids: pd.Series,
) -> RegistryModelAResult:
    seeds = bundle.seeds.reset_index(drop=True)
    split = seeds["split"].astype(str)
    gate_a_train = np.flatnonzero((split == "replay_2017").to_numpy())
    gate_a_validation = np.flatnonzero((split == "replay_2018").to_numpy())
    gate_b_train = np.flatnonzero((split == "official_2018").to_numpy())
    gate_b_validation = np.flatnonzero((split == "official_2019").to_numpy())
    validation_indices = np.flatnonzero((split == "validation_2020").to_numpy())
    source_indices = np.flatnonzero(split.isin(["replay_2017", "official_2018", "replay_2018", "official_2019"]).to_numpy())
    incumbent = np.load(incumbent_cache, allow_pickle=False)
    forward_index = incumbent["forward_index"].astype(int)
    if "train_nct_id" in incumbent.files:
        forward_ids = incumbent["train_nct_id"][forward_index].astype(str)
        forward_prediction = incumbent["forward_prediction"].astype(float)
    else:
        forward_ids = official_train_nct_ids.iloc[forward_index].astype(str).to_numpy()
        forward_prediction = incumbent["blend_forward"].astype(float)
    forward_map = dict(zip(forward_ids, forward_prediction))
    gate_b_incumbent = seeds.iloc[gate_b_validation]["nct_id"].astype(str).map(forward_map)
    comparable = gate_b_incumbent.notna().to_numpy()
    if comparable.sum() < 500:
        raise RuntimeError(f"Incumbent forward cache covers only {int(comparable.sum())} Gate B rows")
    minimal_features = bundle.features_by_strength[20.0][[
        column for column in bundle.features_by_strength[20.0].columns
        if "neighbor_" not in column and not column.startswith("registry_result_reference_")
    ]]
    minimal_prediction_a = fit_registry_predict(
        minimal_features.iloc[gate_a_train], _labels_for_indices(label_map, seeds, gate_a_train),
        seeds.iloc[gate_a_train]["nct_id"], minimal_features.iloc[gate_a_validation],
    )
    minimal_prediction_b = fit_registry_predict(
        minimal_features.iloc[gate_b_train], _labels_for_indices(label_map, seeds, gate_b_train),
        seeds.iloc[gate_b_train]["nct_id"], minimal_features.iloc[gate_b_validation],
    )
    minimal_diagnostics = {
        "gate_a_auc": float(roc_auc_score(_labels_for_indices(label_map, seeds, gate_a_validation), minimal_prediction_a)),
        "gate_b_auc": float(roc_auc_score(_labels_for_indices(label_map, seeds, gate_b_validation), minimal_prediction_b)),
    }
    literature_diagnostics = {}
    literature_predictions = {}
    for c_value in [0.03, 0.1, 0.3]:
        prediction_a = fit_literature_predict(
            bundle.features_by_strength[20.0].iloc[gate_a_train], _labels_for_indices(label_map, seeds, gate_a_train),
            seeds.iloc[gate_a_train]["nct_id"], bundle.features_by_strength[20.0].iloc[gate_a_validation], c_value,
        )
        prediction_b = fit_literature_predict(
            bundle.features_by_strength[20.0].iloc[gate_b_train], _labels_for_indices(label_map, seeds, gate_b_train),
            seeds.iloc[gate_b_train]["nct_id"], bundle.features_by_strength[20.0].iloc[gate_b_validation], c_value,
        )
        auc_a = float(roc_auc_score(_labels_for_indices(label_map, seeds, gate_a_validation), prediction_a))
        auc_b = float(roc_auc_score(_labels_for_indices(label_map, seeds, gate_b_validation), prediction_b))
        literature_diagnostics[str(c_value)] = {
            "gate_a_auc": auc_a, "gate_b_auc": auc_b,
            "stability_adjusted_auc": (auc_a + auc_b) / 2.0 - 0.15 * abs(auc_a - auc_b),
        }
        literature_predictions[c_value] = (prediction_a, prediction_b)
    literature_c = max([0.03, 0.1, 0.3], key=lambda value: literature_diagnostics[str(value)]["stability_adjusted_auc"])
    strength_diagnostics = {}
    gate_predictions = {}
    for strength, features in bundle.features_by_strength.items():
        labels_a_train = _labels_for_indices(label_map, seeds, gate_a_train)
        labels_a_validation = _labels_for_indices(label_map, seeds, gate_a_validation)
        labels_b_train = _labels_for_indices(label_map, seeds, gate_b_train)
        labels_b_validation = _labels_for_indices(label_map, seeds, gate_b_validation)
        selected_registry = features[registry_columns(features)]
        prediction_a = fit_registry_predict(
            selected_registry.iloc[gate_a_train], labels_a_train, seeds.iloc[gate_a_train]["nct_id"], selected_registry.iloc[gate_a_validation]
        )
        prediction_b = fit_registry_predict(
            selected_registry.iloc[gate_b_train], labels_b_train, seeds.iloc[gate_b_train]["nct_id"], selected_registry.iloc[gate_b_validation]
        )
        auc_a = float(roc_auc_score(labels_a_validation, prediction_a))
        auc_b = float(roc_auc_score(labels_b_validation, prediction_b))
        strength_diagnostics[str(int(strength))] = {
            "gate_a_auc": auc_a,
            "gate_b_auc": auc_b,
            "mean_auc": (auc_a + auc_b) / 2.0,
            "stability_adjusted_auc": (auc_a + auc_b) / 2.0 - 0.15 * abs(auc_a - auc_b),
        }
        gate_predictions[strength] = (prediction_a, prediction_b)
    strength = max(bundle.features_by_strength, key=lambda value: strength_diagnostics[str(int(value))]["stability_adjusted_auc"])
    features = bundle.features_by_strength[strength]
    literature_a, literature_b = literature_predictions[literature_c]
    reference_column = "registry_result_reference_count"
    evidence_a = features.iloc[gate_a_validation][reference_column].to_numpy(dtype=float) > 0
    evidence_b_all = features.iloc[gate_b_validation][reference_column].to_numpy(dtype=float) > 0
    literature_weight_scores = {}
    labels_a = _labels_for_indices(label_map, seeds, gate_a_validation)
    labels_b_all = _labels_for_indices(label_map, seeds, gate_b_validation)
    for weight in np.linspace(0.0, 0.5, 6):
        combined_a = routed_rank_blend(gate_predictions[strength][0], literature_a, evidence_a, float(weight))
        combined_b = routed_rank_blend(gate_predictions[strength][1], literature_b, evidence_b_all, float(weight))
        auc_a = float(roc_auc_score(labels_a, combined_a))
        auc_b = float(roc_auc_score(labels_b_all, combined_b))
        literature_weight_scores[f"{weight:.1f}"] = {
            "gate_a_auc": auc_a, "gate_b_auc": auc_b,
            "stability_adjusted_auc": (auc_a + auc_b) / 2.0 - 0.15 * abs(auc_a - auc_b),
        }
    literature_weight = max(np.linspace(0.0, 0.5, 6), key=lambda value: literature_weight_scores[f"{value:.1f}"]["stability_adjusted_auc"])
    gate_a_prediction = routed_rank_blend(gate_predictions[strength][0], literature_a, evidence_a, float(literature_weight))
    gate_b_prediction_all = routed_rank_blend(gate_predictions[strength][1], literature_b, evidence_b_all, float(literature_weight))
    gate_b_prediction = gate_b_prediction_all[comparable]
    gate_b_labels = _labels_for_indices(label_map, seeds, gate_b_validation)[comparable]
    gate_b_incumbent_values = gate_b_incumbent.to_numpy(dtype=float)[comparable]
    linked_gate_b = bundle.linkage.set_index("row_id").loc[seeds.iloc[gate_b_validation]["row_id"], "linked"].to_numpy(dtype=bool)[comparable]
    weight_scores = {}
    for weight in np.linspace(0.0, 1.0, 11):
        candidate = routed_rank_blend(gate_b_incumbent_values, gate_b_prediction, linked_gate_b, float(weight))
        weight_scores[f"{weight:.1f}"] = float(roc_auc_score(gate_b_labels, candidate))
    blend_weight = max(np.linspace(0.0, 1.0, 11), key=lambda value: weight_scores[f"{value:.1f}"])
    selected_gate_prediction = routed_rank_blend(gate_b_incumbent_values, gate_b_prediction, linked_gate_b, float(blend_weight))
    bootstrap = _paired_bootstrap(gate_b_labels, gate_b_incumbent_values, selected_gate_prediction)
    missing_mask = ~linked_gate_b
    no_evidence_incumbent = _slice_auc(gate_b_labels, gate_b_incumbent_values, missing_mask)
    no_evidence_candidate = _slice_auc(gate_b_labels, selected_gate_prediction, missing_mask)
    serious_no_evidence_loss = bool(np.isfinite(no_evidence_incumbent) and np.isfinite(no_evidence_candidate) and no_evidence_candidate < no_evidence_incumbent - 0.02)
    accepted = bool(bootstrap["probability_positive"] >= 0.8 and not serious_no_evidence_loss and blend_weight > 0)
    if not accepted:
        blend_weight = 0.0
    source_labels = _labels_for_indices(label_map, seeds, source_indices)
    selected_registry = features[registry_columns(features)]
    registry_validation = fit_registry_predict(
        selected_registry.iloc[source_indices], source_labels, seeds.iloc[source_indices]["nct_id"], selected_registry.iloc[validation_indices]
    )
    literature_validation = fit_literature_predict(
        features.iloc[source_indices], source_labels, seeds.iloc[source_indices]["nct_id"],
        features.iloc[validation_indices], literature_c,
    )
    evidence_validation = features.iloc[validation_indices][reference_column].to_numpy(dtype=float) > 0
    external_validation = routed_rank_blend(registry_validation, literature_validation, evidence_validation, float(literature_weight))
    linked_validation = bundle.linkage.set_index("row_id").loc[seeds.iloc[validation_indices]["row_id"], "linked"].to_numpy(dtype=bool)
    validation_prediction = routed_rank_blend(incumbent_validation, external_validation, linked_validation, float(blend_weight))
    diagnostics = {
        "model_version": MODEL_VERSION,
        "minimal_registry_clock": minimal_diagnostics,
        "literature_expert": literature_diagnostics,
        "literature_c": literature_c,
        "literature_weight_scores": literature_weight_scores,
        "literature_weight": float(literature_weight),
        "literature_gate_a_auc": float(roc_auc_score(labels_a, literature_a)),
        "literature_gate_b_auc": float(roc_auc_score(labels_b_all, literature_b)),
        "literature_gate_a_evidence_share": float(evidence_a.mean()),
        "literature_gate_b_evidence_share": float(evidence_b_all.mean()),
        "external_gate_a_auc_after_literature": float(roc_auc_score(labels_a, gate_a_prediction)),
        "selected_strength": strength,
        "strength_diagnostics": strength_diagnostics,
        "weight_scores": weight_scores,
        "selected_weight_before_gate": float(max(np.linspace(0.0, 1.0, 11), key=lambda value: weight_scores[f"{value:.1f}"])),
        "selected_weight": float(blend_weight),
        "bootstrap": bootstrap,
        "accepted": accepted,
        "gate_b_incumbent_auc": float(roc_auc_score(gate_b_labels, gate_b_incumbent_values)),
        "gate_b_external_auc": float(roc_auc_score(gate_b_labels, gate_b_prediction)),
        "gate_b_blend_auc": float(roc_auc_score(gate_b_labels, selected_gate_prediction)),
        "gate_b_rank_correlation": float(spearmanr(gate_b_incumbent_values, gate_b_prediction).statistic),
        "gate_b_rows": int(len(gate_b_labels)),
        "gate_b_linked_share": float(linked_gate_b.mean()),
        "no_evidence_count": int(missing_mask.sum()),
        "no_evidence_incumbent_auc": no_evidence_incumbent,
        "no_evidence_candidate_auc": no_evidence_candidate,
        "serious_no_evidence_loss": serious_no_evidence_loss,
    }
    return RegistryModelAResult(
        strength=float(strength),
        blend_weight=float(blend_weight),
        literature_c=float(literature_c),
        literature_weight=float(literature_weight),
        external_validation=external_validation,
        validation_prediction=validation_prediction,
        external_forward_index=seeds.iloc[gate_b_validation]["row_id"].to_numpy()[comparable],
        external_forward_prediction=gate_b_prediction,
        diagnostics=diagnostics,
    )


def fit_registry_model_b(
    bundle: RegistryFeatureBundle,
    label_map: pd.Series,
    model_a: RegistryModelAResult,
    incumbent_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    seeds = bundle.seeds.reset_index(drop=True)
    split = seeds["split"].astype(str)
    source_indices = np.flatnonzero(split.isin(["replay_2017", "official_2018", "replay_2018", "official_2019", "validation_2020"]).to_numpy())
    test_indices = np.flatnonzero((split == "test_2021").to_numpy())
    features = bundle.features_by_strength[model_a.strength]
    labels = _labels_for_indices(label_map, seeds, source_indices)
    selected_registry = features[registry_columns(features)]
    registry_test = fit_registry_predict(
        selected_registry.iloc[source_indices], labels, seeds.iloc[source_indices]["nct_id"], selected_registry.iloc[test_indices]
    )
    literature_test = fit_literature_predict(
        features.iloc[source_indices], labels, seeds.iloc[source_indices]["nct_id"],
        features.iloc[test_indices], model_a.literature_c,
    )
    evidence_test = features.iloc[test_indices]["registry_result_reference_count"].to_numpy(dtype=float) > 0
    external_test = routed_rank_blend(registry_test, literature_test, evidence_test, model_a.literature_weight)
    linked_test = bundle.linkage.set_index("row_id").loc[seeds.iloc[test_indices]["row_id"], "linked"].to_numpy(dtype=bool)
    test_prediction = routed_rank_blend(incumbent_test, external_test, linked_test, model_a.blend_weight)
    return external_test, test_prediction


def save_registry_candidate(
    destination: Path,
    model_a: RegistryModelAResult,
    external_test: np.ndarray,
    test_prediction: np.ndarray,
    linkage_reports: dict[str, Any],
) -> None:
    payload = {
        "val": model_a.validation_prediction,
        "test": test_prediction,
        "external_val": model_a.external_validation,
        "external_test": external_test,
        "external_forward_index": model_a.external_forward_index,
        "external_forward_prediction": model_a.external_forward_prediction,
        "strength": np.asarray([model_a.strength]),
        "blend_weight": np.asarray([model_a.blend_weight]),
        "literature_c": np.asarray([model_a.literature_c]),
        "literature_weight": np.asarray([model_a.literature_weight]),
        "diagnostics_json": np.asarray([json.dumps(model_a.diagnostics, sort_keys=True)]),
        "linkage_reports_json": np.asarray([json.dumps(linkage_reports, sort_keys=True)]),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(destination)
