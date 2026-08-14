# Imports

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import betaln
from sklearn.metrics import roc_auc_score


# Encoded features

@dataclass
class EncodedFeatures:
    frame: pd.DataFrame
    categorical: list[str]


def encode_base(base: pd.DataFrame) -> EncodedFeatures:
    frame = pd.DataFrame(index=base.index)
    categorical = []
    for column in base.columns:
        values = base[column]
        if values.dtype == object or str(values.dtype).startswith("string") or column.startswith("cat_"):
            frame[column] = pd.factorize(values.fillna("__missing__").astype(str), sort=True)[0].astype(np.int32)
            categorical.append(column)
        else:
            frame[column] = pd.to_numeric(values, errors="coerce").astype(np.float32)
    return EncodedFeatures(frame=frame, categorical=categorical)


# Empirical Bayes priors

PRIOR_GROUPS = [
    "lead_sponsor", "sponsor_class", "phase", "source_class", "phase_source",
    "condition", "intervention", "condition_phase", "condition_intervention",
    "country", "facility", "sponsor_condition", "sponsor_phase",
]


def temporal_eb_features(
    keys: dict[str, list[list[str]]],
    timestamps: pd.Series,
    nct_ids: pd.Series,
    reference_indices: np.ndarray,
    target_indices: np.ndarray,
    labels: np.ndarray,
    strength: float,
) -> pd.DataFrame:
    result = pd.DataFrame(index=target_indices)
    target_times = pd.to_datetime(timestamps.iloc[target_indices])
    reference_times = pd.to_datetime(timestamps.iloc[reference_indices])
    reference_labels = labels[reference_indices].astype(float)
    reference_ids = nct_ids.iloc[reference_indices].to_numpy()
    for group_name in PRIOR_GROUPS:
        key_lists = keys[group_name]
        records = []
        for index, timestamp, label, nct_id in zip(reference_indices, reference_times, reference_labels, reference_ids):
            for key in set(key_lists[int(index)]):
                records.append((str(key), timestamp, float(label), int(nct_id), int(index)))
        records_frame = pd.DataFrame(records, columns=["key", "timestamp", "label", "nct_id", "source_index"])
        group_values = {suffix: np.full(len(target_indices), np.nan, dtype=np.float32) for suffix in ["mean", "min", "max", "weighted", "uncertainty", "recency"]}
        for timestamp in sorted(target_times.unique()):
            positions = np.flatnonzero(target_times.to_numpy() == timestamp)
            absolute = target_indices[positions]
            eligible_rows = reference_times + pd.Timedelta(days=365) <= timestamp
            eligible_labels = reference_labels[eligible_rows]
            global_rate = float(eligible_labels.mean()) if len(eligible_labels) else 0.5
            if len(records_frame):
                eligible = records_frame[records_frame["timestamp"] + pd.Timedelta(days=365) <= timestamp].copy()
                target_id_set = set(nct_ids.iloc[absolute].astype(int).tolist())
                eligible = eligible[~eligible["nct_id"].isin(target_id_set)]
            else:
                eligible = records_frame
            if len(eligible):
                eligible["recency_weight"] = np.exp(-((timestamp - eligible["timestamp"]).dt.days / 365.25) / 5.0)
                summary = eligible.groupby("key").agg(count=("label", "size"), successes=("label", "sum"), weight=("recency_weight", "sum"))
                weighted_success = (eligible["label"] * eligible["recency_weight"]).groupby(eligible["key"]).sum()
                summary["rate"] = (summary["successes"] + strength * global_rate) / (summary["count"] + strength)
                summary["recency_rate"] = (weighted_success + strength * global_rate) / (summary["weight"] + strength)
                summary["uncertainty"] = np.sqrt(summary["rate"] * (1.0 - summary["rate"]) / (summary["count"] + strength))
            else:
                summary = pd.DataFrame(columns=["count", "rate", "recency_rate", "uncertainty"])
            for local_position, target_index in zip(positions, absolute):
                target_keys = list(dict.fromkeys(str(value) for value in key_lists[int(target_index)]))
                if not target_keys:
                    rates = np.array([global_rate], dtype=float)
                    counts = np.array([0.0], dtype=float)
                    uncertainties = np.array([np.sqrt(global_rate * (1 - global_rate) / strength)], dtype=float)
                    recency_rates = rates
                else:
                    selected = summary.reindex(target_keys)
                    counts = selected["count"].fillna(0).to_numpy(dtype=float)
                    rates = selected["rate"].fillna(global_rate).to_numpy(dtype=float)
                    uncertainties = selected["uncertainty"].fillna(np.sqrt(global_rate * (1 - global_rate) / strength)).to_numpy(dtype=float)
                    recency_rates = selected["recency_rate"].fillna(global_rate).to_numpy(dtype=float)
                group_values["mean"][local_position] = float(rates.mean())
                group_values["min"][local_position] = float(rates.min())
                group_values["max"][local_position] = float(rates.max())
                group_values["weighted"][local_position] = float(np.average(rates, weights=counts + 1.0))
                group_values["uncertainty"][local_position] = float(uncertainties.mean())
                group_values["recency"][local_position] = float(recency_rates.mean())
        for suffix, values in group_values.items():
            result[f"eb_{group_name}_{suffix}"] = values
    return result.reset_index(drop=True)


# Multiplicity targets

def future_multiplicity_targets(db: Any, labeled_seeds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outcomes = db.table_dict["outcomes"].df[["id", "outcome_type"]]
    analyses = db.table_dict["outcome_analyses"].df.merge(outcomes, left_on="outcome_id", right_on="id", suffixes=("", "_outcome"))
    analyses = analyses[
        analyses["outcome_type"].eq("Primary")
        & analyses["p_value"].between(0, 1, inclusive="both")
        & (analyses["p_value_modifier"].isna() | analyses["p_value_modifier"].ne(">"))
    ][["nct_id", "date", "p_value"]]
    source = labeled_seeds[["row_id", "nct_id", "timestamp"]].merge(analyses, on="nct_id", how="left")
    source = source[
        source["date"].gt(source["timestamp"])
        & source["date"].le(source["timestamp"] + pd.Timedelta(days=365))
    ]
    grouped = source.groupby("row_id")
    count = labeled_seeds["row_id"].map(grouped.size()).fillna(0).to_numpy(dtype=np.int32)
    significant_map = source.assign(significant=source["p_value"].le(0.05).astype(np.int32)).groupby("row_id")["significant"].sum()
    significant = labeled_seeds["row_id"].map(significant_map).fillna(0).to_numpy(dtype=np.int32)
    if np.any(count < 1):
        raise RuntimeError(f"Multiplicity target construction found {(count < 1).sum()} rows without a qualifying analysis")
    fraction = significant / count
    return count, significant, fraction.astype(np.float32)


# Multiplicity models

def _fit_structural_heads(
    matrix: pd.DataFrame,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    counts: np.ndarray,
    fractions: np.ndarray,
    seed: int,
    rounds: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    count_class = np.minimum(counts[train_indices], 6) - 1
    train_set = lgb.Dataset(matrix.iloc[train_indices], label=count_class, free_raw_data=False)
    count_model = lgb.train(
        {
            "objective": "multiclass", "num_class": 6, "metric": "multi_logloss",
            "learning_rate": 0.04, "num_leaves": 15, "max_depth": 5,
            "min_data_in_leaf": 80, "feature_fraction": 0.75,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5,
            "verbosity": -1, "seed": seed, "num_threads": 22,
        },
        train_set,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    fraction_set = lgb.Dataset(
        matrix.iloc[train_indices], label=fractions[train_indices],
        weight=counts[train_indices].astype(float), free_raw_data=False,
    )
    fraction_model = lgb.train(
        {
            "objective": "cross_entropy", "metric": "cross_entropy",
            "learning_rate": 0.04, "num_leaves": 15, "max_depth": 5,
            "min_data_in_leaf": 80, "feature_fraction": 0.75,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 5,
            "verbosity": -1, "seed": seed, "num_threads": 22,
        },
        fraction_set,
        num_boost_round=rounds,
        callbacks=[lgb.log_evaluation(0)],
    )
    count_probability = np.asarray(count_model.predict(matrix.iloc[predict_indices]))
    fraction_probability = np.asarray(fraction_model.predict(matrix.iloc[predict_indices])).clip(1e-4, 1 - 1e-4)
    large = counts[train_indices][counts[train_indices] >= 6]
    representative_large = float(np.clip(large.mean() if len(large) else 6.0, 6.0, 30.0))
    return count_probability, fraction_probability, representative_large


def structural_probability(
    count_probability: np.ndarray,
    fraction_probability: np.ndarray,
    representative_large: np.ndarray | float,
    phi: float,
) -> np.ndarray:
    k_values = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    if np.ndim(representative_large) == 0:
        k_values[-1] = float(representative_large)
        k_matrix = np.tile(k_values, (len(fraction_probability), 1))
    else:
        k_matrix = np.tile(k_values, (len(fraction_probability), 1))
        k_matrix[:, -1] = np.asarray(representative_large)
    q = fraction_probability[:, None].clip(1e-5, 1 - 1e-5)
    alpha = q * phi
    beta = (1.0 - q) * phi
    zero = np.exp(betaln(alpha, beta + k_matrix) - betaln(alpha, beta))
    return (1.0 - (count_probability * zero).sum(axis=1)).clip(1e-5, 1 - 1e-5)


def structural_temporal_oof(
    matrix: pd.DataFrame,
    timestamps: pd.Series,
    counts: np.ndarray,
    fractions: np.ndarray,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = len(matrix)
    count_predictions = np.full((size, 6), 1.0 / 6.0, dtype=np.float32)
    fraction_predictions = np.full(size, 0.35, dtype=np.float32)
    large_values = np.full(size, 6.0, dtype=np.float32)
    unique_times = sorted(pd.to_datetime(timestamps).unique())
    for timestamp in unique_times:
        predict_indices = np.flatnonzero(pd.to_datetime(timestamps).to_numpy() == timestamp)
        train_indices = np.flatnonzero((pd.to_datetime(timestamps) + pd.Timedelta(days=365) <= timestamp).to_numpy())
        if len(train_indices) < (80 if debug else 250) or len(np.unique(np.minimum(counts[train_indices], 6))) < 2:
            continue
        probability, fraction, large = _fit_structural_heads(
            matrix, train_indices, predict_indices, counts, fractions, 17, 60 if debug else 180
        )
        count_predictions[predict_indices] = probability
        fraction_predictions[predict_indices] = fraction
        large_values[predict_indices] = large
    return count_predictions, fraction_predictions, large_values


def select_phi(
    count_probability: np.ndarray,
    fraction_probability: np.ndarray,
    large_values: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, dict[str, float]]:
    scores = {}
    for phi in [5.0, 20.0, 1000.0]:
        probability = structural_probability(count_probability, fraction_probability, large_values, phi)
        scores[str(int(phi))] = float(roc_auc_score(labels[mask], probability[mask]))
    best = max([5.0, 20.0, 1000.0], key=lambda value: scores[str(int(value))])
    return best, scores


def structural_inference(
    matrix: pd.DataFrame,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    counts: np.ndarray,
    fractions: np.ndarray,
    phi: float,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability, fraction, large = _fit_structural_heads(
        matrix, train_indices, predict_indices, counts, fractions, 17, 80 if debug else 240
    )
    structural = structural_probability(probability, fraction, large, phi)
    expected_values = np.array([1, 2, 3, 4, 5, large], dtype=float)
    expected_count = probability @ expected_values
    return structural.astype(np.float32), fraction.astype(np.float32), expected_count.astype(np.float32)
