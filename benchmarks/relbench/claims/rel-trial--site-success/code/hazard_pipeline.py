# Imports

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score

from trial_pipeline import (
    CATEGORICAL_COLUMNS,
    _candidate_truth,
    _paired_component_deltas,
    _pair_rows_for_year,
    build_trial_features,
    decode_edges,
    exact_ratio_decode,
    facility_predictions,
    fit_facility_model,
    fit_heads,
    make_roster_edges,
    monte_carlo_ratio_decode,
)


# Configuration

VERSION = "monthly_hazard_conditional_v2"
HAZARD_BINS = 12
HAZARD_ROUNDS = 800
ROUND_GRID = [400, 600, 800]
RIDGE_GRID = [1.0, 10.0, 100.0]
BOOTSTRAP_DRAWS = 2000
MC_DRAWS = 8192
Z_COLUMNS = [
    "expected_month",
    "q_rank",
    "q_share",
    "snapshot_recency",
    "trial_age",
    "roster_size",
]


# Structures

@dataclass
class HazardFit:
    model: Any
    feature_columns: list[str]
    categorical: list[str]


@dataclass
class ConditionalCalibrator:
    intercept: float
    temperature: float
    coefficients: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    ridge: float
    bounds: dict[str, list[float]]


@dataclass
class FoldOutput:
    year: int
    pair_rows: np.ndarray
    reporting: np.ndarray
    success: np.ndarray
    hazard_q: np.ndarray
    hazard_expected_month: np.ndarray
    hazard_front_mass: np.ndarray
    hazard_back_mass: np.ndarray
    hazard_shape: np.ndarray
    binary_q: np.ndarray
    raw_tabular: np.ndarray
    raw_text: np.ndarray


# Rolling origin

def ensure_rolling_origin(context: Any, bundle: Any, cache: Path, debug: bool) -> dict[str, Any]:
    root = cache / "hazard_lane1" / VERSION / ("debug" if debug else "full") / "origin_2017"
    edge_path = root / "edges.parquet"
    pair_path = root / "pairs.parquet"
    feature_path = root / "features.parquet"
    document_path = root / "documents.jsonl"
    metadata_path = root / "metadata.json"
    if all(path.exists() for path in [edge_path, pair_path, feature_path, document_path, metadata_path]):
        edge = pd.read_parquet(edge_path)
        pairs = pd.read_parquet(pair_path)
        features = pd.read_parquet(feature_path)
        documents = []
        contexts = []
        with document_path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                documents.append(record["document"])
                contexts.append(record["context"])
        state = "hit"
    else:
        seeds = context.train.df[context.train.df["timestamp"].dt.year.eq(2017)][["timestamp", "facility_id"]].reset_index(drop=True)
        if debug:
            seeds = seeds.head(600).copy()
        edge = make_roster_edges(context.db, seeds)
        candidates = edge[["timestamp", "nct_id"]].drop_duplicates().reset_index(drop=True)
        existing = pd.MultiIndex.from_frame(bundle.pairs[["timestamp", "nct_id"]])
        candidate_index = pd.MultiIndex.from_frame(candidates)
        pairs = candidates.loc[~candidate_index.isin(existing)].reset_index(drop=True)
        pairs["pair_row"] = np.arange(len(bundle.pairs), len(bundle.pairs) + len(pairs), dtype=np.int64)
        features, documents, contexts, coverage = build_trial_features(context.db, pairs, bundle.event_info)
        root.mkdir(parents=True, exist_ok=True)
        edge.to_parquet(edge_path, index=False)
        pairs.to_parquet(pair_path, index=False)
        features.to_parquet(feature_path, index=False)
        with document_path.open("w", encoding="utf-8") as stream:
            for document, current_context in zip(documents, contexts):
                stream.write(json.dumps({"document": document, "context": current_context}, ensure_ascii=False) + "\n")
        metadata_path.write_text(json.dumps({"rows": len(pairs), "edges": len(edge), "coverage": coverage}, indent=2) + "\n")
        state = "built"
    if len(pairs):
        existing = pd.MultiIndex.from_frame(bundle.pairs[["timestamp", "nct_id"]])
        new_mask = ~pd.MultiIndex.from_frame(pairs[["timestamp", "nct_id"]]).isin(existing)
        pairs = pairs.loc[new_mask].reset_index(drop=True)
        features = features.loc[new_mask].reset_index(drop=True)
        selected_documents = [document for document, keep in zip(documents, new_mask) if keep]
        selected_contexts = [current for current, keep in zip(contexts, new_mask) if keep]
        if len(pairs):
            pairs["pair_row"] = np.arange(len(bundle.pairs), len(bundle.pairs) + len(pairs), dtype=np.int64)
            features = features.reindex(columns=bundle.features.columns)
            bundle.pairs = pd.concat([bundle.pairs, pairs], ignore_index=True)
            bundle.features = pd.concat([bundle.features, features], ignore_index=True)
            bundle.documents.extend(selected_documents)
            bundle.contexts.extend(selected_contexts)
    bundle.edges[2017] = edge
    for column in CATEGORICAL_COLUMNS:
        if column in bundle.features:
            bundle.features[column] = bundle.features[column].astype("object").fillna("__missing__").astype("category")
    return {
        "state": state,
        "extension_rows": int(len(pairs)),
        "edge_rows": int(len(edge)),
        "facility_rows": int(edge["seed_row"].nunique()) if len(edge) else 0,
    }


# Person periods

def _person_period_path(cache: Path, debug: bool) -> Path:
    return cache / "hazard_lane1" / VERSION / ("debug" if debug else "full") / "person_periods.npz"


def build_person_periods(bundle: Any, cache: Path, debug: bool) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = _person_period_path(cache, debug)
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        arrays = {name: loaded[name] for name in loaded.files}
        return arrays, {"state": "hit", "rows": int(len(arrays["replay_position"]))}
    started = time.time()
    replay = bundle.replay.reset_index(drop=True)
    if debug and len(replay) > 5000:
        replay = replay.groupby(["timestamp", "report_label"], group_keys=False).head(2500).sort_index()
    event_date = replay["nct_id"].map(bundle.event_info.drop_duplicates("nct_id").set_index("nct_id")["date"])
    report_month = np.ceil((event_date - replay["timestamp"]).dt.days.to_numpy(dtype=np.float64) / (365.0 / HAZARD_BINS))
    report_month = np.nan_to_num(report_month, nan=HAZARD_BINS).clip(1, HAZARD_BINS).astype(np.int8)
    maximum_bin = np.where(replay["report_label"].to_numpy(dtype=np.int8) == 1, report_month, HAZARD_BINS)
    if debug:
        maximum_bin = np.minimum(maximum_bin, 3)
    replay_position = np.repeat(replay.index.to_numpy(dtype=np.int32), maximum_bin)
    pair_row = np.repeat(replay["pair_row"].to_numpy(dtype=np.int32), maximum_bin)
    bins = np.concatenate([np.arange(1, int(count) + 1, dtype=np.int8) for count in maximum_bin])
    label = ((bins == np.repeat(report_month, maximum_bin)) & (np.repeat(replay["report_label"].to_numpy(dtype=np.int8), maximum_bin) == 1)).astype(np.int8)
    base_weight = replay["sampling_weight"].to_numpy(dtype=np.float64) / maximum_bin
    origin_total = pd.Series(replay["sampling_weight"].to_numpy(dtype=np.float64)).groupby(replay["timestamp"].reset_index(drop=True)).transform("sum").to_numpy(dtype=np.float64)
    weight = np.repeat(base_weight / np.maximum(origin_total, 1e-12), maximum_bin).astype(np.float32)
    origin_ns = np.repeat(replay["timestamp"].astype("int64").to_numpy(), maximum_bin)
    arrays = {
        "replay_position": replay_position,
        "pair_row": pair_row,
        "bin": bins,
        "label": label,
        "weight": weight,
        "origin_ns": origin_ns,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    elapsed = max(time.time() - started, 1e-9)
    return arrays, {"state": "built", "rows": int(len(pair_row)), "rows_per_minute": float(len(pair_row) / elapsed * 60.0)}


def _hazard_matrix(bundle: Any, pair_rows: np.ndarray, bins: np.ndarray) -> pd.DataFrame:
    matrix = bundle.features.iloc[pair_rows].reset_index(drop=True).copy()
    phase = bins.astype(np.float32) / HAZARD_BINS
    matrix["hazard_bin"] = bins.astype(np.float32)
    matrix["hazard_bin_fraction"] = phase
    matrix["hazard_bin_sin"] = np.sin(2.0 * np.pi * phase).astype(np.float32)
    matrix["hazard_bin_cos"] = np.cos(2.0 * np.pi * phase).astype(np.float32)
    if "trial_age_days" in matrix:
        matrix["hazard_trial_age_days"] = matrix["trial_age_days"].to_numpy(dtype=np.float32) + bins.astype(np.float32) * (365.0 / HAZARD_BINS)
    return matrix


def fit_hazard(bundle: Any, person: dict[str, np.ndarray], cutoff: pd.Timestamp, debug: bool) -> HazardFit:
    cutoff_ns = int((cutoff - pd.Timedelta(days=365)).value)
    selected = np.flatnonzero(person["origin_ns"] <= cutoff_ns)
    matrix = _hazard_matrix(bundle, person["pair_row"][selected], person["bin"][selected])
    labels = person["label"][selected]
    weights = person["weight"][selected]
    categorical = [column for column in CATEGORICAL_COLUMNS if column in matrix]
    dataset = lgb.Dataset(matrix, label=labels, weight=weights, categorical_feature=categorical, free_raw_data=False)
    parameters = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "lambda_l2": 10.0,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 31337,
        "num_threads": 22,
        "force_col_wise": True,
    }
    rounds = 60 if debug else HAZARD_ROUNDS
    model = lgb.train(parameters, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
    return HazardFit(model, list(matrix.columns), categorical)


def predict_hazard(head: HazardFit, bundle: Any, pair_rows: np.ndarray, rounds: int | None = None, debug: bool = False) -> dict[str, np.ndarray]:
    bins_count = 3 if debug else HAZARD_BINS
    repeated = np.repeat(pair_rows.astype(np.int64), bins_count)
    bins = np.tile(np.arange(1, bins_count + 1, dtype=np.int8), len(pair_rows))
    matrix = _hazard_matrix(bundle, repeated, bins)
    hazards = head.model.predict(matrix, num_iteration=rounds).reshape(len(pair_rows), bins_count)
    hazards = np.clip(hazards, 1e-7, 1.0 - 1e-7)
    survival_before = np.concatenate([np.ones((len(pair_rows), 1)), np.cumprod(1.0 - hazards[:, :-1], axis=1)], axis=1)
    mass = survival_before * hazards
    q = mass.sum(axis=1)
    month = np.arange(1, bins_count + 1, dtype=np.float64)
    expected = (mass * month).sum(axis=1) / np.maximum(q, 1e-12)
    split = min(6, bins_count)
    return {
        "q": np.clip(q, 1e-7, 1.0 - 1e-7),
        "expected_month": expected,
        "front_mass": mass[:, :split].sum(axis=1),
        "back_mass": mass[:, split:].sum(axis=1),
        "shape": hazards,
    }


# Fold cache

def _fold_cache_path(cache: Path, year: int, debug: bool) -> Path:
    return cache / "hazard_lane1" / VERSION / ("debug" if debug else "full") / f"fold_{year}.npz"


def build_fold(bundle: Any, person: dict[str, np.ndarray], cache: Path, year: int, debug: bool) -> FoldOutput:
    path = _fold_cache_path(cache, year, debug)
    if path.exists():
        loaded = np.load(path, allow_pickle=False)
        return FoldOutput(**{name: loaded[name] for name in FoldOutput.__dataclass_fields__})
    origin = pd.Timestamp(f"{year}-01-01")
    pair_rows = _pair_rows_for_year(bundle, year)
    reporting, success = _candidate_truth(bundle, pair_rows, origin)
    hazard = fit_hazard(bundle, person, origin, debug)
    rounds_grid = [60] if debug else ROUND_GRID
    predictions = [predict_hazard(hazard, bundle, pair_rows, rounds, debug) for rounds in rounds_grid]
    flat = fit_heads(bundle, origin, "equal", debug)
    matrix = bundle.features.iloc[pair_rows]
    binary_q = np.clip(flat.reporting.predict(matrix), 1e-7, 1.0 - 1e-7)
    raw_tabular = np.clip(flat.success.predict(matrix), 1e-7, 1.0 - 1e-7)
    documents = [bundle.documents[index] for index in pair_rows]
    raw_text = np.clip(flat.text_model.predict_proba(flat.text_vectorizer.transform(documents))[:, 1], 1e-7, 1.0 - 1e-7)
    output = FoldOutput(
        year=np.asarray(year),
        pair_rows=pair_rows,
        reporting=reporting,
        success=success,
        hazard_q=np.stack([item["q"] for item in predictions]),
        hazard_expected_month=np.stack([item["expected_month"] for item in predictions]),
        hazard_front_mass=np.stack([item["front_mass"] for item in predictions]),
        hazard_back_mass=np.stack([item["back_mass"] for item in predictions]),
        hazard_shape=np.stack([item["shape"] for item in predictions]),
        binary_q=binary_q,
        raw_tabular=raw_tabular,
        raw_text=raw_text,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{name: getattr(output, name) for name in FoldOutput.__dataclass_fields__})
    return output


# Calibration

def _fit_platt(values: np.ndarray, labels: np.ndarray) -> Any | None:
    if len(np.unique(labels)) < 2:
        return None
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, max_iter=500)
    model.fit(logits, labels)
    return model


def _apply_platt(values: np.ndarray, model: Any | None) -> np.ndarray:
    values = np.clip(values, 1e-6, 1.0 - 1e-6)
    if model is None:
        return values
    logits = np.log(values / (1.0 - values)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1].clip(1e-6, 1.0 - 1e-6)


def _success_probabilities(bundle: Any, fold: FoldOutput, prior: list[FoldOutput]) -> np.ndarray:
    prior_values = []
    prior_labels = []
    for item in prior:
        raw = 0.75 * item.raw_tabular + 0.25 * item.raw_text
        mask = item.reporting.astype(bool) & np.isfinite(item.success)
        prior_values.append(raw[mask])
        prior_labels.append(item.success[mask].astype(np.int8))
    model = _fit_platt(np.concatenate(prior_values), np.concatenate(prior_labels)) if prior_values else None
    raw = 0.75 * fold.raw_tabular + 0.25 * fold.raw_text
    probability = _apply_platt(raw, model)
    if getattr(bundle, "direct_poles", False) and "registry_expert_probability" in bundle.features:
        expert = pd.to_numeric(bundle.features.iloc[fold.pair_rows]["registry_expert_probability"], errors="coerce").to_numpy(dtype=np.float64)
        covered = np.isfinite(expert)
        probability[covered] = expert[covered]
    return probability


def _binary_probabilities(fold: FoldOutput, prior: list[FoldOutput]) -> np.ndarray:
    model = _fit_platt(
        np.concatenate([item.binary_q for item in prior]),
        np.concatenate([item.reporting for item in prior]),
    ) if prior else None
    return _apply_platt(fold.binary_q, model)


# Conditional roster model

def roster_frame(bundle: Any, fold: FoldOutput, q: np.ndarray, expected_month: np.ndarray) -> pd.DataFrame:
    edge = bundle.edges[int(fold.year)].copy()
    lookup = {
        (pd.Timestamp(row.timestamp), int(row.nct_id)): int(row.pair_row)
        for row in bundle.pairs.iloc[fold.pair_rows].itertuples()
    }
    q_map = dict(zip(fold.pair_rows, q))
    month_map = dict(zip(fold.pair_rows, expected_month))
    truth_map = dict(zip(fold.pair_rows, fold.reporting))
    edge["pair_row"] = [lookup.get((pd.Timestamp(timestamp), int(nct)), -1) for timestamp, nct in zip(edge["timestamp"], edge["nct_id"])]
    edge = edge[edge["pair_row"].ge(0)].copy()
    edge["roster_key"] = int(fold.year) * 1_000_000 + edge["seed_row"].to_numpy(dtype=np.int64)
    edge["q"] = edge["pair_row"].map(q_map)
    edge["expected_month"] = edge["pair_row"].map(month_map)
    edge["y"] = edge["pair_row"].map(truth_map).fillna(0).astype(np.int8)
    edge["q_rank"] = edge.groupby("seed_row")["q"].rank(pct=True)
    q_sum = edge.groupby("seed_row")["q"].transform("sum").clip(lower=1e-12)
    edge["q_share"] = edge["q"] / q_sum
    pair_features = bundle.features.iloc[edge["pair_row"].to_numpy(dtype=np.int64)]
    if "registry_days_since_verification_date" in pair_features:
        edge["snapshot_recency"] = pd.to_numeric(pair_features["registry_days_since_verification_date"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        edge["snapshot_recency"] = np.nan
    edge["trial_age"] = pd.to_numeric(pair_features["trial_age_days"], errors="coerce").to_numpy(dtype=np.float64)
    edge["roster_size"] = np.log1p(edge.groupby("seed_row")["pair_row"].transform("size")).to_numpy(dtype=np.float64)
    return edge.reset_index(drop=True)


def _conditional_objective(parameters: np.ndarray, prepared: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int], ridge: float) -> tuple[float, np.ndarray]:
    intercept = parameters[0]
    temperature = math.exp(parameters[1])
    coefficients = parameters[2:]
    logits, z, y, codes, group_count = prepared
    adjusted = (logits + z @ coefficients) / temperature + intercept
    probability = np.clip(1.0 / (1.0 + np.exp(-np.clip(adjusted, -30.0, 30.0))), 1e-9, 1.0 - 1e-9)
    edge_log = y * np.log(probability) + (1.0 - y) * np.log1p(-probability)
    group_log = np.bincount(codes, weights=edge_log, minlength=group_count)
    absent_log = np.bincount(codes, weights=np.log1p(-probability), minlength=group_count)
    absence = np.exp(absent_log)
    presence = np.clip(1.0 - absence, 1e-12, 1.0)
    presence_log = np.log(presence)
    likelihood = group_log - presence_log
    penalty = ridge * float(coefficients @ coefficients) / max(group_count, 1)
    edge_gradient = (probability - y + probability * (absence / presence)[codes]) / max(group_count, 1)
    gradient = np.empty_like(parameters)
    gradient[0] = edge_gradient.sum()
    gradient[1] = np.dot(edge_gradient, -(adjusted - intercept))
    gradient[2:] = z.T @ edge_gradient / temperature + 2.0 * ridge * coefficients / max(group_count, 1)
    return float(-likelihood.mean() + penalty), gradient


def fit_conditional_calibrator(frames: list[pd.DataFrame], ridge: float) -> ConditionalCalibrator:
    frame = pd.concat(frames, ignore_index=True)
    positive = frame.groupby("roster_key")["y"].transform("sum").gt(0)
    frame = frame[positive].copy()
    z = frame[Z_COLUMNS].to_numpy(dtype=np.float64)
    center = np.nanmedian(z, axis=0)
    scale = np.nanmedian(np.abs(z - center), axis=0) * 1.4826
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
    q = frame["q"].to_numpy(dtype=np.float64)
    logits = np.log(np.clip(q, 1e-7, 1.0 - 1e-7) / np.clip(1.0 - q, 1e-7, 1.0))
    standardized = np.nan_to_num((z - center) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    y = frame["y"].to_numpy(dtype=np.float64)
    codes, groups = pd.factorize(frame["roster_key"], sort=False)
    prepared = (logits, standardized, y, codes, len(groups))
    initial = np.zeros(2 + len(Z_COLUMNS), dtype=np.float64)
    ranges = [
        (2.0, 0.4, 3.0, 3.0),
        (4.0, 0.2, 6.0, 5.0),
        (8.0, 0.1, 12.0, 8.0),
        (12.0, 0.05, 20.0, 12.0),
        (None, 0.01, 100.0, 30.0),
    ]
    for intercept_limit, temperature_lower, temperature_upper, coefficient_limit in ranges:
        intercept_bounds = (None, None) if intercept_limit is None else (-intercept_limit, intercept_limit)
        bounds = [
            intercept_bounds,
            (math.log(temperature_lower), math.log(temperature_upper)),
        ] + [(-coefficient_limit, coefficient_limit)] * len(Z_COLUMNS)
        result = minimize(_conditional_objective, initial, args=(prepared, ridge), method="L-BFGS-B", jac=True, bounds=bounds)
        intercept = float(result.x[0])
        temperature = float(math.exp(result.x[1]))
        intercept_boundary = intercept_limit is not None and abs(abs(intercept) - intercept_limit) <= 0.02
        temperature_boundary = temperature <= temperature_lower + 0.01 or temperature >= temperature_upper - 0.02
        if not intercept_boundary and not temperature_boundary:
            break
        initial = result.x
    return ConditionalCalibrator(
        intercept,
        temperature,
        result.x[2:].astype(np.float64),
        center,
        scale,
        ridge,
        {
            "intercept": [float(bounds[0][0]), float(bounds[0][1])] if bounds[0][0] is not None else [None, None],
            "temperature": [float(math.exp(bounds[1][0])), float(math.exp(bounds[1][1]))],
        },
    )


def apply_conditional_calibrator(frame: pd.DataFrame, calibrator: ConditionalCalibrator) -> pd.DataFrame:
    result = frame.copy()
    q = result["q"].to_numpy(dtype=np.float64)
    logits = np.log(np.clip(q, 1e-7, 1.0 - 1e-7) / np.clip(1.0 - q, 1e-7, 1.0))
    z = result[Z_COLUMNS].to_numpy(dtype=np.float64)
    z = np.nan_to_num((z - calibrator.center) / calibrator.scale, nan=0.0, posinf=0.0, neginf=0.0)
    adjusted = (logits + z @ calibrator.coefficients) / calibrator.temperature + calibrator.intercept
    result["q_prime"] = np.clip(1.0 / (1.0 + np.exp(-np.clip(adjusted, -30.0, 30.0))), 1e-7, 1.0 - 1e-7)
    return result


# Edge decoder

def decode_roster_frame(frame: pd.DataFrame, p_by_pair: dict[int, float], fallback: np.ndarray, year: int) -> dict[str, np.ndarray]:
    size = len(fallback)
    median = fallback.astype(np.float64).copy()
    mean = fallback.astype(np.float64).copy()
    presence = np.zeros(size, dtype=np.float64)
    roster_count = np.zeros(size, dtype=np.int32)
    expected_count = np.zeros(size, dtype=np.float64)
    dominant_share = np.zeros(size, dtype=np.float64)
    max_q = np.zeros(size, dtype=np.float64)
    entropy = np.zeros(size, dtype=np.float64)
    expected_month = np.zeros(size, dtype=np.float64)
    current = frame.copy()
    current["p"] = current["pair_row"].map(p_by_pair)
    current = current[current["q_prime"].notna() & current["p"].notna()]
    for seed_row, group in current.groupby("seed_row", sort=False):
        q = group["q_prime"].to_numpy(dtype=np.float64)
        p = group["p"].to_numpy(dtype=np.float64)
        order = np.argsort(q)[::-1]
        q = q[order]
        p = p[order]
        if len(q) <= 32:
            decoded = exact_ratio_decode(q, p)
        else:
            decoded = monte_carlo_ratio_decode(q, p, 31337 + int(seed_row) + 100000 * year, draws=MC_DRAWS)
        index = int(seed_row)
        median[index], mean[index], presence[index] = decoded
        roster_count[index] = len(q)
        expected_count[index] = q.sum()
        dominant_share[index] = q.max() / max(q.sum(), 1e-12)
        max_q[index] = q.max()
        normalized = q / max(q.sum(), 1e-12)
        entropy[index] = float(-(normalized * np.log(np.clip(normalized, 1e-12, 1.0))).sum())
        expected_month[index] = float(np.average(group["expected_month"], weights=np.maximum(group["q_prime"], 1e-12)))
    return {
        "median": np.clip(median, 0.0, 1.0),
        "mean": np.clip(mean, 0.0, 1.0),
        "presence": presence,
        "roster_count": roster_count,
        "expected_count": expected_count,
        "dominant_share": dominant_share,
        "max_q": max_q,
        "entropy": entropy,
        "expected_month": expected_month,
    }


def large_roster_audit(frame: pd.DataFrame, p_by_pair: dict[int, float], year: int) -> dict[str, Any]:
    exact_errors = []
    median_matches = []
    groups = [(seed_row, group) for seed_row, group in frame.groupby("seed_row", sort=False) if 20 <= len(group) <= 32][:64]
    for seed_row, group in groups:
        q = group["q_prime"].to_numpy(dtype=np.float64)
        p = group["pair_row"].map(p_by_pair).to_numpy(dtype=np.float64)
        exact = exact_ratio_decode(q, p)
        sampled = monte_carlo_ratio_decode(q, p, 991 + int(seed_row) + year * 100000, draws=MC_DRAWS)
        exact_errors.append(abs(exact[1] - sampled[1]))
        median_matches.append(exact[0] == sampled[0])
    return {
        "rosters": len(groups),
        "draws": MC_DRAWS,
        "mean_absolute_mean_error": float(np.mean(exact_errors)) if exact_errors else None,
        "median_match_rate": float(np.mean(median_matches)) if median_matches else None,
    }


# Facility utilities

def _facility_fold(context: Any, seeds: pd.DataFrame, features: pd.DataFrame, year: int, debug: bool) -> tuple[np.ndarray, np.ndarray]:
    origin = pd.Timestamp(f"{year}-01-01")
    model = fit_facility_model(seeds, features, origin, debug)
    indices, fallback = facility_predictions(model, seeds, features, "train", year)
    labels = seeds.iloc[indices]["success_rate"].to_numpy(dtype=np.float64)
    return fallback, labels


def _decode_global(bundle: Any, fold: FoldOutput, q: np.ndarray, p: np.ndarray, fallback: np.ndarray) -> dict[str, np.ndarray]:
    q_map = dict(zip(fold.pair_rows, q))
    p_map = dict(zip(fold.pair_rows, p))
    lookup = {
        (pd.Timestamp(row.timestamp), int(row.nct_id)): int(row.pair_row)
        for row in bundle.pairs.iloc[fold.pair_rows].itertuples()
    }
    return decode_edges(bundle.edges[int(fold.year)], q_map, p_map, lookup, fallback, int(fold.year))


def _snapshot_completeness(frame: pd.DataFrame, size: int) -> np.ndarray:
    values = np.zeros(size, dtype=np.float64)
    if "snapshot_recency" not in frame:
        return values
    complete = frame["snapshot_recency"].notna().groupby(frame["seed_row"]).mean()
    indices = complete.index.to_numpy(dtype=np.int64)
    values[indices] = complete.to_numpy(dtype=np.float64)
    return values


def gate_slices(labels: np.ndarray, prediction: np.ndarray, seed_features: pd.DataFrame, decoded: dict[str, np.ndarray], frame: pd.DataFrame) -> dict[str, Any]:
    specifications: dict[str, pd.Series] = {}
    history = seed_features["history_report_count"].fillna(0)
    specifications["history_depth"] = pd.cut(history, [-1, 0, 5, np.inf], labels=["zero", "sparse", "rich"])
    specifications["roster_size"] = pd.cut(decoded["roster_count"], [-1, 0, 1, 2, np.inf], labels=["zero", "one", "two", "three_plus"])
    specifications["roster_probability"] = pd.cut(decoded["presence"], [-1, 0.35, 0.75, 0.95, 1.0], labels=["low", "medium", "high", "very_high"], include_lowest=True)
    frequency = seed_features.get("country_frequency", pd.Series(np.zeros(len(seed_features))))
    specifications["country_frequency"] = pd.qcut(frequency.rank(method="first"), 4, labels=["rare", "uncommon", "common", "very_common"])
    completeness = _snapshot_completeness(frame, len(labels))
    specifications["snapshot_completeness"] = pd.cut(completeness, [-1, 0, 0.5, 0.999, 1.0], labels=["none", "partial", "mostly", "complete"])
    result = {}
    for axis, values in specifications.items():
        values = pd.Series(values).reset_index(drop=True)
        for value in values.dropna().unique():
            mask = values.eq(value).to_numpy()
            result[f"{axis}:{value}"] = {
                "count": int(mask.sum()),
                "label_mean": float(labels[mask].mean()),
                "mae": float(mean_absolute_error(labels[mask], prediction[mask])),
            }
    return result


# Meta median

def _meta_matrix(decoded: dict[str, np.ndarray], facility_features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "history_report_count",
        "history_success_rate_eb",
        "history_event_recency",
        "history_last_year_count",
        "history_last_three_year_count",
        "candidate_count",
        "condition_neighbor_rate_mean",
        "sponsor_neighbor_rate_mean",
        "country_frequency",
    ]
    matrix = facility_features[[column for column in columns if column in facility_features]].reset_index(drop=True).copy()
    for column in ["median", "mean", "expected_count", "max_q", "expected_month", "entropy", "roster_count", "presence", "dominant_share"]:
        if column in decoded:
            values = decoded[column]
        elif column == "max_q":
            values = decoded["dominant_share"] * decoded["expected_count"]
        elif column == "expected_month":
            values = np.full(len(matrix), 6.5, dtype=np.float64)
        elif column == "entropy":
            values = np.log1p(decoded["roster_count"])
        else:
            values = np.zeros(len(matrix), dtype=np.float64)
        matrix[f"roster_{column}"] = values
    return matrix


def _fit_meta(matrix: pd.DataFrame, labels: np.ndarray, debug: bool) -> Any:
    dataset = lgb.Dataset(matrix, label=labels, free_raw_data=False)
    parameters = {
        "objective": "quantile",
        "alpha": 0.5,
        "metric": "l1",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 200,
        "lambda_l2": 10.0,
        "feature_fraction": 0.8,
        "verbosity": -1,
        "seed": 8087,
        "num_threads": 22,
        "force_col_wise": True,
    }
    return lgb.train(parameters, dataset, num_boost_round=60 if debug else 500, callbacks=[lgb.log_evaluation(0)])


# System gate

def run_reporting_gates(bundle: Any, context: Any, facility_seeds: pd.DataFrame, facility_features: pd.DataFrame, cache: Path, debug: bool) -> tuple[dict[str, Any], dict[int, Any], list[FoldOutput], ConditionalCalibrator]:
    person, person_diagnostic = build_person_periods(bundle, cache, debug)
    years = [2017, 2018, 2019]
    folds = [build_fold(bundle, person, cache, year, debug) for year in years]
    round_grid = [60] if debug else ROUND_GRID
    round_records = {}
    for round_index, rounds in enumerate(round_grid):
        values = []
        for fold in folds[:2]:
            values.append(log_loss(fold.reporting, fold.hazard_q[round_index]))
        round_records[str(rounds)] = float(np.mean(values))
    selected_round = min(round_records, key=round_records.get)
    round_index = round_grid.index(int(selected_round))
    ridge_records = {}
    fold_products: dict[int, Any] = {}
    prior: list[FoldOutput] = []
    roster_frames = {}
    for fold in folds:
        p = _success_probabilities(bundle, fold, prior)
        binary_q = _binary_probabilities(fold, prior)
        q = fold.hazard_q[round_index]
        expected = fold.hazard_expected_month[round_index]
        fallback, labels = _facility_fold(context, facility_seeds, facility_features, int(fold.year), debug)
        raw_decoded = _decode_global(bundle, fold, q, p, fallback)
        binary_decoded = _decode_global(bundle, fold, binary_q, p, fallback)
        frame = roster_frame(bundle, fold, q, expected)
        roster_frames[int(fold.year)] = frame
        fold_products[int(fold.year)] = {
            "p": p,
            "binary_q": binary_q,
            "q": q,
            "fallback": fallback,
            "labels": labels,
            "raw_decoded": raw_decoded,
            "binary_decoded": binary_decoded,
            "frame": frame,
        }
        prior.append(fold)
    if debug:
        selected_ridge = RIDGE_GRID[0]
        calibrator_2018 = fit_conditional_calibrator([roster_frames[2017]], selected_ridge)
    else:
        for ridge in RIDGE_GRID:
            current = fit_conditional_calibrator([roster_frames[2017]], ridge)
            calibrated = apply_conditional_calibrator(roster_frames[2018], current)
            decoded = decode_roster_frame(calibrated, dict(zip(folds[1].pair_rows, fold_products[2018]["p"])), fold_products[2018]["fallback"], 2018)
            ridge_records[str(ridge)] = float(mean_absolute_error(fold_products[2018]["labels"], decoded["median"]))
        selected_ridge = float(min(ridge_records, key=ridge_records.get))
        calibrator_2018 = fit_conditional_calibrator([roster_frames[2017]], selected_ridge)
    calibrated_2018 = apply_conditional_calibrator(roster_frames[2018], calibrator_2018)
    conditional_2018 = decode_roster_frame(calibrated_2018, dict(zip(folds[1].pair_rows, fold_products[2018]["p"])), fold_products[2018]["fallback"], 2018)
    calibrator_2019 = fit_conditional_calibrator([roster_frames[2017], roster_frames[2018]], selected_ridge)
    calibrated_2019 = apply_conditional_calibrator(roster_frames[2019], calibrator_2019)
    conditional_2019 = decode_roster_frame(calibrated_2019, dict(zip(folds[2].pair_rows, fold_products[2019]["p"])), fold_products[2019]["fallback"], 2019)
    fold_products[2018]["conditional_frame"] = calibrated_2018
    fold_products[2018]["conditional_decoded"] = conditional_2018
    fold_products[2019]["conditional_frame"] = calibrated_2019
    fold_products[2019]["conditional_decoded"] = conditional_2019
    fold_mae = {}
    head_metrics = {}
    slices = {}
    for fold in folds:
        year = int(fold.year)
        product = fold_products[year]
        conditional = product.get("conditional_decoded", product["raw_decoded"])
        fold_mae[str(year)] = {
            "binary": float(mean_absolute_error(product["labels"], product["binary_decoded"]["median"])),
            "hazard_raw": float(mean_absolute_error(product["labels"], product["raw_decoded"]["median"])),
            "hazard_conditional": float(mean_absolute_error(product["labels"], conditional["median"])),
        }
        head_metrics[str(year)] = {
            "binary_auc": float(roc_auc_score(fold.reporting, product["binary_q"])),
            "binary_logloss": float(log_loss(fold.reporting, product["binary_q"])),
            "hazard_auc": float(roc_auc_score(fold.reporting, product["q"])),
            "hazard_logloss": float(log_loss(fold.reporting, product["q"])),
        }
        seed_mask = facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(year)
        slices[str(year)] = gate_slices(product["labels"], conditional["median"], facility_features.loc[seed_mask].reset_index(drop=True), conditional, product.get("conditional_frame", product["frame"]))
    gate_a_deltas = _paired_component_deltas(
        fold_products[2019]["labels"],
        fold_products[2019]["raw_decoded"]["median"],
        fold_products[2019]["binary_decoded"]["median"],
        bundle.edges[2019],
        BOOTSTRAP_DRAWS if not debug else 100,
        4319,
    )
    gate_b_deltas = _paired_component_deltas(
        fold_products[2019]["labels"],
        conditional_2019["median"],
        fold_products[2019]["raw_decoded"]["median"],
        bundle.edges[2019],
        BOOTSTRAP_DRAWS if not debug else 100,
        5319,
    )
    gate_c_deltas = _paired_component_deltas(
        fold_products[2019]["labels"],
        conditional_2019["median"],
        fold_products[2019]["binary_decoded"]["median"],
        bundle.edges[2019],
        BOOTSTRAP_DRAWS if not debug else 100,
        6319,
    )
    meta_train = pd.concat([
        _meta_matrix(fold_products[2017]["raw_decoded"], facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2017)].reset_index(drop=True)),
        _meta_matrix(conditional_2018, facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2018)].reset_index(drop=True)),
    ], ignore_index=True)
    meta_labels = np.concatenate([fold_products[2017]["labels"], fold_products[2018]["labels"]])
    meta_model = _fit_meta(meta_train, meta_labels, debug)
    meta_features_2019 = _meta_matrix(conditional_2019, facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2019)].reset_index(drop=True))
    meta_prediction = np.clip(meta_model.predict(meta_features_2019), 0.0, 1.0)
    recent_features = facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2019)].reset_index(drop=True)
    rich_mask = recent_features["history_report_count"].fillna(0).ge(5).to_numpy() & recent_features["history_event_recency"].fillna(np.inf).lt(365).to_numpy()
    routed_prediction = np.where(rich_mask, meta_prediction, conditional_2019["median"])
    meta_deltas = _paired_component_deltas(fold_products[2019]["labels"], meta_prediction, conditional_2019["median"], bundle.edges[2019], BOOTSTRAP_DRAWS if not debug else 100, 7319)
    routed_deltas = _paired_component_deltas(fold_products[2019]["labels"], routed_prediction, conditional_2019["median"], bundle.edges[2019], BOOTSTRAP_DRAWS if not debug else 100, 8319)
    meta_candidates = {
        "global": (meta_prediction, meta_deltas),
        "rich_recent_router": (routed_prediction, routed_deltas),
    }
    selected_meta = min(meta_candidates, key=lambda name: mean_absolute_error(fold_products[2019]["labels"], meta_candidates[name][0]))
    selected_meta_deltas = meta_candidates[selected_meta][1]
    use_meta = float(np.mean(np.asarray(selected_meta_deltas) < 0)) >= 0.8
    final_calibrator = fit_conditional_calibrator([roster_frames[2017], roster_frames[2018], roster_frames[2019]], selected_ridge)
    meta_train_model_a = pd.concat([
        _meta_matrix(fold_products[2017]["raw_decoded"], facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2017)].reset_index(drop=True)),
        _meta_matrix(conditional_2018, facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2018)].reset_index(drop=True)),
        _meta_matrix(conditional_2019, facility_features.loc[facility_seeds["split"].eq("train") & facility_seeds["timestamp"].dt.year.eq(2019)].reset_index(drop=True)),
    ], ignore_index=True)
    meta_labels_model_a = np.concatenate([fold_products[2017]["labels"], fold_products[2018]["labels"], fold_products[2019]["labels"]])
    diagnostics = {
        "person_periods": person_diagnostic,
        "selected_rounds": int(selected_round),
        "round_logloss_2017_2018": round_records,
        "selected_ridge": selected_ridge,
        "ridge_2018_mae": ridge_records,
        "head_metrics": head_metrics,
        "fold_mae": fold_mae,
        "gate_a": _delta_summary(gate_a_deltas),
        "gate_b": _delta_summary(gate_b_deltas),
        "gate_c": _delta_summary(gate_c_deltas),
        "conditional_parameters": calibrator_record(final_calibrator),
        "slices": slices,
        "large_roster_audit": large_roster_audit(calibrated_2019, dict(zip(folds[2].pair_rows, fold_products[2019]["p"])), 2019),
        "meta": {
            "global_mae": float(mean_absolute_error(fold_products[2019]["labels"], meta_prediction)),
            "router_mae": float(mean_absolute_error(fold_products[2019]["labels"], routed_prediction)),
            "base_mae": float(mean_absolute_error(fold_products[2019]["labels"], conditional_2019["median"])),
            "selected": selected_meta,
            "acceptance": _delta_summary(selected_meta_deltas),
            "use_meta": use_meta,
        },
    }
    state = {
        "selected_rounds": int(selected_round),
        "selected_ridge": selected_ridge,
        "use_meta": use_meta,
        "meta_mode": selected_meta,
        "roster_frames": roster_frames,
        "fold_products": fold_products,
        "person": person,
        "meta_train_matrix": meta_train_model_a,
        "meta_train_labels": meta_labels_model_a,
    }
    return diagnostics, state, folds, final_calibrator


def _delta_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "draws": int(len(array)),
        "delta_mae_mean": float(array.mean()),
        "paired_se": float(array.std(ddof=1)),
        "p_delta_mae_below_zero": float(np.mean(array < 0)),
    }


def calibrator_record(calibrator: ConditionalCalibrator) -> dict[str, Any]:
    return {
        "intercept": calibrator.intercept,
        "temperature": calibrator.temperature,
        "coefficients": dict(zip(Z_COLUMNS, calibrator.coefficients.tolist())),
        "ridge": calibrator.ridge,
        "bounds": calibrator.bounds,
    }


# Final model

def _final_fold(bundle: Any, person: dict[str, np.ndarray], year: int, rounds: int, debug: bool) -> FoldOutput:
    origin = pd.Timestamp(f"{year}-01-01")
    pair_rows = _pair_rows_for_year(bundle, year)
    reporting, success = _candidate_truth(bundle, pair_rows, origin)
    hazard_head = fit_hazard(bundle, person, origin, debug)
    hazard = predict_hazard(hazard_head, bundle, pair_rows, rounds, debug)
    flat = fit_heads(bundle, origin, "equal", debug)
    matrix = bundle.features.iloc[pair_rows]
    binary_q = np.clip(flat.reporting.predict(matrix), 1e-7, 1.0 - 1e-7)
    raw_tabular = np.clip(flat.success.predict(matrix), 1e-7, 1.0 - 1e-7)
    documents = [bundle.documents[index] for index in pair_rows]
    raw_text = np.clip(flat.text_model.predict_proba(flat.text_vectorizer.transform(documents))[:, 1], 1e-7, 1.0 - 1e-7)
    return FoldOutput(np.asarray(year), pair_rows, reporting, success, hazard["q"], hazard["expected_month"], hazard["front_mass"], hazard["back_mass"], hazard["shape"], binary_q, raw_tabular, raw_text)


def final_reporting_prediction(bundle: Any, facility_seeds: pd.DataFrame, facility_features: pd.DataFrame, context: Any, state: dict[str, Any], folds: list[FoldOutput], calibrator: ConditionalCalibrator, year: int, split: str, debug: bool) -> tuple[np.ndarray, dict[str, np.ndarray], FoldOutput, ConditionalCalibrator]:
    fold = _final_fold(bundle, state["person"], year, state["selected_rounds"], debug)
    prior_folds = folds + ([state["model_a_fold"]] if year == 2021 and "model_a_fold" in state else [])
    p = _success_probabilities(bundle, fold, prior_folds)
    mask = facility_seeds["split"].eq(split) & facility_seeds["timestamp"].dt.year.eq(year)
    indices = np.flatnonzero(mask.to_numpy())
    origin = pd.Timestamp(f"{year}-01-01")
    facility_model = fit_facility_model(facility_seeds, facility_features, origin, debug)
    _, fallback = facility_predictions(facility_model, facility_seeds, facility_features, split, year)
    frame = roster_frame(bundle, fold, fold.hazard_q, fold.hazard_expected_month)
    if year == 2021:
        previous = state.get("model_a_roster_frame")
        frames = [state["roster_frames"][value] for value in [2017, 2018, 2019]]
        if previous is not None:
            frames.append(previous)
        calibrator = fit_conditional_calibrator(frames, state["selected_ridge"])
    calibrated = apply_conditional_calibrator(frame, calibrator)
    decoded = decode_roster_frame(calibrated, dict(zip(fold.pair_rows, p)), fallback, year)
    decoded["fallback"] = fallback
    prediction = decoded["median"].copy()
    current_features = facility_features.loc[mask].reset_index(drop=True)
    current_meta = _meta_matrix(decoded, current_features)
    if state["use_meta"]:
        meta_matrix = state["meta_train_matrix"]
        meta_labels = state["meta_train_labels"]
        if year == 2021 and "model_a_meta_matrix" in state:
            meta_matrix = pd.concat([meta_matrix, state["model_a_meta_matrix"]], ignore_index=True)
            meta_labels = np.concatenate([meta_labels, state["model_a_labels"]])
        meta_model = _fit_meta(meta_matrix, meta_labels, debug)
        meta_prediction = np.clip(meta_model.predict(current_meta), 0.0, 1.0)
        if state["meta_mode"] == "rich_recent_router":
            rich = current_features["history_report_count"].fillna(0).ge(5).to_numpy()
            recent = current_features["history_event_recency"].fillna(np.inf).lt(365).to_numpy()
            prediction = np.where(rich & recent, meta_prediction, prediction)
        else:
            prediction = meta_prediction
    decoded["selected"] = prediction
    if year == 2020:
        state["model_a_roster_frame"] = frame
        state["model_a_decoded"] = decoded
        state["model_a_fold"] = fold
        state["model_a_meta_matrix"] = current_meta
        source_rows = facility_seeds.loc[mask, "source_row"].to_numpy(dtype=np.int64)
        state["model_a_labels"] = context.val.df.iloc[source_rows][context.target_col].to_numpy(dtype=np.float64)
    return np.clip(prediction, 0.0, 1.0), decoded, fold, calibrator
