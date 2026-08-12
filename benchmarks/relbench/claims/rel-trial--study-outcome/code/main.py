from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from data_pipeline import register_artifact
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir


warnings.filterwarnings("ignore")

ORIGINS = np.asarray([2014, 2015, 2016, 2017, 2018, 2019], dtype=int)
ORIGIN_COUNTS = {2014: 990, 2015: 1044, 2016: 1093, 2017: 1153, 2018: 1128, 2019: 1093}
FAMILIES = ["run_0007", "fusion", "mechanism", "neighborhood"]
BUNDLE_HASH = "3ad2801187c5397b2e8033fec9a91299409888c37b5a5008c1077b88f3ced7be"
OOF_HASH = "0373260102d8b532260060cce8f6bfbd20a429c41be6a9f01e804fe9430dc5bb"
COMPACT_HASH = "1fadafc04bee1887ea98b9aecd5735b6c672d9f12185995c4fddb70723558c8e"
RUN_0007_OOF_HASH = "2d0aad7338bf6daf2d58c6fe3c51f9b6d8924869d20c3d70c73b282b268b0d5b"
RUN_0007_HASHES = {
    "val": "d88621d7c6d981f381300eb41f0d31ad1c082e82e1063377f68a8afd3ccfa5cd",
    "test": "7f0e51d85f3fb4ade2b98cd981e4c86a6fdc08195ec6d92fcbb8e52701af9b48",
}
RUN_0033_HASHES = {
    "val": "892b8a95086ae50b49f5a23822a23671e9df25a2c7675bf4647ee0538e13d55e",
    "test": "007220675d528229c0d7f65887e9072bb536b58730da9067b5fa723b74fc2dac",
}
FUSION_OOF = {
    2014: ("fusion_65daf4667536b9e3dfbd5814.npz", "e4ed53859e1035a215b778769b57d535418912a536e927a92507e02719d835b5"),
    2015: ("fusion_079c06ea0671ee1e556bcb9b.npz", "34f97ee65b3bbbb2f72cef27a85f6221d4c52070d4b5c848982478dfed9f4304"),
    2016: ("fusion_6852acb74fcf65c937b26d34.npz", "3ee9af8482e7c7a327c42bbb7b292811aefbb1ab7fb984daedee551b9949fb51"),
    2017: ("fusion_f3cb75a39e304117dea9ae95.npz", "9e22b57c04cfe938b691e17438b83101e7d80560ce430558fc58e98196beea98"),
    2018: ("fusion_2625d359b85dee23afcfb963.npz", "89c31ff1a6c391f718f658815928356271cf202ae79219daadd7e0fc352a9e67"),
    2019: ("fusion_0a70a15c770265b7b623a396.npz", "83cae0923e36d177ed2b44b9c3224bae724bddf5141891bcb700ca53c11c15b8"),
}
FUSION_FINAL = {
    "val": ("fusion_a1c8b078d14e636c05c4362e.npz", "a772a40a721512e7955cd45d3306b5b03d892857ff5ba5e06014f5ddb21db603"),
    "test": ("fusion_57bd94448556100cfd9e4dd3.npz", "743a8cd832159717918761f8fbd878594ccb626523d84137810f9bdb41321dd6"),
}
MECHANISM_OOF_HASH = "7c8447080cd04339a0034d89a3ca740940142fcc12ef04edbec27a67e6578429"
MECHANISM_FINAL_HASH = "8720d60541817bd3342277c0644dbb462a3e564ea229f40a2a57fe1e9290fa45"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(path: Path, expected: str, name: str) -> None:
    if not path.exists():
        raise RuntimeError(f"required {name} artifact is missing: {path}")
    actual = _digest(path)
    if actual != expected:
        raise RuntimeError(f"required {name} hash changed: expected={expected} actual={actual}")


def _keys(frame: pd.DataFrame, split: str) -> np.ndarray:
    rows = frame.reset_index(drop=True)
    return np.asarray(
        [
            f"{split}:{row}:{int(entity)}:{pd.Timestamp(timestamp).isoformat()}"
            for row, (entity, timestamp) in enumerate(zip(rows["nct_id"], rows["timestamp"]))
        ],
        dtype=np.str_,
    )


def _auc(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=float)
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(predictions, dtype=float)))


def _midranks(values: np.ndarray, origins: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    origins = np.asarray(origins)
    output = np.empty(len(values), dtype=np.float64)
    for origin in np.unique(origins):
        selected = origins == origin
        output[selected] = (rankdata(values[selected], method="average") - 0.5) / selected.sum()
    return output


def _origin_auc(labels: np.ndarray, predictions: np.ndarray, origins: np.ndarray) -> np.ndarray:
    return np.asarray(
        [_auc(labels[origins == origin], predictions[origins == origin]) for origin in np.unique(origins)],
        dtype=float,
    )


def _sanitize_oof(cache_dir: Path) -> dict[str, np.ndarray]:
    source = cache_dir / "relational_neighborhood_rank16_v1" / "oof_components.npz"
    _require(source, OOF_HASH, "neighborhood OOF")
    unsafe = np.load(source, allow_pickle=True)
    payload = {
        "keys": np.asarray(unsafe["keys"].astype(str), dtype=np.str_),
        "origin_year": unsafe["origin_year"].astype(np.int16),
        "outcome": unsafe["outcome"].astype(np.float64),
        "stage_a": unsafe["stage_a"].astype(np.float64),
        "attention": unsafe["attention"].astype(np.float64),
    }
    if len(payload["keys"]) != 6501 or len(np.unique(payload["keys"])) != 6501:
        raise RuntimeError("neighborhood OOF keys are not 6,501 unique rows")
    counts = dict(zip(*np.unique(payload["origin_year"], return_counts=True)))
    if counts != ORIGIN_COUNTS:
        raise RuntimeError(f"neighborhood OOF annual counts changed: {counts}")
    target = source.with_name("oof_components_sanitized_v1.npz")
    if not target.exists():
        temporary = target.with_name(target.stem + f".{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, target)
    safe = np.load(target, allow_pickle=False)
    for name, expected in payload.items():
        equal = np.array_equal(safe[name], expected, equal_nan=True) if expected.dtype.kind in "fc" else np.array_equal(safe[name], expected)
        if not equal:
            raise RuntimeError(f"sanitized neighborhood OOF changed numeric or key array {name}")
    register_artifact(
        cache_dir,
        {
            "name": "lane1 sanitized neighborhood Stage-A OOF",
            "path": str(target.relative_to(cache_dir)),
            "description": "Unicode-key copy of immutable six-origin Stage-A OOF with unchanged labels and predictions",
            "content_key": "relational_neighborhood_rank16_v1:oof_sanitized_v1",
            "rebuild_hint": "Copy the immutable OOF arrays while casting only keys to NumPy Unicode",
        },
    )
    print(f"[recon] sanitized_oof rows={len(payload['keys'])} annual_counts={json.dumps({str(int(k)): int(v) for k, v in counts.items()}, separators=(',', ':'))}")
    return payload


def _align(keys: np.ndarray, values: np.ndarray, target_keys: np.ndarray, name: str) -> np.ndarray:
    keys = np.asarray(keys).astype(str)
    target_keys = np.asarray(target_keys).astype(str)
    if len(keys) != len(np.unique(keys)):
        raise RuntimeError(f"{name} has duplicate keys")
    mapping = dict(zip(keys, np.asarray(values, dtype=float)))
    missing = [key for key in target_keys if key not in mapping]
    extra = set(mapping).difference(target_keys)
    if missing or extra:
        raise RuntimeError(f"{name} key mismatch missing={len(missing)} extra={len(extra)}")
    aligned = np.asarray([mapping[key] for key in target_keys], dtype=np.float64)
    if not np.isfinite(aligned).all():
        raise RuntimeError(f"{name} contains nonfinite predictions")
    return aligned


def _load_oof_families(cache_dir: Path, neighborhood: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    keys = neighborhood["keys"]
    years = neighborhood["origin_year"].astype(int)
    labels = neighborhood["outcome"].astype(float)
    run_path = cache_dir / "family_ensemble_v1" / "run_0007_oof.npz"
    _require(run_path, RUN_0007_OOF_HASH, "run_0007 OOF")
    run = np.load(run_path, allow_pickle=False)
    identity = {
        (int(key.split(":")[2]), ":".join(key.split(":")[3:])): key
        for key in keys.astype(str)
    }
    run_keys = np.asarray(
        [identity[(int(entity), pd.Timestamp(timestamp).isoformat())] for entity, timestamp in zip(run["nct_id"], run["timestamp"])],
        dtype=np.str_,
    )
    run_prediction = _align(run_keys, run["prediction"], keys, "run_0007 OOF")
    if not np.array_equal(_align(run_keys, run["outcome"], keys, "run_0007 labels"), labels):
        raise RuntimeError("run_0007 OOF labels disagree")
    fusion_keys = []
    fusion_values = []
    fusion_root = cache_dir / "lane3_hierarchical_fusion"
    for origin in ORIGINS:
        filename, expected = FUSION_OOF[int(origin)]
        path = fusion_root / filename
        _require(path, expected, f"fusion OOF {origin}")
        part = np.load(path, allow_pickle=False)
        fusion_keys.extend(part["keys"].astype(str).tolist())
        fusion_values.extend(part["prediction_2"].astype(float).tolist())
    fusion_prediction = _align(np.asarray(fusion_keys), np.asarray(fusion_values), keys, "fusion OOF")
    mechanism_path = cache_dir / "lane3_multitask_fusion" / "raw_run0033_mechanistic_oof.npz"
    _require(mechanism_path, MECHANISM_OOF_HASH, "mechanism OOF")
    mechanism = np.load(mechanism_path, allow_pickle=False)
    mechanism_prediction = _align(mechanism["keys"], mechanism["prediction"], keys, "mechanism OOF")
    mechanism_labels = _align(mechanism["keys"], mechanism["outcome"], keys, "mechanism labels")
    if not np.array_equal(mechanism_labels, labels):
        raise RuntimeError("mechanism OOF labels disagree")
    raw = np.column_stack([run_prediction, fusion_prediction, mechanism_prediction, neighborhood["stage_a"]])
    ranked = np.column_stack([_midranks(raw[:, index], years) for index in range(raw.shape[1])])
    bundle = np.load(cache_dir / "relational_neighborhood_rank16_v1" / "official_bundle.npz", allow_pickle=False)
    cluster_map = dict(zip(bundle["keys"].astype(str), bundle["sponsor_clusters"].astype(str)))
    coverage = pd.read_parquet(cache_dir / "common_raw_family_stack_v1" / "exact_ks_features_623076f65f45e3f4f052c2a6.parquet")
    coverage = coverage.set_index("_key").loc[keys.astype(str)].reset_index()
    coverage["lead_sponsor_cluster"] = [cluster_map[key] for key in keys.astype(str)]
    correlations = {}
    for left in range(len(FAMILIES)):
        for right in range(left + 1, len(FAMILIES)):
            correlations[f"{FAMILIES[left]}__{FAMILIES[right]}"] = float(spearmanr(ranked[:, left], ranked[:, right]).statistic)
    print(f"[recon] aligned_oof rows={len(keys)} correlations={json.dumps(correlations, separators=(',', ':'))}")
    return labels, years, ranked, coverage


def _weight_grid() -> list[np.ndarray]:
    return [
        np.asarray(values, dtype=float) / 8.0
        for values in itertools.product(range(6), repeat=4)
        if sum(values) == 8
    ]


def _select_grid(labels: np.ndarray, matrix: np.ndarray, origins: np.ndarray, training_origins: np.ndarray) -> np.ndarray:
    candidates = []
    for weights in _weight_grid():
        prediction = matrix @ weights
        scores = np.asarray([_auc(labels[origins == year], prediction[origins == year]) for year in training_origins])
        objective = 0.75 * scores.mean() + 0.25 * np.mean(np.sort(scores)[:2])
        candidates.append((objective, -scores.std(), tuple(weights.tolist()), weights))
    return max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]


def _uncertainty(labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, origins: np.ndarray, clusters: np.ndarray, seed: int, draws: int = 600) -> dict:
    rng = np.random.default_rng(seed)
    cluster_names = np.unique(clusters.astype(str))
    groups = [np.flatnonzero(clusters.astype(str) == name) for name in cluster_names]
    sponsor_deltas = []
    for _ in range(draws):
        sampled = rng.integers(0, len(groups), len(groups))
        indices = np.concatenate([groups[index] for index in sampled])
        sponsor_deltas.append(
            np.nanmean(_origin_auc(labels[indices], candidate[indices], origins[indices]))
            - np.nanmean(_origin_auc(labels[indices], reference[indices], origins[indices]))
        )
    origin_deltas = _origin_auc(labels, candidate, origins) - _origin_auc(labels, reference, origins)
    origin_draws = [float(np.mean(origin_deltas[rng.integers(0, len(origin_deltas), len(origin_deltas))])) for _ in range(draws)]
    sponsor_se = float(np.std(sponsor_deltas, ddof=1))
    origin_se = float(np.std(origin_draws, ddof=1))
    return {
        "sponsor_se": sponsor_se,
        "origin_se": origin_se,
        "larger_paired_se": max(sponsor_se, origin_se),
        "sponsor_probability_positive": float(np.mean(np.asarray(sponsor_deltas) > 0.0)),
        "origin_probability_positive": float(np.mean(np.asarray(origin_draws) > 0.0)),
    }


def _slice_gates(frame: pd.DataFrame, labels: np.ndarray, candidate: np.ndarray, reference: np.ndarray, matrix: np.ndarray) -> dict:
    phase = frame["phase"].fillna("Missing").astype(str).str.lower()
    conditions = frame["conditions_text"].fillna("").astype(str)
    respiratory = conditions.str.contains("respirat|infect|covid|sars|influenza|pneum|asthma|copd", case=False, regex=True)
    respiratory = respiratory | frame["llm_area_respiratory_infectious"].gt(0.5)
    disagreement = matrix.max(axis=1) - matrix.min(axis=1)
    text_scale = frame["summary_log_words"].to_numpy(dtype=float) + frame["eligibility_log_words"].to_numpy(dtype=float)
    masks = {
        "sparse_sponsor": frame["sponsor_history_count_raw"].le(20).to_numpy(),
        "emerging_condition": frame["condition_history_count_raw"].le(0).to_numpy(),
        "phase_4": phase.str.contains("phase 4").to_numpy(),
        "not_applicable": phase.str.contains("not applicable").to_numpy(),
        "shortest_text": text_scale <= np.quantile(text_scale, 0.25),
        "long_eligibility": frame["eligibility_truncated"].gt(0.5).to_numpy(),
        "respiratory_infectious": respiratory.to_numpy(),
        "high_channel_disagreement": disagreement >= np.quantile(disagreement, 0.75),
    }
    output = {}
    for name, mask in masks.items():
        candidate_auc = _auc(labels[mask], candidate[mask])
        reference_auc = _auc(labels[mask], reference[mask])
        output[name] = {
            "count": int(mask.sum()),
            "candidate_auc": candidate_auc,
            "incumbent_auc": reference_auc,
            "delta": candidate_auc - reference_auc,
        }
    return output


def _stack_gate(labels: np.ndarray, origins: np.ndarray, matrix: np.ndarray, coverage: pd.DataFrame) -> tuple[np.ndarray, dict]:
    incumbent = matrix[:, [0, 2]].mean(axis=1)
    selection_mask = origins <= 2018
    selection_origins = ORIGINS[:5]
    equal_weights = np.full(4, 0.25, dtype=float)
    equal_nested = matrix @ equal_weights
    grid_nested = np.zeros(len(labels), dtype=float)
    nested_weights = {}
    for held in selection_origins:
        training_origins = selection_origins[selection_origins != held]
        weights = _select_grid(labels, matrix, origins, training_origins)
        selected = origins == held
        grid_nested[selected] = matrix[selected] @ weights
        nested_weights[str(int(held))] = weights.tolist()
    frozen_grid = _select_grid(labels, matrix, origins, selection_origins)
    grid_nested[origins == 2019] = matrix[origins == 2019] @ frozen_grid
    equal_uncertainty = _uncertainty(
        labels[selection_mask],
        grid_nested[selection_mask],
        equal_nested[selection_mask],
        origins[selection_mask],
        coverage.loc[selection_mask, "lead_sponsor_cluster"].to_numpy(),
        2337,
    )
    grid_scores = _origin_auc(labels[selection_mask], grid_nested[selection_mask], origins[selection_mask])
    equal_scores = _origin_auc(labels[selection_mask], equal_nested[selection_mask], origins[selection_mask])
    grid_improvement = float(grid_scores.mean() - equal_scores.mean())
    if grid_improvement > equal_uncertainty["larger_paired_se"]:
        selected_name = "capped_simplex"
        selected_weights = frozen_grid
        candidate = grid_nested
    else:
        selected_name = "equal_tie_preference"
        selected_weights = equal_weights
        candidate = equal_nested
    candidate_scores = _origin_auc(labels, candidate, origins)
    incumbent_scores = _origin_auc(labels, incumbent, origins)
    uncertainty = _uncertainty(
        labels,
        candidate,
        incumbent,
        origins,
        coverage["lead_sponsor_cluster"].to_numpy(),
        1337,
    )
    improvement = float(candidate_scores.mean() - incumbent_scores.mean())
    improved_origins = int(np.sum(candidate_scores > incumbent_scores))
    candidate_worst_two = float(np.mean(np.sort(candidate_scores)[:2]))
    incumbent_worst_two = float(np.mean(np.sort(incumbent_scores)[:2]))
    candidate_cvar = float(np.mean(np.sort(candidate_scores)[:3]))
    incumbent_cvar = float(np.mean(np.sort(incumbent_scores)[:3]))
    slices = _slice_gates(coverage, labels, candidate, incumbent, matrix)
    frozen_2019_delta = float(candidate_scores[-1] - incumbent_scores[-1])
    passed = bool(
        improvement >= uncertainty["larger_paired_se"]
        and improved_origins >= 4
        and candidate_worst_two >= incumbent_worst_two
        and candidate_cvar >= incumbent_cvar
        and min(value["delta"] for value in slices.values()) >= -0.02
        and frozen_2019_delta >= 0.0
    )
    diagnostics = {
        "families": FAMILIES,
        "precommitted_designs": {
            "equal": {"weights": equal_weights.tolist(), "selection_origin_auc": equal_scores.tolist()},
            "capped_simplex": {
                "step": 0.125,
                "cap": 0.625,
                "nested_weights": nested_weights,
                "frozen_weights": frozen_grid.tolist(),
                "selection_origin_auc": grid_scores.tolist(),
                "nested_improvement_over_equal": grid_improvement,
                "paired_to_equal": equal_uncertainty,
            },
        },
        "selected_design": selected_name,
        "selected_weights": selected_weights.tolist(),
        "incumbent_origin_auc": incumbent_scores.tolist(),
        "candidate_origin_auc": candidate_scores.tolist(),
        "improvement": improvement,
        "improved_origins": improved_origins,
        "incumbent_worst_two": incumbent_worst_two,
        "candidate_worst_two": candidate_worst_two,
        "incumbent_cvar_worst_three": incumbent_cvar,
        "candidate_cvar_worst_three": candidate_cvar,
        "frozen_2019_delta": frozen_2019_delta,
        "paired_uncertainty": uncertainty,
        "slice_gates": slices,
        "passed": passed,
    }
    print(f"[stack_gate] {json.dumps(diagnostics, separators=(',', ':'))}")
    return selected_weights, diagnostics


def _feature_matrix(cache_dir: Path, expected_keys: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    bundle_path = cache_dir / "relational_neighborhood_rank16_v1" / "official_bundle.npz"
    compact_path = cache_dir / "common_raw_family_stack_v1" / "exact_ks_features_623076f65f45e3f4f052c2a6.parquet"
    _require(bundle_path, BUNDLE_HASH, "official neighborhood bundle")
    _require(compact_path, COMPACT_HASH, "exact-K/S compact features")
    bundle = np.load(bundle_path, allow_pickle=False)
    if bundle["keys"].astype(str).tolist() != expected_keys.astype(str).tolist():
        raise RuntimeError("official neighborhood bundle is reordered or not aligned")
    if bundle["features"].shape != (13779, 2646) or len(bundle["feature_names"]) != 2646:
        raise RuntimeError(f"official neighborhood geometry changed: {bundle['features'].shape}")
    compact = pd.read_parquet(compact_path)
    if compact["_key"].duplicated().any():
        raise RuntimeError("exact-K/S compact features contain duplicate keys")
    compact = compact.set_index("_key").loc[expected_keys.astype(str)]
    expected_labels = compact["outcome"].to_numpy(dtype=float)
    excluded = {"outcome", "K", "S", "is_replay", "origin_year", "_row_id"}
    compact_columns = [
        column
        for column in compact.select_dtypes(include=[np.number, "bool"]).columns
        if column not in excluded
    ]
    forbidden = [column for column in compact_columns if column.lower() in {"label", "future_k", "future_s", "replay_marker", "origin_metadata"}]
    if forbidden:
        raise RuntimeError(f"forbidden Stage-A feature columns found: {forbidden}")
    compact_values = compact[compact_columns].to_numpy(dtype=np.float32)
    neighborhood_values = bundle["features"].astype(np.float32)
    values = np.concatenate([compact_values, neighborhood_values], axis=1)
    values[np.isinf(values)] = np.nan
    names = compact_columns + bundle["feature_names"].astype(str).tolist()
    if len(names) != len(set(names)):
        raise RuntimeError("Stage-A feature names are not unique")
    print(f"[features] official_rows={len(values)} neighborhood_features=2646 compact_features={len(compact_columns)} total_features={values.shape[1]}")
    return values, names, expected_labels


def _fit_stage_a(values: np.ndarray, labels: np.ndarray, estimators: int) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=15,
        max_depth=5,
        min_child_samples=100,
        learning_rate=0.035,
        n_estimators=estimators,
        feature_fraction=0.65,
        bagging_fraction=0.85,
        bagging_freq=1,
        lambda_l1=0.2,
        lambda_l2=3.0,
        max_bin=63,
        random_state=1337,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "11")),
        verbosity=-1,
        deterministic=True,
        force_col_wise=True,
    )
    model.fit(values, labels)
    return model


def _final_neighborhood(
    cache_dir: Path,
    values: np.ndarray,
    names: list[str],
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    train_keys: np.ndarray,
    val_keys: np.ndarray,
    test_keys: np.ndarray,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    directory = cache_dir / "relational_neighborhood_rank16_v1"
    path = directory / "final_stage_a_components_v1.npz"
    if path.exists() and not debug:
        data = np.load(path, allow_pickle=False)
        valid = (
            data["val_keys"].astype(str).tolist() == val_keys.astype(str).tolist()
            and data["test_keys"].astype(str).tolist() == test_keys.astype(str).tolist()
            and np.array_equal(data["train_labels"], train_labels)
            and np.array_equal(data["val_labels"], val_labels)
            and data["feature_names"].astype(str).tolist() == names
        )
        if valid:
            print(f"[stage_a_final] cache_hit rows_a={len(train_labels)} rows_b={len(train_labels) + len(val_labels)}")
            return data["val_prediction"].astype(float), data["test_prediction"].astype(float), {"cache_hit": True, "estimators": 450}
    train_count = len(train_labels)
    val_count = len(val_labels)
    if debug:
        rng = np.random.default_rng(1337)
        selected = rng.choice(train_count, size=min(3500, train_count), replace=False)
        estimators = 80
        model_a = _fit_stage_a(values[selected], train_labels[selected], estimators)
        val_prediction = model_a.predict_proba(values[train_count : train_count + val_count])[:, 1]
        source_b_indices = np.concatenate([selected, np.arange(train_count, train_count + val_count)])
        source_b_labels = np.concatenate([train_labels[selected], val_labels])
        model_b = _fit_stage_a(values[source_b_indices], source_b_labels, estimators)
        test_prediction = model_b.predict_proba(values[train_count + val_count :])[:, 1]
        return val_prediction, test_prediction, {"cache_hit": False, "debug": True, "estimators": estimators, "model_a_rows": len(selected), "model_b_rows": len(source_b_indices)}
    model_a = _fit_stage_a(values[:train_count], train_labels, 450)
    val_prediction = model_a.predict_proba(values[train_count : train_count + val_count])[:, 1].astype(np.float64)
    model_b_values = values[: train_count + val_count]
    model_b_labels = np.concatenate([train_labels, val_labels])
    model_b = _fit_stage_a(model_b_values, model_b_labels, 450)
    test_prediction = model_b.predict_proba(values[train_count + val_count :])[:, 1].astype(np.float64)
    payload = {
        "train_keys": train_keys.astype(np.str_),
        "val_keys": val_keys.astype(np.str_),
        "test_keys": test_keys.astype(np.str_),
        "train_labels": train_labels.astype(np.float64),
        "val_labels": val_labels.astype(np.float64),
        "feature_names": np.asarray(names, dtype=np.str_),
        "val_prediction": val_prediction,
        "test_prediction": test_prediction,
    }
    temporary = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    register_artifact(
        cache_dir,
        {
            "name": "lane1 neighborhood Stage-A Model-A and Model-B vectors",
            "path": str(path.relative_to(cache_dir)),
            "description": "Shallow LightGBM Model A trained on official train and Model B trained on train plus validation",
            "content_key": "relational_neighborhood_rank16_v1:final_stage_a_components_v1",
            "rebuild_hint": "Fit the fixed 450-tree Stage-A configuration on the aligned official feature bundle",
        },
    )
    return val_prediction, test_prediction, {"cache_hit": False, "estimators": 450, "model_a_rows": train_count, "model_b_rows": train_count + val_count}


def _load_final_families(cache_dir: Path, split: str, keys: np.ndarray, neighborhood: np.ndarray) -> np.ndarray:
    finalist = cache_dir / "cross_branch_finalists" / f"generic_exp_1_run_0007_{split}.npy"
    _require(finalist, RUN_0007_HASHES[split], f"run_0007 {split}")
    run_prediction = np.load(finalist, allow_pickle=False).astype(float)
    fusion_name, fusion_hash = FUSION_FINAL[split]
    fusion_path = cache_dir / "lane3_hierarchical_fusion" / fusion_name
    _require(fusion_path, fusion_hash, f"fusion {split}")
    fusion = np.load(fusion_path, allow_pickle=False)
    fusion_prediction = _align(fusion["keys"], fusion["prediction_2"], keys, f"fusion {split}")
    mechanism_path = cache_dir / "lane3_multitask_fusion" / "raw_run0033_mechanistic_final.npz"
    _require(mechanism_path, MECHANISM_FINAL_HASH, "mechanism final")
    mechanism = np.load(mechanism_path, allow_pickle=False)
    mechanism_prediction = _align(mechanism[f"{split}_keys"], mechanism[f"{split}_prediction"], keys, f"mechanism {split}")
    if run_prediction.shape != neighborhood.shape or run_prediction.shape != fusion_prediction.shape:
        raise RuntimeError(f"final {split} family shapes disagree")
    return np.column_stack([run_prediction, fusion_prediction, mechanism_prediction, neighborhood])


def _verify_incumbent(cache_dir: Path) -> dict:
    output = {}
    for split, expected in RUN_0033_HASHES.items():
        path = cache_dir / "cross_branch_finalists" / f"generic_exp_5_run_0033_{split}.npy"
        _require(path, expected, f"run_0033 {split}")
        output[split] = expected
    print(f"[recon] run_0033_hashes={json.dumps(output, separators=(',', ':'))}")
    return output


def main() -> None:
    started = time.time()
    debug = is_debug()
    cache_dir = shared_cache_dir()
    output_dir = Path("output_data_generic_exp_13")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[environment] "
        + json.dumps(
            {
                "python": platform.python_version(),
                "lightgbm": lgb.__version__,
                "numpy": np.__version__,
                "debug": debug,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
            separators=(",", ":"),
        )
    )
    context = load_task()
    train = context.train.df.copy().reset_index(drop=True)
    val = context.val.df.copy().reset_index(drop=True)
    test = context.test.df.copy().reset_index(drop=True)
    train_keys = _keys(train, "train")
    val_keys = _keys(val, "val")
    test_keys = _keys(test, "test")
    official_keys = np.concatenate([train_keys, val_keys, test_keys])
    incumbent_hashes = _verify_incumbent(cache_dir)
    neighborhood_oof = _sanitize_oof(cache_dir)
    labels, origins, oof_matrix, coverage = _load_oof_families(cache_dir, neighborhood_oof)
    weights, gate = _stack_gate(labels, origins, oof_matrix, coverage)
    if not gate["passed"]:
        incumbent_val = np.load(cache_dir / "cross_branch_finalists" / "generic_exp_5_run_0033_val.npy", allow_pickle=False).astype(np.float64)
        incumbent_test = np.load(cache_dir / "cross_branch_finalists" / "generic_exp_5_run_0033_test.npy", allow_pickle=False).astype(np.float64)
        save_predictions(incumbent_val, incumbent_test)
        diagnostics = {"status": "structural_gate_failed_duplicate_incumbent", "stack_gate": gate, "run_0033_hashes": incumbent_hashes}
        run_data_dir().joinpath("metrics.json").write_text(json.dumps(diagnostics, indent=2))
        print("[submission] structural gate failed; wrote immutable run_0033 fallback")
        return
    values, feature_names, cached_labels = _feature_matrix(cache_dir, official_keys)
    train_labels = train["outcome"].to_numpy(dtype=float)
    val_labels = val["outcome"].to_numpy(dtype=float)
    if not np.array_equal(cached_labels[: len(train)], train_labels):
        raise RuntimeError("cached official train labels disagree with task labels")
    if not np.array_equal(cached_labels[len(train) : len(train) + len(val)], val_labels):
        raise RuntimeError("cached official validation labels disagree with task labels")
    if np.isfinite(cached_labels[len(train) + len(val) :]).any():
        raise RuntimeError("cached test rows unexpectedly expose outcomes")
    neighborhood_val, neighborhood_test, final_diagnostics = _final_neighborhood(
        cache_dir,
        values,
        feature_names,
        train_labels,
        val_labels,
        train_keys,
        val_keys,
        test_keys,
        debug,
    )
    val_raw = _load_final_families(cache_dir, "val", val_keys, neighborhood_val)
    test_raw = _load_final_families(cache_dir, "test", test_keys, neighborhood_test)
    val_ranked = np.column_stack([_midranks(val_raw[:, index], val["timestamp"].to_numpy()) for index in range(4)])
    test_ranked = np.column_stack([_midranks(test_raw[:, index], test["timestamp"].to_numpy()) for index in range(4)])
    val_predictions = np.clip(val_ranked @ weights, 1e-5, 1.0 - 1e-5).astype(np.float64)
    test_predictions = np.clip(test_ranked @ weights, 1e-5, 1.0 - 1e-5).astype(np.float64)
    if val_predictions.shape != (960,) or test_predictions.shape != (825,):
        raise RuntimeError("final prediction shapes violate the task contract")
    if not np.isfinite(val_predictions).all() or not np.isfinite(test_predictions).all():
        raise RuntimeError("final predictions contain nonfinite values")
    final_correlations = {}
    for left in range(4):
        for right in range(left + 1, 4):
            final_correlations[f"{FAMILIES[left]}__{FAMILIES[right]}"] = float(spearmanr(val_ranked[:, left], val_ranked[:, right]).statistic)
    diagnostics = {
        "status": "gates_passed",
        "model_a_fit_sources": ["official_train_only"],
        "model_b_fit_sources": ["official_train", "official_validation"],
        "replay_rows_used": 0,
        "stack_gate": gate,
        "frozen_weights": weights.tolist(),
        "stage_a_final": final_diagnostics,
        "run_0033_hashes": incumbent_hashes,
        "final_family_rank_correlations": final_correlations,
        "candidate_validation_auc_diagnostic_only_after_freeze": _auc(val_labels, val_predictions),
        "prediction_summary": {
            "val_shape": list(val_predictions.shape),
            "test_shape": list(test_predictions.shape),
            "val_min": float(val_predictions.min()),
            "val_max": float(val_predictions.max()),
            "val_std": float(val_predictions.std()),
            "test_min": float(test_predictions.min()),
            "test_max": float(test_predictions.max()),
            "test_std": float(test_predictions.std()),
        },
        "elapsed_seconds": time.time() - started,
    }
    save_predictions(val_predictions, test_predictions)
    run_data_dir().joinpath("metrics.json").write_text(json.dumps(diagnostics, indent=2))
    output_dir.joinpath("metrics.json").write_text(json.dumps(diagnostics, indent=2))
    np.save(output_dir / "val_predictions.npy", val_predictions)
    np.save(output_dir / "test_predictions.npy", test_predictions)
    print(f"[diagnostics] validation_auc_after_freeze={diagnostics['candidate_validation_auc_diagnostic_only_after_freeze']:.10f} correlations={json.dumps(final_correlations, separators=(',', ':'))}")
    print(f"[timing] phase=complete elapsed_seconds={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
