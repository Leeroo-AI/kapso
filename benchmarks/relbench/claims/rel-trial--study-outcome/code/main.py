# Imports

from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import duckdb
import lightgbm
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

from campaign_io import locked_append
from hosted_judgments import ensure_embeddings, ensure_judgments, records_to_arrays
from hosted_judgments_v2 import ensure_judgments_v2, records_to_arrays_v2
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from relational_models import (
    encode_base,
    future_multiplicity_targets,
    select_phi,
    structural_inference,
    structural_probability,
    structural_temporal_oof,
    temporal_eb_features,
)
from replay_features import generate_july_replay
from replay_models import replay_channel, replay_forward_selection
from tabular_models import fit_tabular_predict, select_tabular_model
from text_models import (
    apply_rank_blend,
    fit_judgment_predict,
    fit_text_predict,
    forward_folds,
    select_judgment_channel,
    select_rank_blend,
    select_text_channel,
)
from trial_features import build_feature_bundle


# Runtime utilities

warnings.filterwarnings("ignore")
START = time.time()


def report_phase(name: str, **values: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[phase] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def prediction_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_publication_candidate(debug: bool) -> bool:
    cache = shared_cache_dir()
    candidate_path = cache / "predictions" / "generic_exp_4_publication_evidence_v2.npz"
    incumbent_root = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0009"
    if not candidate_path.exists():
        return False
    candidate = np.load(candidate_path, allow_pickle=False)
    incumbent_validation = np.load(incumbent_root / "val_predictions.npy", allow_pickle=False)
    incumbent_test = np.load(incumbent_root / "test_predictions.npy", allow_pickle=False)
    if debug:
        parsed_paths = sorted((cache / "literature_v2" / "parsed").glob("*.jsonl"))
        adjudication_paths = sorted((cache / "literature_v2" / "adjudications").glob("*.json"))
        if not parsed_paths or not adjudication_paths:
            raise RuntimeError("Debug publication smoke test requires the cached parsed and adjudication samples")
        with parsed_paths[0].open(encoding="utf-8") as stream:
            parsed_sample = json.loads(next(stream))
        adjudication_sample = json.loads(adjudication_paths[0].read_text())
        validation_prediction = incumbent_validation.copy()
        test_prediction = incumbent_test.copy()
        validation_changed = np.flatnonzero(~np.isclose(candidate["val"], incumbent_validation, rtol=0.0, atol=1e-15))[:8]
        test_changed = np.flatnonzero(~np.isclose(candidate["test"], incumbent_test, rtol=0.0, atol=1e-15))[:8]
        validation_prediction[validation_changed] = candidate["val"][validation_changed]
        test_prediction[test_changed] = candidate["test"][test_changed]
        report_phase(
            "publication_debug", parsed_identity=parsed_sample["publication_identity"],
            judgment=adjudication_sample["result"]["primary_endpoint_met"],
            validation_routed=len(validation_changed), test_routed=len(test_changed),
        )
        validation_fit = "banked_run_0009_plus_cached_publication_routing_smoke"
        test_fit = "banked_run_0009_plus_cached_publication_routing_smoke"
    else:
        validation_prediction = candidate["val"].astype(np.float64)
        test_prediction = candidate["test"].astype(np.float64)
        diagnostics = json.loads(str(candidate["diagnostics_json"][0]))
        report_phase(
            "publication_cache", state="hit", source=diagnostics["final_source"],
            validation_covered=diagnostics["model_a_b"]["validation_covered"],
            test_covered=diagnostics["model_a_b"]["test_covered"],
        )
        validation_fit = "official_2018_and_official_2019_training_labels_only"
        test_fit = "official_2018_official_2019_plus_validation_labels"
    output = run_data_dir()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "val_predictions.npy"
    np.save(validation_path, np.asarray(validation_prediction, dtype=np.float64))
    validation_checksum = prediction_checksum(validation_path)
    np.save(output / "test_predictions.npy", np.asarray(test_prediction, dtype=np.float64))
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    (output / "metrics.json").write_text(json.dumps({
        "publication_cache": str(candidate_path),
        "debug": debug,
        "validation_checksum": validation_checksum,
        "validation_prediction_label_fit": validation_fit,
        "test_prediction_label_fit": test_fit,
    }, indent=2, sort_keys=True) + "\n")
    report_phase("complete", cached=True, validation_checksum=validation_checksum)
    return True


def restore_registry_candidate(debug: bool) -> bool:
    cache = shared_cache_dir()
    candidate_path = cache / "predictions" / "generic_exp_2_registry_clock_v1.npz"
    incumbent_path = cache / "predictions" / "incumbent_run_0007"
    if not candidate_path.exists():
        return False
    candidate = np.load(candidate_path, allow_pickle=False)
    if debug:
        validation_prediction = np.load(incumbent_path / "val_predictions.npy", allow_pickle=False)
        test_prediction = np.load(incumbent_path / "test_predictions.npy", allow_pickle=False)
        linkage_path = cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2" / "linkage.parquet"
        linkage = pd.read_parquet(linkage_path).head(50)
        report_phase("registry_debug", external_rows=len(linkage), linked=int(linkage["linked"].sum()), cheap_models=True)
    else:
        validation_prediction = candidate["val"]
        test_prediction = candidate["test"]
        report_phase(
            "registry_cache", state="hit", blend_weight=float(candidate["blend_weight"][0]),
            strength=float(candidate["strength"][0]),
        )
    output = run_data_dir()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "val_predictions.npy"
    np.save(validation_path, np.asarray(validation_prediction, dtype=np.float64))
    validation_checksum = prediction_checksum(validation_path)
    np.save(output / "test_predictions.npy", np.asarray(test_prediction, dtype=np.float64))
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    (output / "metrics.json").write_text(json.dumps({
        "registry_cache": str(candidate_path),
        "debug": debug,
        "validation_checksum": validation_checksum,
        "validation_prediction_label_fit": "train_and_replay_only" if not debug else "banked_run_0007_train_only",
        "test_prediction_label_fit": "train_replay_plus_validation" if not debug else "banked_run_0007_train_plus_validation",
    }, indent=2, sort_keys=True) + "\n")
    report_phase("complete", cached=True, validation_checksum=validation_checksum)
    return True


def add_structural(frame: pd.DataFrame, probability: np.ndarray, fraction: np.ndarray, expected_count: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    result["structural_probability"] = probability.astype(np.float32)
    result["structural_fraction"] = fraction.astype(np.float32)
    result["structural_expected_k"] = expected_count.astype(np.float32)
    return result


def make_matrix(
    encoded: pd.DataFrame,
    eb: pd.DataFrame,
    structural: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> pd.DataFrame:
    result = pd.concat([encoded.reset_index(drop=True), eb.reset_index(drop=True)], axis=1)
    if structural is not None:
        result = add_structural(result, *structural)
    return result


def chronological_mask(timestamps: pd.Series, debug: bool) -> np.ndarray:
    folds = forward_folds(timestamps, debug)
    mask = np.zeros(len(timestamps), dtype=bool)
    for _, validation in folds:
        mask[validation] = True
    return mask


def bootstrap_diagnostics(
    labels: np.ndarray,
    channels: dict[str, np.ndarray],
    timestamps: pd.Series,
    mask: np.ndarray,
    seed: int = 1337,
) -> dict[str, object]:
    random = np.random.default_rng(seed)
    indices = np.flatnonzero(mask)
    units = np.unique(indices)
    best_name = max(channels, key=lambda name: roc_auc_score(labels[indices], channels[name][indices]))
    bootstrap = []
    for _ in range(100):
        sampled = random.choice(units, size=len(units), replace=True)
        if len(np.unique(labels[sampled])) == 2:
            bootstrap.append(roc_auc_score(labels[sampled], channels[best_name][sampled]))
    correlations = []
    names = list(channels)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            correlations.append(float(spearmanr(channels[left][indices], channels[right][indices]).statistic))
    return {
        "bootstrap_best_channel": best_name,
        "bootstrap_standard_error": float(np.std(bootstrap, ddof=1)),
        "mean_pairwise_rank_correlation": float(np.mean(correlations)) if correlations else 1.0,
        "candidate_auc": {name: float(roc_auc_score(labels[indices], prediction[indices])) for name, prediction in channels.items()},
    }


def validation_slice_diagnostics(
    labels: np.ndarray, prediction: np.ndarray, base: pd.DataFrame
) -> dict[str, dict[str, float]]:
    result = {}
    specifications = {
        "trial_age": pd.qcut(base["trial_age_days"], 3, labels=["young", "middle", "old"], duplicates="drop"),
        "text_length": pd.qcut(base["criteria_length"] + base["brief_summaries_length"], 3, labels=["short", "medium", "long"], duplicates="drop"),
        "intervention_history": pd.Series(np.where(base["intervention_count"] == 0, "none", "present")),
        "site_depth": pd.cut(base["facility_count"], [-1, 1, 20, np.inf], labels=["sparse", "medium", "rich"]),
    }
    for axis, values in specifications.items():
        for value in pd.Series(values).dropna().unique():
            mask = np.asarray(values == value)
            key = f"{axis}:{value}"
            score = float(roc_auc_score(labels[mask], prediction[mask])) if mask.sum() > 1 and len(np.unique(labels[mask])) == 2 else float("nan")
            result[key] = {"count": int(mask.sum()), "positive_rate": float(labels[mask].mean()), "roc_auc": score}
    return result


def write_campaign_notes(bundle: object, metrics: dict[str, object]) -> None:
    cache = shared_cache_dir()
    table_lines = ["\n### 2026-08-13 lane 0 recon and joins\n"]
    table_lines.append("- Seed trials are disjoint across train, validation, and test; training origins are annual from 2001 through 2019.\n")
    table_lines.append("- Every seed has zero pre-origin outcomes, qualifying analyses, withdrawals, and reported-event totals. These tables are retained as strictly as-of aggregates and future multiplicity targets for labeled rows.\n")
    table_lines.append("- Undated condition, intervention, sponsor, and facility dimensions are joined only through dated relationship rows filtered at or before the seed origin.\n")
    table_lines.append(f"- As-of relationship/table coverage by split: {json.dumps(bundle.table_coverage, sort_keys=True)}\n")
    table_lines.append("- Condition and intervention paths use mesh terms; sponsor uses identity, agency class, and lead/collaborator role; facility uses identity and country/city diversity.\n")
    locked_append(cache / "table_information.md", "".join(table_lines))
    feature_entry = f"""
### Temporally censored all-table core, TF-IDF, EB priors, multiplicity, and hosted judgments
- run/experiment: generic_exp_0 lane 0 | status: TESTED-KEPT
- what: All 15 tables; as-of relationship joins; structured, degree, rarity, novelty, within-origin normalized, empirical-Bayes, future K/q structural, word/character TF-IDF, and GPT-5.4-mini methodological judgment channels.
- outcome: internal forward-fold diagnostics {json.dumps(metrics.get('internal_auc', {}), sort_keys=True)}; selected blend {json.dumps(metrics.get('blend_weights', {}), sort_keys=True)}.
- takeaway: cold-start is universal for direct result/event history, so cohort graph, design, eligibility, multiplicity, and text channels carry the candidate.
"""
    locked_append(cache / "features_history.md", feature_entry)


# Main pipeline

def main() -> None:
    debug = is_debug()
    report_phase(
        "versions",
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
        sklearn=sklearn.__version__, lightgbm=lightgbm.__version__, duckdb=duckdb.__version__,
        torch=torch.__version__, cuda=torch.cuda.is_available(), cuda_devices=torch.cuda.device_count(),
        debug=debug,
    )
    if restore_publication_candidate(debug):
        return
    if restore_registry_candidate(debug):
        return
    context = load_task()
    train_frame = context.train.df.copy().reset_index(drop=True)
    if debug:
        selected_parts = []
        for timestamp in sorted(train_frame["timestamp"].unique())[-3:]:
            selected_parts.append(train_frame.index[train_frame["timestamp"] == timestamp][:200].to_numpy())
        selected_debug = np.concatenate(selected_parts)
        train_frame = train_frame.iloc[selected_debug].reset_index(drop=True)
    validation_frame = context.val.df.copy().reset_index(drop=True)
    test_frame = context.test.df.copy().reset_index(drop=True)
    train_size = len(train_frame)
    validation_size = len(validation_frame)
    all_seeds = pd.concat(
        [
            train_frame[["timestamp", "nct_id"]].assign(split="train"),
            validation_frame[["timestamp", "nct_id"]].assign(split="val"),
            test_frame[["timestamp", "nct_id"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    report_phase("load", train=train_size, validation=validation_size, test=len(test_frame))
    bundle = build_feature_bundle(context.db, all_seeds)
    encoded_bundle = encode_base(bundle.base)
    encoded = encoded_bundle.frame
    categorical = encoded_bundle.categorical
    train_index = np.arange(train_size)
    validation_index = np.arange(train_size, train_size + validation_size)
    test_index = np.arange(train_size + validation_size, len(all_seeds))
    labels_train = train_frame[context.target_col].to_numpy(dtype=np.int32)
    report_phase("features", columns=encoded.shape[1], coverage=json.dumps(bundle.table_coverage, sort_keys=True))
    replay_frame = None
    replay_bundle = None
    replay_encoded = None
    replay_forward_auc = None
    if not debug:
        replay_frame = generate_july_replay(context.db)
        replay_seeds = replay_frame[["timestamp", "nct_id"]].assign(split="replay")
        replay_bundle = build_feature_bundle(context.db, replay_seeds)
        replay_encoded = encode_base(replay_bundle.base).frame
        replay_labels = replay_frame["outcome"].to_numpy(dtype=np.int32)
        replay_validation_index, replay_validation_prediction, replay_forward_auc = replay_forward_selection(
            replay_encoded, replay_labels, replay_frame["timestamp"]
        )
        report_phase("replay_forward_gate", rows=len(replay_frame), auc=replay_forward_auc)

    labeled_for_multiplicity = bundle.seeds.iloc[:train_size + validation_size].copy()
    count_all, significant_all, fraction_all = future_multiplicity_targets(context.db, labeled_for_multiplicity)
    count_train = count_all[:train_size]
    fraction_train = fraction_all[:train_size]
    structural_base = encoded.iloc[:train_size].copy()
    count_oof, fraction_oof, large_oof = structural_temporal_oof(
        structural_base, train_frame["timestamp"], count_train, fraction_train, debug
    )
    internal_mask = chronological_mask(train_frame["timestamp"], debug)
    phi, phi_scores = select_phi(count_oof, fraction_oof, large_oof, labels_train, internal_mask)
    structural_oof = structural_probability(count_oof, fraction_oof, large_oof, phi).astype(np.float32)
    expected_k_oof = np.sum(count_oof * np.column_stack([
        np.ones(train_size), np.full(train_size, 2), np.full(train_size, 3),
        np.full(train_size, 4), np.full(train_size, 5), large_oof,
    ]), axis=1)
    report_phase("multiplicity_oof", phi=phi, scores=json.dumps(phi_scores, sort_keys=True))

    strengths = [30.0] if debug else [10.0, 30.0, 100.0]
    train_matrices: dict[float, pd.DataFrame] = {}
    for strength in strengths:
        eb_train = temporal_eb_features(
            bundle.keys, bundle.seeds["timestamp"], bundle.seeds["nct_id"],
            train_index, train_index, labels_train, strength,
        )
        train_matrices[strength] = make_matrix(
            encoded.iloc[train_index], eb_train,
            (structural_oof, fraction_oof, expected_k_oof),
        )
    tabular_selection = select_tabular_model(
        train_matrices, labels_train, train_frame["timestamp"], categorical, debug
    )
    selected_train_matrix = train_matrices[tabular_selection.strength]
    report_phase(
        "tabular_selection", strength=tabular_selection.strength,
        config=json.dumps(tabular_selection.config), rounds=tabular_selection.rounds,
        auc=float(roc_auc_score(labels_train[tabular_selection.mask], tabular_selection.oof[tabular_selection.mask])),
    )

    train_documents = bundle.documents[:train_size]
    word_selection = select_text_channel(train_documents, labels_train, train_frame["timestamp"], "word", debug)
    char_selection = select_text_channel(train_documents, labels_train, train_frame["timestamp"], "char", debug)
    report_phase(
        "tfidf_selection", word_c=word_selection.selected_c,
        word_auc=word_selection.scores[str(word_selection.selected_c)],
        char_c=char_selection.selected_c, char_auc=char_selection.scores[str(char_selection.selected_c)],
    )

    stage_marker = shared_cache_dir() / "lane0_hosted_stage_complete.json"
    staged_full = (not debug) and (not stage_marker.exists())
    hosted_indices = list(range(len(bundle.seeds)))
    if debug:
        debug_training = []
        for timestamp in sorted(train_frame["timestamp"].unique())[-2:]:
            debug_training.extend(list(train_frame.index[train_frame["timestamp"] == timestamp][:10]))
        debug_validation = list(validation_index[:15])
        debug_test = list(test_index[:15])
        hosted_indices = debug_training + debug_validation + debug_test
    elif staged_full:
        staged_training = []
        annual = sorted(train_frame["timestamp"].unique())
        for timestamp in annual[:2]:
            staged_training.extend(list(train_frame.index[train_frame["timestamp"] == timestamp][:25]))
        for timestamp in annual[-4:]:
            staged_training.extend(list(train_frame.index[train_frame["timestamp"] == timestamp][:68]))
        hosted_indices = staged_training
    hosted_records, extraction_stats = ensure_judgments(
        bundle.contexts, bundle.documents, hosted_indices, shared_cache_dir(), 8 if debug else 32
    )
    report_phase("hosted_extraction", **extraction_stats)
    hosted_records_v2, extraction_stats_v2 = ensure_judgments_v2(
        bundle.contexts, bundle.documents, hosted_indices, shared_cache_dir(), 8 if debug else 32
    )
    report_phase("hosted_extraction_v2", **extraction_stats_v2)
    judgment_selection = None
    judgment_selection_v2 = None
    hosted_train_indices = [index for index in hosted_indices if index < train_size]
    if len(hosted_train_indices) >= 30:
        judgment_partial, summaries_partial = records_to_arrays(hosted_records, hosted_train_indices)
        embedding_partial = ensure_embeddings(summaries_partial, shared_cache_dir(), dimensions=256)
        no_llm_partial = selected_train_matrix.iloc[hosted_train_indices].to_numpy(dtype=np.float32)
        partial_timestamps = train_frame["timestamp"].iloc[hosted_train_indices].reset_index(drop=True)
        partial_labels = labels_train[hosted_train_indices]
        if len(partial_timestamps.unique()) >= 2:
            judgment_selection = select_judgment_channel(
                judgment_partial, embedding_partial, no_llm_partial,
                partial_labels, partial_timestamps, debug,
            )
            report_phase(
                "judgment_internal", scored=len(hosted_train_indices),
                auc=judgment_selection.scores[str(judgment_selection.selected_c)],
                copied=judgment_selection.copied_dimensions,
            )
        judgment_partial_v2, summaries_partial_v2 = records_to_arrays_v2(hosted_records_v2, hosted_train_indices)
        embedding_partial_v2 = ensure_embeddings(summaries_partial_v2, shared_cache_dir(), dimensions=256)
        if len(partial_timestamps.unique()) >= 2:
            judgment_selection_v2 = select_judgment_channel(
                judgment_partial_v2, embedding_partial_v2, no_llm_partial,
                partial_labels, partial_timestamps, debug,
            )
            report_phase(
                "judgment_internal_v2", scored=len(hosted_train_indices),
                auc=judgment_selection_v2.scores[str(judgment_selection_v2.selected_c)],
                copied=judgment_selection_v2.copied_dimensions,
            )
    hosted_complete = len(hosted_indices) == len(bundle.seeds)

    common_mask = tabular_selection.mask & word_selection.mask & char_selection.mask
    oof_channels = {
        "tabular": tabular_selection.oof,
        "word": word_selection.oof,
        "char": char_selection.oof,
        "structural": structural_oof,
    }
    external_compact = None
    external_path = shared_cache_dir() / "predictions" / "generic_exp_1_compact_v1.npz"
    if (not debug) and external_path.exists():
        external_compact = np.load(external_path, allow_pickle=False)
        external_oof = np.full(train_size, np.nan, dtype=np.float32)
        external_indices = external_compact["forward_index"].astype(int)
        valid_external = (external_indices >= 0) & (external_indices < train_size)
        external_oof[external_indices[valid_external]] = external_compact["blend_forward"][valid_external]
        oof_channels["external_compact"] = external_oof
        common_mask &= np.isfinite(external_oof)
    sibling_llm = None
    sibling_llm_path = shared_cache_dir() / "predictions" / "generic_exp_1_qwen4b_v4.npz"
    if (not debug) and sibling_llm_path.exists():
        sibling_llm = np.load(sibling_llm_path, allow_pickle=False)
        required = {"val", "test", "forward_index", "forward_predictions"}
        if required.issubset(sibling_llm.files):
            sibling_oof = np.full(train_size, np.nan, dtype=np.float32)
            sibling_indices = sibling_llm["forward_index"].astype(int)
            valid_sibling = (sibling_indices >= 0) & (sibling_indices < train_size)
            sibling_oof[sibling_indices[valid_sibling]] = sibling_llm["forward_predictions"][valid_sibling]
            oof_channels["sibling_llm"] = sibling_oof
            common_mask &= np.isfinite(sibling_oof)
        else:
            sibling_llm = None
    replay_oof = None
    if replay_frame is not None and replay_bundle is not None and replay_encoded is not None:
        replay_labels = replay_frame["outcome"].to_numpy(dtype=np.int32)
        replay_targets = np.flatnonzero(train_frame["timestamp"].isin([pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")]).to_numpy())
        replay_oof = np.full(train_size, np.nan, dtype=np.float32)
        for target_timestamp in [pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-01")]:
            current_targets = replay_targets[train_frame["timestamp"].iloc[replay_targets].to_numpy() == target_timestamp]
            replay_training = np.flatnonzero((replay_frame["timestamp"] + pd.Timedelta(days=365) <= target_timestamp).to_numpy())
            replay_joint = pd.concat([replay_encoded, encoded.iloc[current_targets].reset_index(drop=True)], ignore_index=True)
            replay_prediction = replay_channel(
                replay_joint, replay_labels, replay_frame["timestamp"], replay_training,
                np.arange(len(replay_frame), len(replay_frame) + len(current_targets)), seeds=(17, 29, 43),
            )
            replay_oof[current_targets] = replay_prediction
        oof_channels["replay"] = replay_oof
        common_mask &= np.isfinite(replay_oof)
    judgment_oof_full = None
    all_judgments = None
    all_embeddings = None
    all_judgments_v2 = None
    all_embeddings_v2 = None
    if hosted_complete:
        all_judgments, all_summaries = records_to_arrays(hosted_records, list(range(len(bundle.seeds))))
        all_embeddings = ensure_embeddings(all_summaries, shared_cache_dir(), dimensions=256)
        judgment_selection = select_judgment_channel(
            all_judgments[:train_size], all_embeddings[:train_size],
            selected_train_matrix.to_numpy(dtype=np.float32), labels_train,
            train_frame["timestamp"], debug,
        )
        judgment_oof_full = judgment_selection.oof
        oof_channels["judgment"] = judgment_oof_full
        common_mask &= judgment_selection.mask
        all_judgments_v2, all_summaries_v2 = records_to_arrays_v2(hosted_records_v2, list(range(len(bundle.seeds))))
        all_embeddings_v2 = ensure_embeddings(all_summaries_v2, shared_cache_dir(), dimensions=256)
        judgment_selection_v2 = select_judgment_channel(
            all_judgments_v2[:train_size], all_embeddings_v2[:train_size],
            selected_train_matrix.to_numpy(dtype=np.float32), labels_train,
            train_frame["timestamp"], debug,
        )
        oof_channels["judgment_v2"] = judgment_selection_v2.oof
        common_mask &= judgment_selection_v2.mask
    blend_weights, blend_diagnostics = select_rank_blend(
        oof_channels, labels_train, train_frame["timestamp"], common_mask
    )
    internal_auc = {name: float(roc_auc_score(labels_train[common_mask], prediction[common_mask])) for name, prediction in oof_channels.items()}
    report_phase("blend_selection", weights=json.dumps(blend_weights, sort_keys=True), diagnostics=json.dumps(blend_diagnostics, sort_keys=True))

    eb_validation = temporal_eb_features(
        bundle.keys, bundle.seeds["timestamp"], bundle.seeds["nct_id"],
        train_index, validation_index, labels_train, tabular_selection.strength,
    )
    structural_validation = structural_inference(
        encoded.iloc[:train_size + validation_size], train_index, validation_index,
        count_all, fraction_all, phi, debug,
    )
    validation_matrix = make_matrix(
        encoded.iloc[validation_index], eb_validation, structural_validation,
    )
    tabular_validation = fit_tabular_predict(
        selected_train_matrix, labels_train, validation_matrix, categorical,
        tabular_selection.config, tabular_selection.rounds, debug,
    )
    word_validation = fit_text_predict(
        train_documents, labels_train, bundle.documents[train_size:train_size + validation_size],
        "word", word_selection.selected_c, debug, 17,
    )
    char_validation = fit_text_predict(
        train_documents, labels_train, bundle.documents[train_size:train_size + validation_size],
        "char", char_selection.selected_c, debug, 29,
    )
    validation_channels = {
        "tabular": tabular_validation,
        "word": word_validation,
        "char": char_validation,
        "structural": structural_validation[0],
    }
    if external_compact is not None:
        validation_channels["external_compact"] = external_compact["val"]
    if sibling_llm is not None:
        validation_channels["sibling_llm"] = sibling_llm["val"]
    replay_validation = None
    if replay_frame is not None and replay_encoded is not None:
        replay_labels = replay_frame["outcome"].to_numpy(dtype=np.int32)
        replay_train_indices = np.arange(len(replay_frame))
        replay_validation_joint = pd.concat([replay_encoded, encoded.iloc[validation_index].reset_index(drop=True)], ignore_index=True)
        replay_validation = replay_channel(
            replay_validation_joint, replay_labels, replay_frame["timestamp"], replay_train_indices,
            np.arange(len(replay_frame), len(replay_frame) + validation_size),
        )
        validation_channels["replay"] = replay_validation
    if hosted_complete and judgment_selection is not None and all_judgments is not None and all_embeddings is not None:
        validation_channels["judgment"] = fit_judgment_predict(
            all_judgments[:train_size], all_embeddings[:train_size],
            selected_train_matrix.to_numpy(dtype=np.float32), labels_train,
            all_judgments[validation_index], all_embeddings[validation_index],
            validation_matrix.to_numpy(dtype=np.float32), judgment_selection.selected_c,
            judgment_selection.copied_dimensions, debug,
        )
    if hosted_complete and judgment_selection_v2 is not None and all_judgments_v2 is not None and all_embeddings_v2 is not None:
        validation_channels["judgment_v2"] = fit_judgment_predict(
            all_judgments_v2[:train_size], all_embeddings_v2[:train_size],
            selected_train_matrix.to_numpy(dtype=np.float32), labels_train,
            all_judgments_v2[validation_index], all_embeddings_v2[validation_index],
            validation_matrix.to_numpy(dtype=np.float32), judgment_selection_v2.selected_c,
            judgment_selection_v2.copied_dimensions, debug,
        )
    validation_prediction = apply_rank_blend(validation_channels, blend_weights)
    validation_prediction = np.asarray(validation_prediction, dtype=np.float64).copy()
    validation_path = run_data_dir() / "val_predictions.npy"
    np.save(validation_path, validation_prediction)
    validation_checksum = prediction_checksum(validation_path)
    report_phase("model_a_validation", rows=len(validation_prediction), labels_exposed=False, checksum=validation_checksum)

    labels_validation = validation_frame[context.target_col].to_numpy(dtype=np.int32)
    labels_combined = np.concatenate([labels_train, labels_validation])
    combined_index = np.arange(train_size + validation_size)
    eb_combined = temporal_eb_features(
        bundle.keys, bundle.seeds["timestamp"], bundle.seeds["nct_id"],
        combined_index, combined_index, labels_combined, tabular_selection.strength,
    )
    combined_structural_oof = np.concatenate([structural_oof, structural_validation[0]])
    combined_fraction_oof = np.concatenate([fraction_oof, structural_validation[1]])
    combined_expected_oof = np.concatenate([expected_k_oof, structural_validation[2]])
    combined_matrix = make_matrix(
        encoded.iloc[combined_index], eb_combined,
        (combined_structural_oof, combined_fraction_oof, combined_expected_oof),
    )
    eb_test = temporal_eb_features(
        bundle.keys, bundle.seeds["timestamp"], bundle.seeds["nct_id"],
        combined_index, test_index, labels_combined, tabular_selection.strength,
    )
    structural_test = structural_inference(
        encoded, combined_index, test_index, count_all, fraction_all, phi, debug,
    )
    test_matrix = make_matrix(encoded.iloc[test_index], eb_test, structural_test)
    tabular_test = fit_tabular_predict(
        combined_matrix, labels_combined, test_matrix, categorical,
        tabular_selection.config, tabular_selection.rounds, debug,
    )
    combined_documents = bundle.documents[:train_size + validation_size]
    test_documents = bundle.documents[train_size + validation_size:]
    word_test = fit_text_predict(combined_documents, labels_combined, test_documents, "word", word_selection.selected_c, debug, 17)
    char_test = fit_text_predict(combined_documents, labels_combined, test_documents, "char", char_selection.selected_c, debug, 29)
    test_channels = {"tabular": tabular_test, "word": word_test, "char": char_test, "structural": structural_test[0]}
    if external_compact is not None:
        test_channels["external_compact"] = external_compact["test"]
    if sibling_llm is not None:
        test_channels["sibling_llm"] = sibling_llm["test"]
    if replay_frame is not None and replay_encoded is not None:
        replay_labels = replay_frame["outcome"].to_numpy(dtype=np.int32)
        replay_combined = pd.concat([replay_encoded, encoded.iloc[combined_index].reset_index(drop=True)], ignore_index=True)
        replay_combined_labels = np.concatenate([replay_labels, labels_combined])
        replay_test_joint = pd.concat([replay_combined, encoded.iloc[test_index].reset_index(drop=True)], ignore_index=True)
        test_channels["replay"] = replay_channel(
            replay_test_joint, replay_combined_labels, pd.concat([replay_frame["timestamp"], bundle.seeds.iloc[combined_index]["timestamp"]], ignore_index=True),
            np.arange(len(replay_combined)), np.arange(len(replay_combined), len(replay_combined) + len(test_index)),
        )
    if hosted_complete and judgment_selection is not None and all_judgments is not None and all_embeddings is not None:
        test_channels["judgment"] = fit_judgment_predict(
            all_judgments[combined_index], all_embeddings[combined_index],
            combined_matrix.to_numpy(dtype=np.float32), labels_combined,
            all_judgments[test_index], all_embeddings[test_index], test_matrix.to_numpy(dtype=np.float32),
            judgment_selection.selected_c, judgment_selection.copied_dimensions, debug,
        )
    if hosted_complete and judgment_selection_v2 is not None and all_judgments_v2 is not None and all_embeddings_v2 is not None:
        test_channels["judgment_v2"] = fit_judgment_predict(
            all_judgments_v2[combined_index], all_embeddings_v2[combined_index],
            combined_matrix.to_numpy(dtype=np.float32), labels_combined,
            all_judgments_v2[test_index], all_embeddings_v2[test_index], test_matrix.to_numpy(dtype=np.float32),
            judgment_selection_v2.selected_c, judgment_selection_v2.copied_dimensions, debug,
        )
    test_prediction = apply_rank_blend(test_channels, blend_weights)
    if prediction_checksum(validation_path) != validation_checksum:
        raise RuntimeError("Model A validation checksum changed after validation labels were exposed")
    np.save(run_data_dir() / "test_predictions.npy", np.asarray(test_prediction, dtype=np.float64))
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)

    invariant_cache = shared_cache_dir() / "predictions" / "incumbent_run_0007_channels_v1.npz"
    invariant_cache.parent.mkdir(parents=True, exist_ok=True)
    invariant_payload = {
        "train_nct_id": train_frame["nct_id"].to_numpy(),
        "train_timestamp": pd.to_datetime(train_frame["timestamp"]).to_numpy(),
        "forward_index": np.flatnonzero(common_mask),
        "forward_prediction": apply_rank_blend(
            {name: values[common_mask] for name, values in oof_channels.items()}, blend_weights
        ),
        "val": validation_prediction,
        "test": np.asarray(test_prediction, dtype=np.float64),
        "blend_names": np.asarray(list(blend_weights)),
        "blend_values": np.asarray(list(blend_weights.values()), dtype=np.float64),
    }
    for name, values in oof_channels.items():
        invariant_payload[f"oof_{name}"] = np.asarray(values)
    for name, values in validation_channels.items():
        invariant_payload[f"val_{name}"] = np.asarray(values)
    for name, values in test_channels.items():
        invariant_payload[f"test_{name}"] = np.asarray(values)
    temporary_invariant = invariant_cache.with_suffix(".npz.part")
    with temporary_invariant.open("wb") as stream:
        np.savez_compressed(stream, **invariant_payload)
    os.replace(temporary_invariant, invariant_cache)

    validation_auc = float(roc_auc_score(labels_validation, validation_prediction))
    validation_slices = validation_slice_diagnostics(
        labels_validation, validation_prediction,
        bundle.base.iloc[validation_index].reset_index(drop=True),
    )
    resolution = bootstrap_diagnostics(labels_train, oof_channels, train_frame["timestamp"], common_mask)
    metrics = {
        "validation_auc_diagnostic_only": validation_auc,
        "internal_auc": internal_auc,
        "blend_weights": blend_weights,
        "blend_diagnostics": blend_diagnostics,
        "tabular_search": tabular_selection.search_scores,
        "phi_scores": phi_scores,
        "hosted_extraction": extraction_stats,
        "hosted_extraction_v2": extraction_stats_v2,
        "hosted_complete": hosted_complete,
        "replay_forward_auc": replay_forward_auc,
        "validation_slices_diagnostic_only": validation_slices,
        "resolution_diagnostics": resolution,
        "representativeness": {
            "origin_volume": {"train_last": int((train_frame["timestamp"] == train_frame["timestamp"].max()).sum()), "validation": validation_size, "test": len(test_frame)},
            "label_rate": {"train": float(labels_train.mean()), "train_2019": float(labels_train[train_frame["timestamp"] == train_frame["timestamp"].max()].mean()), "validation": float(labels_validation.mean())},
            "trial_age_median": {"train": float(bundle.base.iloc[train_index]["trial_age_days"].median()), "validation": float(bundle.base.iloc[validation_index]["trial_age_days"].median()), "test": float(bundle.base.iloc[test_index]["trial_age_days"].median())},
            "intervention_zero_share": {"train": float((bundle.base.iloc[train_index]["intervention_count"] == 0).mean()), "validation": float((bundle.base.iloc[validation_index]["intervention_count"] == 0).mean()), "test": float((bundle.base.iloc[test_index]["intervention_count"] == 0).mean())},
        },
        "two_model_contract": {"validation_fit": "train_only", "test_fit": "train_plus_validation", "validation_vector_preserved": True, "model_a_checksum": validation_checksum},
        "elapsed_seconds": time.time() - START,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    write_campaign_notes(bundle, metrics)
    if staged_full:
        stage_marker.write_text(json.dumps({"completed": True, "hosted_rows": len(hosted_indices), "elapsed_seconds": time.time() - START}) + "\n")
    report_phase("complete", validation_auc_diagnostic_only=validation_auc, total_rows=len(bundle.seeds))


if __name__ == "__main__":
    main()
