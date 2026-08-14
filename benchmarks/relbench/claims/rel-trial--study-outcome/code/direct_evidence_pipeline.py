# Imports

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

import publication_pipeline as publication
from build_cross_lane_ensemble import rank_blend
from campaign_io import locked_append, register_artifact
from kapso_datasets.common import shared_cache_dir
from publication_evidence import adjudicate_candidates
from registry_clock import DIRECT_EVIDENCE_VERSION, snapshot_direct_evidence


# Configuration

START = time.time()
POSITIVE_POLE = 0.995
NEGATIVE_POLE = 0.005
BASELINE_SCORE = 0.7500927005570969
BASELINE_VALIDATION_SHA = "f257c648a44fa3c6b4a178f9ffec8c21fdc8318be0ff69c46cd744cbf7a50246"
BASELINE_TEST_SHA = "f6261ae077f247a2bf8f518b490617f7dd4b279652234a0d66cf78ab990d79cb"
ORIGIN_NAMES = {
    "official_2017": pd.Timestamp("2017-01-01"),
    "official_2018": pd.Timestamp("2018-01-01"),
    "official_2019": pd.Timestamp("2019-01-01"),
}


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[direct-evidence] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _auc(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if mask is not None:
        labels = labels[mask]
        prediction = prediction[mask]
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, prediction))


def _ap(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    labels = np.asarray(labels)
    prediction = np.asarray(prediction)
    if mask is not None:
        labels = labels[mask]
        prediction = prediction[mask]
    return float(average_precision_score(labels, prediction)) if len(labels) else float("nan")


def _route(baseline: np.ndarray, evidence: pd.DataFrame, maximum_rows: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(baseline, dtype=np.float64).copy()
    routed = np.flatnonzero(~evidence["direct_abstain"].to_numpy(dtype=bool))
    if maximum_rows is not None:
        routed = routed[:maximum_rows]
    positive = evidence["direct_positive"].to_numpy(dtype=bool)
    negative = evidence["direct_complete_negative"].to_numpy(dtype=bool)
    selected = np.zeros(len(result), dtype=bool)
    selected[routed] = True
    result[selected & positive] = POSITIVE_POLE
    result[selected & negative] = NEGATIVE_POLE
    return np.clip(result, 0.0, 1.0), selected


def _archive_run() -> tuple[Path, np.ndarray, np.ndarray, dict[str, Any]]:
    work_dir = Path(os.environ["RELBENCH_WORK_DIR"])
    run_dir = work_dir / "runs" / "run_0005"
    manifest_line = (run_dir / "manifest.txt").read_text().strip()
    manifest = json.loads(manifest_line.split(" ", 1)[1])
    selection = json.loads((run_dir / "private" / "selection.json").read_text())
    validation_path = run_dir / "val_predictions.npy"
    test_path = run_dir / "test_predictions.npy"
    validation_sha = file_sha256(validation_path)
    test_sha = file_sha256(test_path)
    if float(manifest["score"]) != BASELINE_SCORE:
        raise RuntimeError(f"run_0005 score changed: {manifest['score']}")
    if validation_sha != BASELINE_VALIDATION_SHA or test_sha != BASELINE_TEST_SHA:
        raise RuntimeError("run_0005 stored prediction hashes changed")
    if selection.get("status") != "final":
        raise RuntimeError(f"run_0005 is not banked: {selection}")
    validation = np.load(validation_path, allow_pickle=False).astype(np.float64)
    test = np.load(test_path, allow_pickle=False).astype(np.float64)
    diagnostics = {
        "run": manifest["run"],
        "score": float(manifest["score"]),
        "evaluator_id": manifest["evaluator_id"],
        "validation_sha256": validation_sha,
        "test_sha256": test_sha,
        "selection_status": selection["status"],
    }
    return run_dir, validation, test, diagnostics


# Evidence cache

def _evidence_content_key(linkage: pd.DataFrame, projected_root: Path) -> str:
    digest = hashlib.sha256(DIRECT_EVIDENCE_VERSION.encode())
    columns = ["row_id", "nct_id", "timestamp", "split", "external_nct_id", "linked"]
    digest.update(linkage[columns].to_json(orient="records", date_format="iso").encode())
    for metadata_path in sorted(projected_root.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        digest.update(str(metadata.get("snapshot_date", "")).encode())
        digest.update(str(metadata.get("archive_sha256", "")).encode())
    return digest.hexdigest()


def _coverage(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "linked": int(frame["linked"].sum()),
        "linkage": float(frame["linked"].mean()) if len(frame) else 0.0,
        "qualifying": int(frame["min_qualifying_pvalue"].notna().sum()),
        "positive": int(frame["direct_positive"].sum()),
        "complete_negative": int(frame["direct_complete_negative"].sum()),
        "abstention": int(frame["direct_abstain"].sum()),
        "incomplete_qualifying": int((frame["min_qualifying_pvalue"].notna() & frame["direct_abstain"]).sum()),
        "snapshot_eligible": bool(frame["snapshot_eligible"].all()) if len(frame) else False,
        "snapshot_date": str(frame["snapshot_date"].iloc[0]) if len(frame) else "",
        "archive_sha": str(frame["archive_sha"].iloc[0]) if len(frame) else "",
    }


def build_evidence_cache(cache: Path, linkage: pd.DataFrame, projected_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = cache / DIRECT_EVIDENCE_VERSION
    evidence_path = root / "evidence.parquet"
    metadata_path = root / "metadata.json"
    content_key = _evidence_content_key(linkage, projected_root)
    if evidence_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("content_key") == content_key and metadata.get("version") == DIRECT_EVIDENCE_VERSION:
            evidence = pd.read_parquet(evidence_path)
            report("evidence_cache", state="hit", rows=len(evidence), content_key=content_key)
            return evidence, metadata
    parts = []
    for split, current in linkage.groupby("split", sort=False):
        origin_values = pd.to_datetime(current["timestamp"]).dt.normalize().unique()
        if len(origin_values) != 1:
            raise RuntimeError(f"Direct-evidence split spans origins: {split}")
        part = snapshot_direct_evidence(current, projected_root, pd.Timestamp(origin_values[0]))
        part["split"] = split
        parts.append(part)
        report("snapshot", split=split, diagnostics=json.dumps(_coverage(part), sort_keys=True))
    evidence = pd.concat(parts, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    diagnostics = {split: _coverage(current) for split, current in evidence.groupby("split", sort=False)}
    metadata = {
        "version": DIRECT_EVIDENCE_VERSION,
        "content_key": content_key,
        "rows": int(len(evidence)),
        "coverage": diagnostics,
        "build_seconds": time.time() - START,
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = evidence_path.with_suffix(".parquet.part")
    evidence.to_parquet(temporary, index=False)
    os.replace(temporary, evidence_path)
    temporary_metadata = metadata_path.with_suffix(".json.part")
    temporary_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_metadata, metadata_path)
    report("evidence_cache", state="built", rows=len(evidence), content_key=content_key)
    return evidence, metadata


# Historical gate

def _historical_baseline(cache: Path) -> dict[str, np.ndarray]:
    path = cache / DIRECT_EVIDENCE_VERSION / "historical_run0005_baseline.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            result = {name: stored[name].copy() for name in stored.files}
        report("historical_baseline", state="hit", path=path)
        return result
    artifacts = publication._load_artifacts(cache)
    publication.artifacts_global = artifacts
    registry_predictions, _ = publication._registry_predictions_and_ablations(artifacts, run_ablations=False)
    aligned = publication.build_aligned_stacker(artifacts, registry_predictions)
    records, candidates, contexts, _ = publication.retrieve_gate_origins(artifacts, cache)
    adjudications = {}
    for split in ["official_2018", "official_2019"]:
        adjudications[split], diagnostics = adjudicate_candidates(candidates[split], contexts[split], cache, concurrency=32)
        report("historical_adjudication", split=split, calls=diagnostics["calls"], cache_hits=diagnostics["cache_hits"])
    gate = publication.fit_publication_gate(artifacts, aligned, registry_predictions, records, adjudications)
    binary_deltas = [
        _auc(aligned["labels_2018"], gate["binary_2018"]) - _auc(aligned["labels_2018"], gate["candidate_2018"]),
        _auc(aligned["labels_2019"], gate["binary_2019"]) - _auc(aligned["labels_2019"], gate["candidate_2019"]),
    ]
    if min(binary_deltas) < 0:
        raise RuntimeError(f"Stored run_0005 binary route cannot be reconstructed: {binary_deltas}")
    with np.load(cache / "predictions" / "generic_exp_1_tfidf_v1.npz", allow_pickle=False) as text:
        if not bool(json.loads(str(text["diagnostics_json"][0]))["forward_gate"]):
            raise RuntimeError("The run_0005 TF-IDF channel is not forward accepted")
        forward_indices = text["forward_index"].astype(np.int64)
        expected = np.concatenate([
            np.flatnonzero(artifacts["invariant"]["train_timestamp"] == np.datetime64("2018-01-01")),
            np.flatnonzero(artifacts["invariant"]["train_timestamp"] == np.datetime64("2019-01-01")),
        ])
        if not np.array_equal(forward_indices, expected):
            raise RuntimeError("The run_0005 TF-IDF forward rows changed")
        count_2018 = len(aligned["labels_2018"])
        text_2018 = text["forward_predictions"][:count_2018].astype(np.float64)
        text_2019 = text["forward_predictions"][count_2018:].astype(np.float64)
    with np.load(cache / "predictions" / "generic_exp_0_literature_binary_tfidf_v1.npz", allow_pickle=False) as bank:
        bank_diagnostics = json.loads(str(bank["diagnostics_json"][0]))
    weight = float(bank_diagnostics["selected_tfidf_weight"])
    if not bool(bank_diagnostics.get("cross_ensemble_accepted")):
        raise RuntimeError("The archived run_0005 cross-family ensemble is not accepted")
    prediction_2018 = rank_blend(gate["binary_2018"], text_2018, weight)
    prediction_2019 = rank_blend(gate["binary_2019"], text_2019, weight)
    timestamps = artifacts["invariant"]["train_timestamp"]
    indices_2017 = np.flatnonzero(timestamps == np.datetime64("2017-01-01"))
    matrix_2017 = {
        name: artifacts["invariant"][f"oof_{name}"][indices_2017]
        for name in ["tabular", "word", "char", "structural"]
    }
    prediction_2017 = publication._invariant_blend(matrix_2017)
    result = {
        "official_2017_nct_id": artifacts["invariant"]["train_nct_id"][indices_2017].astype(np.int64),
        "official_2017_labels": artifacts["invariant"]["train_labels"][indices_2017].astype(np.int8),
        "official_2017_prediction": prediction_2017,
        "official_2017_literature_covered": np.zeros(len(indices_2017), dtype=bool),
        "official_2018_nct_id": artifacts["invariant"]["train_nct_id"][expected[:len(prediction_2018)]].astype(np.int64),
        "official_2018_labels": aligned["labels_2018"].astype(np.int8),
        "official_2018_prediction": prediction_2018,
        "official_2018_literature_covered": gate["covered_2018"].astype(bool),
        "official_2019_nct_id": artifacts["invariant"]["train_nct_id"][expected[len(prediction_2018):]].astype(np.int64),
        "official_2019_labels": aligned["labels_2019"].astype(np.int8),
        "official_2019_prediction": prediction_2019,
        "official_2019_literature_covered": gate["covered_2019"].astype(bool),
        "selected_tfidf_weight": np.asarray([weight], dtype=np.float64),
    }
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result)
    os.replace(temporary, path)
    report("historical_baseline", state="built", path=path, weight=weight)
    return result


def _clustered_bootstrap(origins: list[dict[str, np.ndarray]], draws: int = 2000) -> dict[str, float]:
    random = np.random.default_rng(1337)
    deltas = []
    for _ in range(draws):
        origin_deltas = []
        origin_weights = []
        for origin in origins:
            labels = origin["labels"]
            groups = origin["nct_id"]
            unique_groups = np.unique(groups)
            selected_groups = random.choice(unique_groups, size=len(unique_groups), replace=True)
            group_rows = {group: np.flatnonzero(groups == group) for group in unique_groups}
            sampled = np.concatenate([group_rows[group] for group in selected_groups])
            if len(np.unique(labels[sampled])) < 2:
                continue
            delta = roc_auc_score(labels[sampled], origin["candidate"][sampled]) - roc_auc_score(labels[sampled], origin["baseline"][sampled])
            origin_deltas.append(float(delta))
            origin_weights.append(len(labels))
        if origin_deltas:
            deltas.append(float(np.average(origin_deltas, weights=origin_weights)))
    values = np.asarray(deltas, dtype=np.float64)
    return {
        "draws": int(len(values)),
        "mean_delta": float(values.mean()),
        "standard_error": float(values.std(ddof=1)),
        "probability_positive": float((values > 0).mean()),
        "lower_10": float(np.quantile(values, 0.10)),
        "upper_90": float(np.quantile(values, 0.90)),
        "cluster": "nct_id_within_origin",
    }


def _gate_origin(name: str, baseline: np.ndarray, labels: np.ndarray, nct_id: np.ndarray, evidence: pd.DataFrame, literature_covered: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    candidate, routed = _route(baseline, evidence)
    if not np.array_equal(candidate[~routed], baseline[~routed]):
        raise RuntimeError(f"No-evidence slice changed for {name}")
    diagnostics = {
        **_coverage(evidence),
        "routed": int(routed.sum()),
        "literature_overlap": int((routed & literature_covered).sum()),
        "incumbent_auc": _auc(labels, baseline),
        "candidate_auc": _auc(labels, candidate),
        "delta": _auc(labels, candidate) - _auc(labels, baseline),
        "covered_incumbent_auc": _auc(labels, baseline, routed),
        "covered_candidate_auc": _auc(labels, candidate, routed),
        "covered_incumbent_ap": _ap(labels, baseline, routed),
        "covered_candidate_ap": _ap(labels, candidate, routed),
        "covered_label_rate": float(labels[routed].mean()) if routed.any() else float("nan"),
        "prediction_rank_correlation": float(spearmanr(baseline, candidate).statistic),
        "no_evidence_unchanged": bool(np.array_equal(candidate[~routed], baseline[~routed])),
    }
    arrays = {"labels": labels, "nct_id": nct_id, "baseline": baseline, "candidate": candidate}
    return diagnostics, arrays


def historical_gate(cache: Path, evidence: pd.DataFrame, linkage: pd.DataFrame, projected_root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    historical = _historical_baseline(cache)
    gate_reports = {}
    bootstrap_origins = []
    candidates = {}
    official_2017 = pd.DataFrame({
        "row_id": np.arange(len(historical["official_2017_nct_id"]), dtype=np.int64),
        "nct_id": historical["official_2017_nct_id"],
        "timestamp": pd.Timestamp("2017-01-01"),
        "split": "official_2017",
        "external_nct_id": np.nan,
        "linked": False,
    })
    evidence_2017 = snapshot_direct_evidence(official_2017, projected_root, ORIGIN_NAMES["official_2017"])
    evidence_by_name = {
        "official_2017": evidence_2017,
        "official_2018": evidence[evidence["split"].eq("official_2018")].reset_index(drop=True),
        "official_2019": evidence[evidence["split"].eq("official_2019")].reset_index(drop=True),
    }
    for name in ORIGIN_NAMES:
        current = evidence_by_name[name]
        expected_nct = historical[f"{name}_nct_id"]
        if not np.array_equal(current["nct_id"].to_numpy(dtype=np.int64), expected_nct):
            raise RuntimeError(f"Historical direct-evidence alignment failed for {name}")
        diagnostics, arrays = _gate_origin(
            name,
            historical[f"{name}_prediction"].astype(np.float64),
            historical[f"{name}_labels"].astype(np.int8),
            expected_nct,
            current,
            historical[f"{name}_literature_covered"].astype(bool),
        )
        gate_reports[name] = diagnostics
        candidates[name] = arrays["candidate"]
        bootstrap_origins.append(arrays)
        report("origin_gate", origin=name, diagnostics=json.dumps(diagnostics, sort_keys=True, allow_nan=True))
    bootstrap = _clustered_bootstrap(bootstrap_origins, draws=2000)
    available = [values for values in gate_reports.values() if values["snapshot_eligible"]]
    accepted = bool(available and all(values["delta"] >= 0 for values in available) and bootstrap["probability_positive"] >= 0.8)
    diagnostics = {
        "accepted": accepted,
        "origins": gate_reports,
        "bootstrap": bootstrap,
        "positive_pole": POSITIVE_POLE,
        "negative_pole": NEGATIVE_POLE,
        "fixed_without_label_tuning": True,
    }
    report("historical_gate", diagnostics=json.dumps(diagnostics, sort_keys=True, allow_nan=True))
    return diagnostics, candidates


# Candidate

def _persist_candidate(path: Path, payload: dict[str, np.ndarray], diagnostics: dict[str, Any]) -> None:
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload, diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, path)


def build_candidate(debug: bool = False) -> Path:
    cache = shared_cache_dir()
    run_dir, baseline_validation, baseline_test, archive = _archive_run()
    feature_root = cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2"
    linkage = pd.read_parquet(feature_root / "linkage.parquet")
    projected_root = cache / "registry_clock_lane0" / "projected"
    evidence, evidence_metadata = build_evidence_cache(cache, linkage, projected_root)
    gate, historical_candidates = historical_gate(cache, evidence, linkage, projected_root)
    if not gate["accepted"]:
        raise RuntimeError(f"Snapshot direct-evidence gate rejected: {json.dumps(gate, sort_keys=True, allow_nan=True)}")
    validation_evidence = evidence[evidence["split"].eq("validation_2020")].reset_index(drop=True)
    test_evidence = evidence[evidence["split"].eq("test_2021")].reset_index(drop=True)
    invariant_path = cache / "predictions" / "generic_exp_0_invariant_channels_v1.npz"
    with np.load(invariant_path, allow_pickle=False) as invariant:
        if not np.array_equal(validation_evidence["nct_id"].to_numpy(dtype=np.int64), invariant["validation_nct_id"].astype(np.int64)):
            raise RuntimeError("Validation direct-evidence rows are not in task order")
        if not np.array_equal(test_evidence["nct_id"].to_numpy(dtype=np.int64), invariant["test_nct_id"].astype(np.int64)):
            raise RuntimeError("Test direct-evidence rows are not in task order")
    maximum_rows = 16 if debug else None
    validation, validation_routed = _route(baseline_validation, validation_evidence, maximum_rows)
    freeze_path = cache / DIRECT_EVIDENCE_VERSION / "model_a" / ("debug_validation_predictions.npy" if debug else "validation_predictions.npy")
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(freeze_path, validation.astype(np.float64))
    validation_checksum = file_sha256(freeze_path)
    report("model_a_saved", checksum=validation_checksum, routed=int(validation_routed.sum()), validation_labels_loaded=False)
    test, test_routed = _route(baseline_test, test_evidence, maximum_rows)
    if file_sha256(freeze_path) != validation_checksum:
        raise RuntimeError("Snapshot direct Model-A checksum changed during Model-B materialization")
    if file_sha256(run_dir / "val_predictions.npy") != BASELINE_VALIDATION_SHA or file_sha256(run_dir / "test_predictions.npy") != BASELINE_TEST_SHA:
        raise RuntimeError("Archived run_0005 changed during candidate construction")
    if not np.array_equal(validation[~validation_routed], baseline_validation[~validation_routed]):
        raise RuntimeError("Validation abstention rows changed")
    if not np.array_equal(test[~test_routed], baseline_test[~test_routed]):
        raise RuntimeError("Test abstention rows changed")
    validation_summary = _coverage(validation_evidence)
    validation_summary["routed"] = int(validation_routed.sum())
    validation_summary["no_evidence_unchanged"] = True
    test_summary = _coverage(test_evidence)
    test_summary["routed"] = int(test_routed.sum())
    test_summary["no_evidence_unchanged"] = True
    diagnostics = {
        "accepted": True,
        "variant": "snapshot_direct_debug" if debug else "snapshot_direct",
        "archive_baseline": archive,
        "evidence": evidence_metadata,
        "historical_gate": gate,
        "validation": validation_summary,
        "test": test_summary,
        "validation_checksum": validation_checksum,
        "validation_prediction_label_fit": "archived_run_0005_model_a_plus_fixed_snapshot_expert_without_validation_labels",
        "test_prediction_label_fit": "archived_run_0005_model_b_plus_fixed_snapshot_expert",
        "debug_evidence_limit": 16 if debug else None,
        "elapsed_seconds": time.time() - START,
    }
    candidate_name = "generic_exp_2_snapshot_direct_debug_v1.npz" if debug else "generic_exp_2_snapshot_direct_v1.npz"
    candidate_path = cache / "predictions" / candidate_name
    payload = {
        "val": validation.astype(np.float64),
        "test": test.astype(np.float64),
        "validation_expert_probability": validation_evidence["expert_probability"].to_numpy(dtype=np.float64),
        "test_expert_probability": test_evidence["expert_probability"].to_numpy(dtype=np.float64),
        "validation_routed": validation_routed,
        "test_routed": test_routed,
        "validation_nct_id": validation_evidence["nct_id"].to_numpy(dtype=np.int64),
        "test_nct_id": test_evidence["nct_id"].to_numpy(dtype=np.int64),
        "historical_2018": historical_candidates["official_2018"],
        "historical_2019": historical_candidates["official_2019"],
    }
    _persist_candidate(candidate_path, payload, diagnostics)
    if not debug:
        register_artifact(cache, {
            "name": "generic_exp_2 SHA-verified snapshot direct-evidence candidate",
            "path": f"predictions/{candidate_name}",
            "description": "Exact pre-origin primary p-value expert over immutable run_0005 Model A/B predictions with conservative complete-negative abstention and three-origin gate diagnostics.",
            "content_key": f"rel-trial-study-outcome:{DIRECT_EVIDENCE_VERSION}:{evidence_metadata['content_key']}",
            "rebuild_hint": "Run direct_evidence_pipeline.py from the projected six-snapshot registry cache and banked run_0005 archive.",
        })
        memory_marker = cache / DIRECT_EVIDENCE_VERSION / "campaign_memory_recorded"
        if not memory_marker.exists():
            locked_append(cache / "features_history.md", f'''\n### SHA-verified target-trial primary p-value expert
- run/experiment: generic_exp_2 lane 0 | status: TESTED-KEPT
- what: Exact snapshot outcomes-to-analyses join with official modifier/range filters, fixed 0.995/0.005 poles, complete-negative rule, and run_0005 preservation on abstentions.
- outcome: gate {json.dumps(gate, sort_keys=True, allow_nan=True)}; validation-era coverage {json.dumps(validation_summary, sort_keys=True)}; test-era coverage {json.dumps(test_summary, sort_keys=True)}.
- takeaway: direct target evidence dominates weaker routing where available; unlinked, incomplete, and date-invalid records retain run_0005 byte-for-byte.
''')
            locked_append(cache / "table_information.md", f'''\n### 2026-08-14 SHA-verified snapshot direct evidence
- Snapshot `outcomes.id` joins `outcome_analyses.outcome_id`; only Primary rows and p-values in [0,1] with null/non-`>` modifiers qualify.
- Every projected archive SHA is checked against its declared pin and strictly precedes its seed origin; results submission, QC, and posting dates contribute only when strictly pre-origin.
- Conservative negatives require a qualifying analysis for every registered/reported primary endpoint plus completed results reporting. Coverage by split: {json.dumps(evidence_metadata['coverage'], sort_keys=True)}.
''')
            memory_marker.write_text("recorded\n")
    report("candidate", path=candidate_path, validation_checksum=validation_checksum, validation_routed=int(validation_routed.sum()), test_routed=int(test_routed.sum()))
    return candidate_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    build_candidate(arguments.debug)


if __name__ == "__main__":
    main()
