# Imports

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from campaign_io import locked_append, register_artifact
from jats_evidence import EXTRACTOR_VERSION, aggregate_trial_verdicts, extract_origin, save_recon_fixtures
from kapso_datasets.common import load_task, run_data_dir, shared_cache_dir
from publication_evidence import build_trial_contexts, retrieve_origin
from replay_features import generate_july_replay


# Configuration

START = time.time()
POSITIVE_POLE = 0.995
NEGATIVE_POLE = 0.005
FALLBACK_SCORE = 0.7632025697041176
FALLBACK_VALIDATION_SHA = "c34c8481d20a1cf5adbb11a466aa7f4ab203bf05bf1b073c265cceee7019be3b"
FALLBACK_TEST_SHA = "d78b03000fd5e8bc329e83939c4205238eb55a27595a38f0003db1eff75944d8"
CANDIDATE_NAME = "generic_exp_4_jats_direct_v1.npz"
ORIGINS = {
    "replay_2017": pd.Timestamp("2017-07-01"),
    "official_2018": pd.Timestamp("2018-01-01"),
    "replay_2018": pd.Timestamp("2018-07-01"),
    "official_2019": pd.Timestamp("2019-01-01"),
    "validation_2020": pd.Timestamp("2020-01-01"),
    "test_2021": pd.Timestamp("2021-01-01"),
}


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[jats-direct] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n")
    os.replace(temporary, path)


# Fallback

def banked_fallback() -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    run_root = Path(os.environ["RELBENCH_WORK_DIR"]) / "runs" / "run_0006"
    validation_path = run_root / "val_predictions.npy"
    test_path = run_root / "test_predictions.npy"
    selection = json.loads((run_root / "private" / "selection.json").read_text())
    manifest = json.loads((run_root / "manifest.txt").read_text().split(" ", 1)[1])
    validation_hash = file_sha256(validation_path)
    test_hash = file_sha256(test_path)
    if selection.get("status") != "final":
        raise RuntimeError(f"run_0006 is not finalized: {selection}")
    if float(manifest["score"]) != FALLBACK_SCORE:
        raise RuntimeError(f"run_0006 score changed: {manifest['score']}")
    if validation_hash != FALLBACK_VALIDATION_SHA or test_hash != FALLBACK_TEST_SHA:
        raise RuntimeError("run_0006 prediction hashes changed")
    cache_root = shared_cache_dir() / "jats_endpoint_evidence_v1" / "fallback"
    cache_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(validation_path, cache_root / "val_predictions.npy")
    shutil.copy2(test_path, cache_root / "test_predictions.npy")
    diagnostics = {
        "run": "run_0006",
        "score": float(manifest["score"]),
        "status": selection["status"],
        "validation_sha256": validation_hash,
        "test_sha256": test_hash,
    }
    report("fallback", diagnostics=json.dumps(diagnostics, sort_keys=True))
    return np.load(validation_path, allow_pickle=False).astype(np.float64), np.load(test_path, allow_pickle=False).astype(np.float64), diagnostics


# Evidence

def _origin_records(split: str, linkage: pd.DataFrame, cache: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    origin = ORIGINS[split]
    return retrieve_origin(linkage, origin, cache)


def load_or_extract_origin(
    split: str,
    linkage: pd.DataFrame,
    projected_root: Path,
    cache: Path,
    debug: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = cache / "jats_endpoint_evidence_v1" / "origins" / split
    documents_path = root / "documents.parquet"
    facts_path = root / "facts.parquet"
    diagnostics_path = root / "diagnostics.json"
    if not debug and documents_path.exists() and facts_path.exists() and diagnostics_path.exists():
        documents = pd.read_parquet(documents_path)
        facts = pd.read_parquet(facts_path)
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["cache_hit"] = True
        report("origin_cache", split=split, documents=len(documents), facts=len(facts))
        return documents, facts, diagnostics
    records, retrieval = _origin_records(split, linkage, cache)
    contexts = build_trial_contexts(linkage, ORIGINS[split], projected_root)
    maximum = 16 if debug else None
    documents, facts, extraction, fixtures = extract_origin(records, contexts, cache, workers=4 if debug else 36, maximum_documents=maximum)
    fixture_diagnostics = save_recon_fixtures(fixtures, root / "fixtures.json", maximum=16 if debug else 50)
    diagnostics = {
        "split": split,
        "origin": ORIGINS[split].strftime("%Y-%m-%d"),
        "retrieval": retrieval,
        "extraction": extraction,
        "fixtures": fixture_diagnostics,
        "cache_hit": False,
    }
    if not debug:
        atomic_parquet(documents, documents_path)
        atomic_parquet(facts, facts_path)
        atomic_json(diagnostics, diagnostics_path)
    report("origin_extracted", split=split, diagnostics=json.dumps(diagnostics, sort_keys=True))
    return documents, facts, diagnostics


def aligned_verdicts(split: str, linkage: pd.DataFrame, documents: pd.DataFrame, direct: pd.DataFrame) -> pd.DataFrame:
    rows = linkage[linkage["split"].eq(split)].copy().reset_index(drop=True)
    direct_rows = direct[direct["split"].eq(split)].copy().reset_index(drop=True)
    if not np.array_equal(rows["nct_id"].to_numpy(dtype=np.int64), direct_rows["nct_id"].to_numpy(dtype=np.int64)):
        raise RuntimeError(f"Direct-evidence alignment failed for {split}")
    prior = direct_rows.drop_duplicates("external_nct_id", keep="first") if "external_nct_id" in direct_rows else None
    trial = aggregate_trial_verdicts(documents, prior)
    columns = ["external_nct_id", "jats_positive", "jats_complete_negative", "jats_abstain", "jats_conflict", "jats_document_count"]
    aligned = rows.merge(trial[columns], on="external_nct_id", how="left")
    for name in ["jats_positive", "jats_complete_negative", "jats_conflict"]:
        aligned[name] = aligned[name].eq(True)
    aligned["jats_abstain"] = ~aligned["jats_abstain"].eq(False)
    aligned["jats_document_count"] = aligned["jats_document_count"].fillna(0).astype(int)
    aligned["existing_direct"] = ~direct_rows["direct_abstain"].to_numpy(dtype=bool)
    aligned["new_route"] = (~aligned["existing_direct"]) & (~aligned["jats_abstain"]) & (~aligned["jats_conflict"])
    aligned["route_positive"] = aligned["new_route"] & aligned["jats_positive"]
    aligned["route_negative"] = aligned["new_route"] & aligned["jats_complete_negative"]
    return aligned


# Historical labels

def labels_by_split() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    context = load_task()
    train = context.train.df.copy().reset_index(drop=True)
    validation_ids = set(context.val.df["nct_id"].astype(int))
    test_ids = set(context.test.df["nct_id"].astype(int))
    replay = generate_july_replay(context.db, start_year=2017, end_year=2018)
    replay = replay[~replay["nct_id"].astype(int).isin(validation_ids | test_ids)].reset_index(drop=True)
    result = {}
    for year in [2017, 2018]:
        frame = replay[pd.to_datetime(replay["timestamp"]).eq(pd.Timestamp(f"{year}-07-01"))]
        result[f"replay_{year}"] = (frame["nct_id"].to_numpy(dtype=np.int64), frame["outcome"].to_numpy(dtype=np.int8))
    for year in [2018, 2019]:
        frame = train[pd.to_datetime(train["timestamp"]).eq(pd.Timestamp(f"{year}-01-01"))]
        result[f"official_{year}"] = (frame["nct_id"].to_numpy(dtype=np.int64), frame[context.target_col].to_numpy(dtype=np.int8))
    return result


# Gate

def auc(labels: np.ndarray, prediction: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        labels = labels[mask]
        prediction = prediction[mask]
    if len(labels) < 2 or len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, prediction))


def apply_routes(baseline: np.ndarray, verdicts: pd.DataFrame, positive: float = POSITIVE_POLE, negative: float = NEGATIVE_POLE) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(baseline, dtype=np.float64).copy()
    routed = verdicts["new_route"].to_numpy(dtype=bool)
    result[verdicts["route_positive"].to_numpy(dtype=bool)] = positive
    result[verdicts["route_negative"].to_numpy(dtype=bool)] = negative
    return result, routed


def clustered_bootstrap(origins: list[dict[str, np.ndarray]], draws: int = 2000) -> dict[str, float]:
    random = np.random.default_rng(1337)
    values = []
    for _ in range(draws):
        deltas = []
        weights = []
        for origin in origins:
            groups = np.unique(origin["nct_id"])
            sampled_groups = random.choice(groups, len(groups), replace=True)
            positions = {value: np.flatnonzero(origin["nct_id"] == value) for value in groups}
            indices = np.concatenate([positions[value] for value in sampled_groups])
            if len(np.unique(origin["labels"][indices])) < 2:
                continue
            deltas.append(auc(origin["labels"][indices], origin["candidate"][indices]) - auc(origin["labels"][indices], origin["baseline"][indices]))
            weights.append(len(indices))
        if deltas:
            values.append(float(np.average(deltas, weights=weights)))
    array = np.asarray(values, dtype=np.float64)
    return {
        "draws": len(array),
        "mean_delta": float(array.mean()),
        "standard_error": float(array.std(ddof=1)),
        "probability_positive": float((array > 0).mean()),
        "lower_10": float(np.quantile(array, 0.10)),
        "upper_90": float(np.quantile(array, 0.90)),
        "cluster": "origin_and_nct_id",
    }


def historical_gate(verdicts: dict[str, pd.DataFrame], snapshot: np.lib.npyio.NpzFile) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = labels_by_split()
    baselines = {
        "official_2018": snapshot["historical_2018"].astype(np.float64),
        "official_2019": snapshot["historical_2019"].astype(np.float64),
    }
    reports = {}
    candidates = {}
    bootstrap_origins = []
    for split in ["replay_2017", "official_2018", "replay_2018", "official_2019"]:
        nct_id, current_labels = labels[split]
        if not np.array_equal(nct_id, verdicts[split]["nct_id"].to_numpy(dtype=np.int64)):
            raise RuntimeError(f"Historical label alignment failed for {split}")
        if split in baselines:
            baseline = baselines[split]
        else:
            baseline = np.full(len(current_labels), 0.5, dtype=np.float64)
        candidate, routed = apply_routes(baseline, verdicts[split])
        if not np.array_equal(candidate[~routed], baseline[~routed]):
            raise RuntimeError(f"Abstentions changed for {split}")
        route_labels = current_labels[routed]
        route_accuracy = float(np.mean(route_labels == verdicts[split].loc[routed, "route_positive"].to_numpy(dtype=bool))) if routed.any() else float("nan")
        report_row = {
            "rows": len(current_labels),
            "label_rate": float(current_labels.mean()),
            "routed": int(routed.sum()),
            "positive_routes": int(verdicts[split]["route_positive"].sum()),
            "negative_routes": int(verdicts[split]["route_negative"].sum()),
            "route_accuracy": route_accuracy,
            "baseline_auc": auc(current_labels, baseline),
            "candidate_auc": auc(current_labels, candidate),
            "delta": auc(current_labels, candidate) - auc(current_labels, baseline),
            "routed_auc": auc(current_labels, candidate, routed),
            "routed_average_precision": float(average_precision_score(current_labels[routed], candidate[routed])) if routed.any() and len(np.unique(current_labels[routed])) > 1 else float("nan"),
            "no_evidence_unchanged": True,
            "prediction_rank_correlation": float(spearmanr(baseline, candidate).statistic) if np.std(baseline) > 0 else float("nan"),
            "baseline_kind": "run_0006_historical" if split in baselines else "uninformative_replay_control",
        }
        reports[split] = report_row
        candidates[split] = candidate
        if split in baselines and routed.any():
            bootstrap_origins.append({"labels": current_labels, "nct_id": nct_id, "baseline": baseline, "candidate": candidate})
        report("gate_origin", split=split, diagnostics=json.dumps(report_row, sort_keys=True, allow_nan=True))
    bootstrap = clustered_bootstrap(bootstrap_origins)
    informative = [reports[name] for name in ["official_2018", "official_2019"] if reports[name]["routed"] > 0]
    accepted = bool(
        informative
        and sum(value["routed"] for value in informative) > 0
        and all(value["delta"] >= 0.0 for value in informative)
        and all(reports[name]["delta"] >= 0.0 for name in ["replay_2017", "replay_2018"] if reports[name]["routed"] > 0)
        and bootstrap["probability_positive"] >= 0.8
    )
    diagnostics = {
        "accepted": accepted,
        "origins": reports,
        "bootstrap": bootstrap,
        "acceptance": "increased_unique_coverage_nonnegative_all_informative_origins_probability_positive_at_least_0.8",
        "positive_pole": POSITIVE_POLE,
        "negative_pole": NEGATIVE_POLE,
    }
    report("historical_gate", diagnostics=json.dumps(diagnostics, sort_keys=True, allow_nan=True))
    return diagnostics, candidates


# Candidate

def persist_candidate(cache: Path, payload: dict[str, np.ndarray], diagnostics: dict[str, Any]) -> Path:
    path = cache / "predictions" / CANDIDATE_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload, diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, path)
    register_artifact(cache, {
        "name": "generic_exp_4 structured JATS endpoint evidence candidate",
        "path": f"predictions/{CANDIDATE_NAME}",
        "description": "Structured pre-origin JATS table and result-sentence endpoint facts routed after the banked run_0006 direct expert with byte-exact abstentions.",
        "content_key": f"rel-trial-study-outcome:{EXTRACTOR_VERSION}:run-0006-fallback",
        "rebuild_hint": "Run jats_direct_pipeline.py after the literature_v3 and registry snapshot caches are available.",
    })
    return path


def build(debug: bool) -> Path:
    cache = shared_cache_dir()
    fallback_validation, fallback_test, fallback_diagnostics = banked_fallback()
    feature_root = cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2"
    linkage = pd.read_parquet(feature_root / "linkage.parquet")
    direct = pd.read_parquet(cache / "snapshot_direct_evidence_v1" / "evidence.parquet")
    projected_root = cache / "registry_clock_lane0" / "projected"
    splits = ["official_2018"] if debug else list(ORIGINS)
    documents = {}
    facts = {}
    extraction = {}
    verdicts = {}
    for split in splits:
        documents[split], facts[split], extraction[split] = load_or_extract_origin(split, linkage, projected_root, cache, debug)
        verdicts[split] = aligned_verdicts(split, linkage, documents[split], direct)
    if debug:
        output = run_data_dir()
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "val_predictions.npy", fallback_validation)
        np.save(output / "test_predictions.npy", fallback_test)
        diagnostics = {"debug": True, "parsed_xml_limit": 16, "extraction": extraction, "fallback": fallback_diagnostics}
        (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
        subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
        return cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz"
    with np.load(cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz", allow_pickle=False) as snapshot:
        gate, historical = historical_gate(verdicts, snapshot)
        validation, validation_routed = apply_routes(fallback_validation, verdicts["validation_2020"])
        test, test_routed = apply_routes(fallback_test, verdicts["test_2021"])
    if not gate["accepted"]:
        validation = fallback_validation.copy()
        test = fallback_test.copy()
        validation_routed = np.zeros(len(validation), dtype=bool)
        test_routed = np.zeros(len(test), dtype=bool)
    if not np.array_equal(validation[~validation_routed], fallback_validation[~validation_routed]):
        raise RuntimeError("Validation abstention rows changed")
    if not np.array_equal(test[~test_routed], fallback_test[~test_routed]):
        raise RuntimeError("Test abstention rows changed")
    freeze_root = cache / "jats_endpoint_evidence_v1" / "model_a"
    freeze_root.mkdir(parents=True, exist_ok=True)
    validation_path = freeze_root / "validation_predictions.npy"
    np.save(validation_path, validation.astype(np.float64))
    validation_checksum = file_sha256(validation_path)
    report("model_a_frozen", checksum=validation_checksum, labels_loaded=False, routed=int(validation_routed.sum()))
    context = load_task()
    _ = context.val.df[context.target_col].to_numpy(dtype=np.int8)
    if file_sha256(validation_path) != validation_checksum:
        raise RuntimeError("Model A validation checksum changed after validation labels were loaded")
    diagnostics = {
        "accepted": gate["accepted"],
        "variant": "jats_structured_direct_v1" if gate["accepted"] else "run_0006_byte_exact_fallback",
        "fallback": fallback_diagnostics,
        "extraction": extraction,
        "historical_gate": gate,
        "validation": {
            "rows": len(validation),
            "new_routed": int(validation_routed.sum()),
            "positive_routes": int(verdicts["validation_2020"]["route_positive"].sum()) if gate["accepted"] else 0,
            "negative_routes": int(verdicts["validation_2020"]["route_negative"].sum()) if gate["accepted"] else 0,
            "abstentions_byte_exact": True,
        },
        "test": {
            "rows": len(test),
            "new_routed": int(test_routed.sum()),
            "positive_routes": int(verdicts["test_2021"]["route_positive"].sum()) if gate["accepted"] else 0,
            "negative_routes": int(verdicts["test_2021"]["route_negative"].sum()) if gate["accepted"] else 0,
            "abstentions_byte_exact": True,
        },
        "validation_checksum": validation_checksum,
        "validation_prediction_label_fit": "banked_run_0006_model_a_plus_fixed_forward_gated_jats_expert_without_validation_labels",
        "test_prediction_label_fit": "banked_run_0006_model_b_plus_same_frozen_jats_design",
        "elapsed_seconds": time.time() - START,
    }
    payload = {
        "val": validation.astype(np.float64),
        "test": test.astype(np.float64),
        "validation_routed": validation_routed,
        "test_routed": test_routed,
        "validation_nct_id": verdicts["validation_2020"]["nct_id"].to_numpy(dtype=np.int64),
        "test_nct_id": verdicts["test_2021"]["nct_id"].to_numpy(dtype=np.int64),
        "historical_2018": historical["official_2018"],
        "historical_2019": historical["official_2019"],
    }
    path = persist_candidate(cache, payload, diagnostics)
    locked_append(cache / "features_history.md", f'''\n### Structured JATS endpoint-linked statistical evidence\n- run/experiment: generic_exp_4 lane 0 | status: {"TESTED-KEPT" if gate["accepted"] else "TESTED-REJECTED"}\n- what: Safe pre-origin PMC XML, span-aware hierarchical tables, result-sentence p-values, separately normalized endpoint/time-frame identity, MedCPT document similarity as an auxiliary only, and fixed 0.995/0.005 routing after run_0006.\n- outcome: extraction {json.dumps(extraction, sort_keys=True, allow_nan=True)}; gate {json.dumps(gate, sort_keys=True, allow_nan=True)}; validation new routes {int(validation_routed.sum())}; test new routes {int(test_routed.sum())}.\n- takeaway: Deterministic literature facts are admitted only when they add unique coverage, remain nonnegative on every informative historical origin, and clear the clustered paired-bootstrap gate; all other rows are byte-identical to run_0006.\n''')
    locked_append(cache / "table_information.md", f'''\n### 2026-08-14 structured pre-origin JATS endpoint facts\n- Full-text XML remains admissible only with a PMCID match, complete pre-origin publication date, no correction/retraction marker, and no unverified or post-origin version marker.\n- Tables preserve captions, hierarchical headers, row labels, row/column spans, missing cells, and footnotes; facts retain document IDs, XML paths, extraction rules, surrounding text, and content hashes.\n- Recon and extraction by origin: {json.dumps(extraction, sort_keys=True, allow_nan=True)}.\n''')
    report("candidate", path=path, accepted=gate["accepted"], validation_checksum=validation_checksum, validation_new_routes=int(validation_routed.sum()), test_new_routes=int(test_routed.sum()))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    build(arguments.debug)


if __name__ == "__main__":
    main()
