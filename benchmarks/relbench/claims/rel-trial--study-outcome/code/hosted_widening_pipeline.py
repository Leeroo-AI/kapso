# Imports

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
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
from sklearn.metrics import roc_auc_score

from campaign_io import locked_append, register_artifact
from hosted_judgments_v2 import MODEL as HOSTED_MODEL
from jats_direct_pipeline import (
    FALLBACK_TEST_SHA,
    FALLBACK_VALIDATION_SHA,
    ORIGINS,
    aligned_verdicts,
    apply_routes,
    banked_fallback,
    clustered_bootstrap,
    file_sha256,
    labels_by_split,
)
from kapso_datasets.common import load_task, run_data_dir, shared_cache_dir
from publication_evidence import (
    PROMPT_VERSION,
    _gzip_read,
    _section_payloads,
    adjudicate_candidates,
    build_trial_contexts,
    live_payload_probes,
    prefilter_candidates,
    retrieve_origin,
)


# Configuration

START = time.time()
HOSTED_VERSION = "uncapped-hosted-primary-adjudication-v1"
CANDIDATE_NAME = "generic_exp_4_uncapped_hosted_v1.npz"
POLICIES = ["agreement", "explicit_exact", "hosted_exact"]
INTERVALS = [0.985, 0.990, 0.995]


# Runtime

def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[hosted-widening] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


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


# Payloads

def cached_full_text_payloads(records: pd.DataFrame, cache: Path, workers: int = 36, maximum: int | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if maximum is not None:
        records = records.head(maximum)
    xml_root = cache / "literature_v3" / "raw" / "full_text_xml"
    completed = []
    def work(position: int, row: pd.Series) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
        item = row.to_dict()
        pmcid = str(item.get("pmcid", "")).upper()
        path = xml_root / f"{pmcid}.xml.gz"
        windows = []
        reason = "abstract_fallback_no_cached_safe_xml"
        if pmcid and path.exists():
            try:
                windows, reason = _section_payloads(_gzip_read(path), row)
            except Exception as error:
                reason = f"abstract_fallback_xml_failure:{type(error).__name__}"
        payloads = []
        if windows:
            for index, window in enumerate(windows):
                current = dict(item)
                current["full_text"] = window
                current["full_text_safe"] = True
                current["full_text_reason"] = reason
                current["payload_type"] = "full-text"
                current["document_window_id"] = f"{pmcid}:window-{index + 1}"
                current["content_hash"] = hashlib.sha256(window.encode("utf-8", errors="replace")).hexdigest()
                payloads.append(current)
        else:
            item["full_text"] = ""
            item["full_text_safe"] = False
            item["full_text_reason"] = reason
            item["payload_type"] = "abstract"
            item["document_window_id"] = f"{item.get('publication_identity', '')}:abstract"
            payloads.append(item)
        return position, payloads, {"safe": int(bool(windows)), "windows": len(payloads), "cached_xml": int(path.exists())}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(work, position, row) for position, (_, row) in enumerate(records.iterrows())]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda value: value[0])
    frame = pd.DataFrame([item for _, rows, _ in completed for item in rows]).reset_index(drop=True)
    diagnostics = {
        "admissible_records": len(records),
        "payload_windows": len(frame),
        "safe_full_text_records": sum(item["safe"] for _, _, item in completed),
        "cached_xml_records": sum(item["cached_xml"] for _, _, item in completed),
        "abstract_fallback_records": sum(not item["safe"] for _, _, item in completed),
        "workers": workers,
    }
    return frame, diagnostics


def load_or_adjudicate(split: str, linkage: pd.DataFrame, projected_root: Path, cache: Path, debug: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = cache / "uncapped_hosted_primary_v1" / "origins" / split
    path = root / "adjudications.parquet"
    diagnostics_path = root / "diagnostics.json"
    if not debug and path.exists() and diagnostics_path.exists():
        frame = pd.read_parquet(path)
        diagnostics = json.loads(diagnostics_path.read_text())
        diagnostics["cache_hit"] = True
        report("origin_cache", split=split, rows=len(frame))
        return frame, diagnostics
    records, retrieval = retrieve_origin(linkage, ORIGINS[split], cache)
    contexts = build_trial_contexts(linkage, ORIGINS[split], projected_root)
    candidates = prefilter_candidates(records, contexts, maximum=10**9)
    candidates, payloads = cached_full_text_payloads(candidates, cache, workers=4 if debug else 36, maximum=16 if debug else None)
    adjudications, hosted = adjudicate_candidates(candidates, contexts, cache, concurrency=1 if debug else 32)
    diagnostics = {
        "split": split,
        "origin": ORIGINS[split].strftime("%Y-%m-%d"),
        "retrieval": retrieval,
        "uncapped_candidates": int(len(candidates)),
        "payloads": payloads,
        "hosted": hosted,
        "cache_hit": False,
    }
    if not debug:
        atomic_parquet(adjudications, path)
        atomic_json(diagnostics, diagnostics_path)
    report("origin_adjudicated", split=split, diagnostics=json.dumps(diagnostics, sort_keys=True))
    return adjudications, diagnostics


# Verdicts

def hosted_trial_verdicts(adjudications: pd.DataFrame, deterministic: pd.DataFrame) -> pd.DataFrame:
    if adjudications.empty:
        return pd.DataFrame(columns=["external_nct_id", *[f"{policy}_positive" for policy in POLICIES], *[f"{policy}_negative" for policy in POLICIES]])
    frame = adjudications.copy()
    identity_exact = frame["judgment_is_this_trial"].astype(bool) & frame["judgment_trial_identity"].eq("exact")
    identity_probable = frame["judgment_is_this_trial"].astype(bool) & frame["judgment_trial_identity"].isin(["exact", "probable"])
    final_primary = frame["judgment_final_status"].eq("final") & frame["judgment_report_type"].eq("primary-results")
    sufficient = ~frame["judgment_insufficient_evidence"].astype(bool)
    explicit = frame["judgment_explicit_p_value"].notna() & frame["judgment_explicit_p_value"].astype(str).str.strip().ne("")
    verdict = frame["judgment_primary_endpoint_met"].isin(["yes", "no"])
    publication_types = frame["publication_types"].map(lambda value: " ".join(value) if isinstance(value, (list, tuple, np.ndarray)) else str(value)).str.casefold()
    clinical_report = publication_types.str.contains(r"clinical trial|randomized controlled trial|multicenter study", regex=True) & ~publication_types.str.contains(r"review|meta-analysis|systematic review", regex=True)
    frame["eligible_explicit"] = identity_exact & final_primary & sufficient & explicit & verdict & clinical_report & (frame["judgment_endpoint_match"] >= 4) & (frame["judgment_confidence"] >= 4)
    frame["eligible_hosted"] = identity_exact & final_primary & sufficient & verdict & clinical_report & (frame["judgment_endpoint_match"] >= 5) & (frame["judgment_confidence"] >= 5)
    deterministic_map = {}
    if len(deterministic):
        for nct_id, current in deterministic.groupby("queried_nct_id"):
            positive = bool(current["positive"].any())
            negative = bool(current["complete_negative"].any())
            deterministic_map[str(nct_id)] = "yes" if positive and not negative else "no" if negative and not positive else ""
    rows = []
    for nct_id, current in frame.groupby("queried_nct_id", sort=False):
        explicit_values = set(current.loc[current["eligible_explicit"], "judgment_primary_endpoint_met"].astype(str))
        hosted_values = set(current.loc[current["eligible_hosted"], "judgment_primary_endpoint_met"].astype(str))
        deterministic_value = deterministic_map.get(str(nct_id), "")
        agreement_values = set()
        if deterministic_value and deterministic_value in set(current.loc[current["eligible_hosted"], "judgment_primary_endpoint_met"].astype(str)):
            agreement_values.add(deterministic_value)
        item: dict[str, Any] = {"external_nct_id": str(nct_id)}
        for policy, values in [("agreement", agreement_values), ("explicit_exact", explicit_values), ("hosted_exact", hosted_values)]:
            item[f"{policy}_positive"] = values == {"yes"}
            item[f"{policy}_negative"] = values == {"no"}
            item[f"{policy}_conflict"] = len(values) > 1
        item["adjudicated_windows"] = len(current)
        rows.append(item)
    return pd.DataFrame(rows)


def align_hosted(split: str, linkage: pd.DataFrame, direct: pd.DataFrame, adjudications: pd.DataFrame, deterministic: pd.DataFrame) -> pd.DataFrame:
    rows = linkage[linkage["split"].eq(split)].copy().reset_index(drop=True)
    direct_rows = direct[direct["split"].eq(split)].copy().reset_index(drop=True)
    trial = hosted_trial_verdicts(adjudications, deterministic)
    rows = rows.merge(trial, on="external_nct_id", how="left")
    rows["existing_direct"] = ~direct_rows["direct_abstain"].to_numpy(dtype=bool)
    for policy in POLICIES:
        positive = rows.get(f"{policy}_positive", pd.Series(False, index=rows.index)).eq(True)
        negative = rows.get(f"{policy}_negative", pd.Series(False, index=rows.index)).eq(True)
        conflict = rows.get(f"{policy}_conflict", pd.Series(False, index=rows.index)).eq(True)
        rows[f"{policy}_positive"] = positive & ~rows["existing_direct"] & ~conflict
        rows[f"{policy}_negative"] = negative & ~rows["existing_direct"] & ~conflict
    return rows


def policy_route(baseline: np.ndarray, aligned: pd.DataFrame, policies: list[str], pole: float) -> tuple[np.ndarray, np.ndarray]:
    result = np.asarray(baseline, dtype=np.float64).copy()
    positive = np.zeros(len(result), dtype=bool)
    negative = np.zeros(len(result), dtype=bool)
    for policy in policies:
        positive |= aligned[f"{policy}_positive"].to_numpy(dtype=bool)
        negative |= aligned[f"{policy}_negative"].to_numpy(dtype=bool)
    conflict = positive & negative
    positive &= ~conflict
    negative &= ~conflict
    result[positive] = pole
    result[negative] = 1.0 - pole
    return result, positive | negative


# Gate

def auc(labels: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(labels, prediction))


def select_policy(aligned: dict[str, pd.DataFrame], snapshot: np.lib.npyio.NpzFile) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = labels_by_split()
    baselines = {
        "official_2018": snapshot["historical_2018"].astype(np.float64),
        "official_2019": snapshot["historical_2019"].astype(np.float64),
    }
    designs = []
    policy_sets = [["agreement"], ["agreement", "explicit_exact"], ["agreement", "explicit_exact", "hosted_exact"]]
    for policies in policy_sets:
        for pole in INTERVALS:
            nct_id, current_labels = labels["official_2018"]
            prediction, routed = policy_route(baselines["official_2018"], aligned["official_2018"], policies, pole)
            designs.append({
                "policies": policies,
                "pole": pole,
                "design_delta": auc(current_labels, prediction) - auc(current_labels, baselines["official_2018"]),
                "design_routed": int(routed.sum()),
            })
    selected = max(designs, key=lambda value: (value["design_delta"], -len(value["policies"]), -abs(value["pole"] - 0.985)))
    reports = {}
    candidates = {}
    bootstrap_rows = []
    for split in ["replay_2017", "official_2018", "replay_2018", "official_2019"]:
        nct_id, current_labels = labels[split]
        baseline = baselines.get(split, np.full(len(current_labels), 0.5, dtype=np.float64))
        prediction, routed = policy_route(baseline, aligned[split], selected["policies"], selected["pole"])
        route_expected = np.zeros(len(current_labels), dtype=bool)
        for policy in selected["policies"]:
            route_expected |= aligned[split][f"{policy}_positive"].to_numpy(dtype=bool)
        reports[split] = {
            "rows": len(current_labels),
            "routed": int(routed.sum()),
            "positive_routes": int((routed & route_expected).sum()),
            "route_accuracy": float(np.mean(current_labels[routed] == route_expected[routed])) if routed.any() else float("nan"),
            "baseline_auc": auc(current_labels, baseline),
            "candidate_auc": auc(current_labels, prediction),
            "delta": auc(current_labels, prediction) - auc(current_labels, baseline),
            "no_evidence_unchanged": bool(np.array_equal(prediction[~routed], baseline[~routed])),
            "baseline_kind": "run_0006_historical" if split in baselines else "uninformative_replay_control",
        }
        candidates[split] = prediction
        if split in baselines and routed.any():
            bootstrap_rows.append({"labels": current_labels, "nct_id": nct_id, "baseline": baseline, "candidate": prediction})
    bootstrap = clustered_bootstrap(bootstrap_rows) if bootstrap_rows else {
        "draws": 0, "mean_delta": 0.0, "standard_error": 0.0, "probability_positive": 0.0, "lower_10": 0.0, "upper_90": 0.0, "cluster": "origin_and_nct_id",
    }
    informative = [reports[name] for name in ["official_2018", "official_2019"] if reports[name]["routed"] > 0]
    accepted = bool(
        informative
        and sum(value["routed"] for value in informative) > 0
        and all(value["delta"] >= 0 for value in informative)
        and all(reports[name]["delta"] >= 0 for name in ["replay_2017", "replay_2018"] if reports[name]["routed"] > 0)
        and bootstrap["probability_positive"] >= 0.8
    )
    diagnostics = {
        "accepted": accepted,
        "selected": selected,
        "design_grid": designs,
        "origins": reports,
        "bootstrap": bootstrap,
        "selection_fit": "official_2018_only_then_frozen_for_replay_and_sealed_official_2019_confirmation",
        "interval_grid": INTERVALS,
    }
    report("historical_gate", diagnostics=json.dumps(diagnostics, sort_keys=True, allow_nan=True))
    return diagnostics, candidates


# Candidate

def persist(cache: Path, validation: np.ndarray, test: np.ndarray, validation_routed: np.ndarray, test_routed: np.ndarray, diagnostics: dict[str, Any]) -> Path:
    path = cache / "predictions" / CANDIDATE_NAME
    temporary = path.with_suffix(".npz.part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, val=validation, test=test, validation_routed=validation_routed, test_routed=test_routed, diagnostics_json=np.asarray([json.dumps(diagnostics, sort_keys=True, allow_nan=True)]))
    os.replace(temporary, path)
    register_artifact(cache, {
        "name": "generic_exp_4 uncapped hosted publication adjudication",
        "path": f"predictions/{CANDIDATE_NAME}",
        "description": "Uncapped pre-origin publication adjudication with cached safe full-text windows, abstract fallbacks, forward-selected confidence classes, and run_0006-preserving evidence-last routing.",
        "content_key": f"rel-trial-study-outcome:{HOSTED_VERSION}:{HOSTED_MODEL}:{PROMPT_VERSION}",
        "rebuild_hint": "Run hosted_widening_pipeline.py after origin retrieval and JATS extraction; hosted responses resume from content-addressed caches.",
    })
    return path


def build(debug: bool) -> Path:
    cache = shared_cache_dir()
    fallback_validation, fallback_test, fallback = banked_fallback()
    probe = live_payload_probes(cache, maximum_calls=1)
    report("live_probe", rows=probe["calls"], diagnostics=json.dumps(probe, sort_keys=True))
    feature_root = cache / "registry_clock_lane0" / "features" / "registry_clock_features_v2"
    linkage = pd.read_parquet(feature_root / "linkage.parquet")
    direct = pd.read_parquet(cache / "snapshot_direct_evidence_v1" / "evidence.parquet")
    projected_root = cache / "registry_clock_lane0" / "projected"
    splits = ["official_2018"] if debug else list(ORIGINS)
    adjudications = {}
    extraction = {}
    aligned = {}
    for split in splits:
        adjudications[split], extraction[split] = load_or_adjudicate(split, linkage, projected_root, cache, debug)
        deterministic_path = cache / "jats_endpoint_evidence_v1" / "origins" / split / "documents.parquet"
        deterministic = pd.read_parquet(deterministic_path) if deterministic_path.exists() else pd.DataFrame()
        aligned[split] = align_hosted(split, linkage, direct, adjudications[split], deterministic)
    if debug:
        output = run_data_dir()
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "val_predictions.npy", fallback_validation)
        np.save(output / "test_predictions.npy", fallback_test)
        (output / "metrics.json").write_text(json.dumps({"debug": True, "probe": probe, "extraction": extraction}, indent=2, sort_keys=True, allow_nan=True) + "\n")
        subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
        return cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz"
    with np.load(cache / "predictions" / "generic_exp_2_snapshot_direct_v1.npz", allow_pickle=False) as snapshot:
        gate, historical = select_policy(aligned, snapshot)
    validation, validation_routed = policy_route(fallback_validation, aligned["validation_2020"], gate["selected"]["policies"], gate["selected"]["pole"])
    test, test_routed = policy_route(fallback_test, aligned["test_2021"], gate["selected"]["policies"], gate["selected"]["pole"])
    if not gate["accepted"]:
        validation = fallback_validation.copy()
        test = fallback_test.copy()
        validation_routed = np.zeros(len(validation), dtype=bool)
        test_routed = np.zeros(len(test), dtype=bool)
    if not np.array_equal(validation[~validation_routed], fallback_validation[~validation_routed]):
        raise RuntimeError("Hosted validation abstentions changed")
    if not np.array_equal(test[~test_routed], fallback_test[~test_routed]):
        raise RuntimeError("Hosted test abstentions changed")
    freeze_root = cache / "uncapped_hosted_primary_v1" / "model_a"
    freeze_root.mkdir(parents=True, exist_ok=True)
    validation_path = freeze_root / "validation_predictions.npy"
    np.save(validation_path, validation.astype(np.float64))
    checksum = file_sha256(validation_path)
    report("model_a_frozen", checksum=checksum, labels_loaded=False, routed=int(validation_routed.sum()))
    context = load_task()
    _ = context.val.df[context.target_col].to_numpy(dtype=np.int8)
    if file_sha256(validation_path) != checksum:
        raise RuntimeError("Hosted Model A checksum changed after validation label access")
    diagnostics = {
        "accepted": gate["accepted"],
        "variant": "uncapped_hosted_primary_v1" if gate["accepted"] else "run_0006_byte_exact_fallback",
        "fallback": fallback,
        "probe": probe,
        "adjudication": extraction,
        "historical_gate": gate,
        "validation_new_routes": int(validation_routed.sum()),
        "test_new_routes": int(test_routed.sum()),
        "validation_checksum": checksum,
        "validation_prediction_label_fit": "banked_run_0006_model_a_plus_nested_forward_origin_hosted_design_without_validation_labels",
        "test_prediction_label_fit": "banked_run_0006_model_b_plus_same_frozen_hosted_design",
        "elapsed_seconds": time.time() - START,
    }
    path = persist(cache, validation.astype(np.float64), test.astype(np.float64), validation_routed, test_routed, diagnostics)
    locked_append(cache / "features_history.md", f'''\n### Uncapped hosted endpoint adjudication\n- run/experiment: generic_exp_4 lane 0 | status: {"TESTED-KEPT" if gate["accepted"] else "TESTED-REJECTED"}\n- what: Every admissible publication, safe cached full-text result/table windows or abstract fallback, structured primary-endpoint adjudication at concurrency 32, deterministic-source agreement, and nested forward-origin confidence/interval selection.\n- outcome: adjudication {json.dumps(extraction, sort_keys=True, allow_nan=True)}; gate {json.dumps(gate, sort_keys=True, allow_nan=True)}; validation new routes {int(validation_routed.sum())}; test new routes {int(test_routed.sum())}.\n- takeaway: Hosted widening is admitted only when unique routes remain nonnegative on all four historical origins and clear the clustered-bootstrap probability gate; otherwise the output is byte-identical run_0006.\n''')
    report("candidate", path=path, accepted=gate["accepted"], validation_new_routes=int(validation_routed.sum()), test_new_routes=int(test_routed.sum()))
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    build(arguments.debug)


if __name__ == "__main__":
    main()
