from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from campaign_io import locked_append, register_artifact
from kapso_datasets.common import load_task, run_data_dir, shared_cache_dir
from registry_clock import FEATURE_VERSION, build_registry_features, load_registry_bundle, refresh_literature_features, save_registry_bundle
from registry_model import fit_registry_model_b, save_registry_candidate, select_registry_model_a
from replay_features import generate_july_replay


START = time.time()
warnings.filterwarnings("ignore")


def report(name: str, **values: object) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[registry] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_seeds(context: object) -> tuple[pd.DataFrame, pd.Series]:
    train = context.train.df.copy().reset_index(drop=True)
    validation = context.val.df.copy().reset_index(drop=True)
    test = context.test.df.copy().reset_index(drop=True)
    replay = generate_july_replay(context.db, start_year=2017, end_year=2018)
    validation_ids = set(validation["nct_id"].astype(int))
    test_ids = set(test["nct_id"].astype(int))
    replay = replay[~replay["nct_id"].astype(int).isin(validation_ids | test_ids)].reset_index(drop=True)
    pieces = [
        replay[replay["timestamp"] == pd.Timestamp("2017-07-01")][["timestamp", "nct_id", "outcome"]].assign(split="replay_2017"),
        train[train["timestamp"] == pd.Timestamp("2018-01-01")][["timestamp", "nct_id", context.target_col]].rename(columns={context.target_col: "outcome"}).assign(split="official_2018"),
        replay[replay["timestamp"] == pd.Timestamp("2018-07-01")][["timestamp", "nct_id", "outcome"]].assign(split="replay_2018"),
        train[train["timestamp"] == pd.Timestamp("2019-01-01")][["timestamp", "nct_id", context.target_col]].rename(columns={context.target_col: "outcome"}).assign(split="official_2019"),
        validation[["timestamp", "nct_id", context.target_col]].rename(columns={context.target_col: "outcome"}).assign(split="validation_2020"),
        test[["timestamp", "nct_id"]].assign(outcome=np.nan, split="test_2021"),
    ]
    seeds = pd.concat(pieces, ignore_index=True)
    seeds["row_id"] = np.arange(len(seeds), dtype=np.int64)
    label_map = seeds.set_index("row_id")["outcome"]
    return seeds.drop(columns=["outcome"]), label_map


def run(debug: bool) -> None:
    cache = shared_cache_dir()
    context = load_task()
    seeds, label_map = build_seeds(context)
    report("seeds", rows=len(seeds), strata=json.dumps(seeds["split"].value_counts().to_dict(), sort_keys=True))
    feature_cache = cache / "registry_clock_lane0" / "features" / FEATURE_VERSION
    if (feature_cache / "reports.json").exists():
        bundle = load_registry_bundle(feature_cache)
        report("features_cache", state="hit", columns=bundle.features_by_strength[20.0].shape[1])
    else:
        bundle = build_registry_features(context.db, seeds, cache / "registry_clock_lane0" / "projected")
        save_registry_bundle(bundle, feature_cache)
        report("features_cache", state="built", columns=bundle.features_by_strength[20.0].shape[1])
    if refresh_literature_features(bundle, cache / "registry_clock_lane0" / "projected"):
        save_registry_bundle(bundle, feature_cache)
        report("literature_features", state="refreshed", columns=bundle.features_by_strength[20.0].shape[1])
    split_reports = {}
    for split, current in bundle.linkage.groupby("split"):
        split_reports[split] = {
            "rows": int(len(current)),
            "linked": int(current["linked"].sum()),
            "coverage": float(current["linked"].mean()),
            "exact": int((current["match_type"] == "exact").sum()),
            "fuzzy": int((current["match_type"] == "fuzzy").sum()),
        }
    validation_coverage = split_reports["validation_2020"]["coverage"]
    test_coverage = split_reports["test_2021"]["coverage"]
    report("linkage", split_reports=json.dumps(split_reports, sort_keys=True))
    if validation_coverage < 0.90 or test_coverage < 0.90:
        raise RuntimeError(f"Registry linkage coverage gate failed: validation={validation_coverage:.4f} test={test_coverage:.4f}")
    incumbent_exact = cache / "predictions" / "incumbent_run_0007"
    incumbent_channels = cache / "predictions" / "incumbent_run_0007_channels_v1.npz"
    if not incumbent_channels.exists():
        incumbent_channels = cache / "predictions" / "generic_exp_1_compact_v1.npz"
        report("incumbent_forward_cache", state="compact_fallback", path=incumbent_channels)
    incumbent_validation = np.load(incumbent_exact / "val_predictions.npy", allow_pickle=False)
    incumbent_test = np.load(incumbent_exact / "test_predictions.npy", allow_pickle=False)
    model_a = select_registry_model_a(
        bundle, label_map, incumbent_channels, incumbent_validation,
        context.train.df["nct_id"].reset_index(drop=True),
    )
    report("model_a_gate", diagnostics=json.dumps(model_a.diagnostics, sort_keys=True))
    output = run_data_dir()
    output.mkdir(parents=True, exist_ok=True)
    validation_path = output / "val_predictions.npy"
    np.save(validation_path, np.asarray(model_a.validation_prediction, dtype=np.float64))
    validation_checksum = file_sha256(validation_path)
    report("model_a_saved", checksum=validation_checksum, labels_exposed=False)
    validation_rows = np.flatnonzero((bundle.seeds["split"] == "validation_2020").to_numpy())
    validation_labels = context.val.df[context.target_col].to_numpy(dtype=np.int32)
    label_map = label_map.copy()
    label_map.loc[bundle.seeds.iloc[validation_rows]["row_id"].to_numpy()] = validation_labels
    external_test, test_prediction = fit_registry_model_b(bundle, label_map, model_a, incumbent_test)
    if file_sha256(validation_path) != validation_checksum:
        raise RuntimeError("Model A validation vector changed after validation labels were exposed")
    np.save(output / "test_predictions.npy", np.asarray(test_prediction, dtype=np.float64))
    subprocess_result = __import__("subprocess").run([sys.executable, "kapso_datasets/check_predictions.py"], check=True, capture_output=True, text=True)
    print(subprocess_result.stdout, end="")
    candidate_cache = cache / "predictions" / "generic_exp_2_registry_clock_v1.npz"
    save_registry_candidate(candidate_cache, model_a, external_test, test_prediction, {"by_snapshot": bundle.reports, "by_split": split_reports})
    diagnostics = {
        "linkage_by_snapshot": bundle.reports,
        "linkage_by_split": split_reports,
        "model_a": model_a.diagnostics,
        "validation_checksum": validation_checksum,
        "validation_prediction_label_fit": "train_and_replay_only",
        "test_prediction_label_fit": "train_replay_plus_validation",
        "elapsed_seconds": time.time() - START,
    }
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    diagnostics_cache = cache / "registry_clock_lane0" / "diagnostics.json"
    diagnostics_cache.write_text(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=True) + "\n")
    register_artifact(cache, {
        "name": "generic_exp_2 point-in-time registry clock candidate",
        "path": "predictions/generic_exp_2_registry_clock_v1.npz",
        "description": "Conservatively linked pre-origin AACT registry-clock and neighborhood expert with recent forward-gate diagnostics and Model A/B vectors.",
        "content_key": "rel-trial-study-outcome:generic-exp-2:registry-clock-v1",
        "rebuild_hint": "Run registry_pipeline.py after projecting the six declared safe snapshots.",
    })
    locked_append(cache / "features_history.md", f'''\n### Point-in-time registry evidence clock and neighborhood expert — measurement\n- run/experiment: generic_exp_2 lane 0 | status: {"TESTED-KEPT" if model_a.diagnostics["accepted"] else "TESTED-REJECTED"}\n- what: Safe-snapshot status/completion/enrollment/results-QC clocks plus empirical-Bayes sponsor, condition, intervention, facility, country, one-hop, and two-hop public-result neighborhoods.\n- outcome: linkage {json.dumps(split_reports, sort_keys=True)}; gate {json.dumps(model_a.diagnostics, sort_keys=True)}.\n- takeaway: external weight is accepted only by the paired 2,000-resample recent-origin gate; unmatched trials route to the exact run_0007 incumbent.\n''')
    locked_append(cache / "table_information.md", f'''\n### 2026-08-13 point-in-time AACT snapshots\n- Safe snapshots 2017-06-13, 2017-12-17, 2018-06-01, 2018-12-01, 2019-12-01, and 2020-12-01 were projected from official PostgreSQL dumps with parser {FEATURE_VERSION}.\n- Sanitized study identities are mapped to external NCT accessions using exact normalized titles plus start-date compatibility and at least two unused audit-field agreements; ambiguous assignments are one-to-one.\n- Linkage coverage by modeling stratum: {json.dumps(split_reports, sort_keys=True)}.\n- Public neighborhood labels reproduce valid primary p-value filters and additionally require results_first_posted_date before the snapshot; the target trial itself is excluded.\n''')
    report("complete", validation_checksum=validation_checksum, blend_weight=model_a.blend_weight)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    arguments = parser.parse_args()
    run(arguments.debug)


if __name__ == "__main__":
    main()
