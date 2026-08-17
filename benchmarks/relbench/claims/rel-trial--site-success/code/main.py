# Imports

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import duckdb
import lightgbm
import numpy as np
import pandas as pd
import sklearn

from campaign_io import locked_append, register_artifact
from hazard_pipeline import (
    VERSION as HAZARD_VERSION,
    calibrator_record,
    ensure_rolling_origin,
    final_reporting_prediction,
    run_reporting_gates,
)
from hosted_judgments import ensure_judgments, forward_measurement
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from registry_features import CATEGORICAL as REGISTRY_CATEGORICAL, cached_registry_matrix
from trial_pipeline import (
    VERSION,
    build_bundle,
    build_facility_features,
    connected_bootstrap,
    slice_diagnostics,
    target_exact_direct_gate,
)


# Runtime

warnings.filterwarnings("ignore")
START = time.time()


def report(name: str, **values: Any) -> None:
    payload = " ".join(f"{key}={value}" for key, value in values.items())
    print(f"[phase] {name} elapsed={time.time() - START:.2f}s {payload}".rstrip(), flush=True)


def prediction_checksum(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def candidate_cache_path(cache: Path, debug: bool) -> Path:
    sources = [Path("main.py"), Path("hazard_pipeline.py"), Path("trial_pipeline.py"), Path("registry_features.py")]
    digest = hashlib.sha256()
    for path in sources:
        digest.update(path.read_bytes())
    mode = "debug" if debug else "full"
    return cache / "predictions" / f"lane1_{HAZARD_VERSION}_{mode}_{digest.hexdigest()[:16]}"


def load_cached_candidate(path: Path, output: Path) -> bool:
    if not (path / "READY").exists():
        return False
    validation = np.load(path / "val_predictions.npy", allow_pickle=False)
    test = np.load(path / "test_predictions.npy", allow_pickle=False)
    np.save(output / "val_predictions.npy", validation)
    np.save(output / "test_predictions.npy", test)
    metrics = json.loads((path / "metrics.json").read_text())
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n")
    report(
        "candidate_cache",
        state="hit",
        val_shape=validation.shape,
        test_shape=test.shape,
        val_mean=f"{validation.mean():.6f}",
        test_mean=f"{test.mean():.6f}",
        validation_checksum=prediction_checksum(validation),
        hosted_rows=metrics.get("hosted", {}).get("forward", {}).get("rows", 0),
    )
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    return True


# Hosted measurement

def hosted_indices(bundle: Any, count: int) -> list[int]:
    reporting = bundle.replay[
        bundle.replay["report_label"].eq(1)
        & bundle.replay["timestamp"].dt.year.between(2012, 2019)
    ].drop_duplicates("nct_id", keep="last").copy()
    reporting["hash"] = pd.util.hash_pandas_object(reporting[["nct_id", "timestamp"]], index=False).to_numpy()
    pieces = []
    per_label = count // 2
    for label in [0, 1]:
        current = reporting[reporting["success_all"].eq(label)].sort_values("hash")
        pieces.append(current.head(per_label))
    selected = pd.concat(pieces).sort_values(["timestamp", "hash"])
    if len(selected) < count:
        remaining = reporting[~reporting["pair_row"].isin(selected["pair_row"])].sort_values("hash")
        selected = pd.concat([selected, remaining.head(count - len(selected))])
    return selected["pair_row"].astype(int).tolist()


def run_hosted_measurement(bundle: Any, cache: Path, debug: bool) -> dict[str, Any]:
    count = 64 if debug else 512
    indices = hosted_indices(bundle, count)
    origins = [str(bundle.pairs.at[index, "timestamp"]) for index in range(len(bundle.pairs))]
    judgments, extraction = ensure_judgments(
        origins,
        bundle.contexts,
        bundle.documents,
        indices,
        cache,
        concurrency=12 if debug else 32,
    )
    replay = bundle.replay.drop_duplicates("pair_row", keep="last").set_index("pair_row")
    labels = np.asarray([replay.at[index, "success_all"] for index in indices], dtype=np.int8)
    years = np.asarray([pd.Timestamp(bundle.pairs.at[index, "timestamp"]).year for index in indices], dtype=np.int32)
    numeric = bundle.features.select_dtypes(include=[np.number]).iloc[indices]
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median()).fillna(0.0)
    variances = numeric.var().sort_values(ascending=False)
    structured = numeric[variances.head(48).index].to_numpy(dtype=np.float64)
    measurement = forward_measurement(judgments, structured, labels, years)
    report(
        "hosted_text",
        requested=int(extraction["requested"]),
        new=int(extraction["new"]),
        rate=f"{extraction['new_rows_per_second']:.3f}",
        forward_rows=measurement["rows"],
        forward_auc=measurement["auc"],
        forward_mae=measurement["mae"],
    )
    return {"extraction": extraction, "forward": measurement, "indices": len(indices)}


def modernbert_measurement(cache: Path) -> dict[str, Any]:
    gates = sorted((cache / "modernbert_lane0").glob("*/forward_2019_gate.json"))
    if not gates:
        return {"state": "unavailable"}
    gate = json.loads(gates[-1].read_text())
    no_encoder = max(
        item["auc"] for item in gate["blend_grid"] if item["modernbert_weight"] == 0.0
    )
    delta = float(gate["best_nonnegative_blend"]["auc"] - no_encoder)
    return {
        "state": "rejected_by_forward_gate",
        "rows": gate["rows"],
        "auc": gate["modernbert_auc"],
        "best_blend_weight": gate["best_nonnegative_blend"]["modernbert_weight"],
        "blend_auc_delta": delta,
        "artifact": str(gates[-1].parent.relative_to(cache)),
    }


# Campaign memory

def record_campaign_memory(cache: Path, bundle: Any, metrics: dict[str, Any]) -> None:
    locked_append(cache / "table_information.md", f"""
### 2026-08-16 monthly reporting hazard extension
- Quarterly replay expands to {metrics.get('reporting_system', {}).get('person_periods', {}).get('rows', 0):,} at-risk person-period rows. A positive trial exits at its exact report month; censored trials contribute every month through day 365.
- The conditional roster model consumes six standardized edge features and maximizes reporter-subset likelihood conditional on at least one reporter. Its frozen parameters are {json.dumps(metrics.get('reporting_system', {}).get('conditional_parameters', {}), sort_keys=True)}.
- The added 2017 facility roster uses the physical 2017-01-01 database state and the same `facilities_studies.date <= origin` rule as later origins.
""")
    locked_append(cache / "features_history.md", f"""
### Monthly discrete-time reporting hazard and conditional roster recalibration
- run/experiment: generic_exp_3 lane 1 | status: {'TESTED-KEPT' if metrics.get('reporting_system', {}).get('gate_c', {}).get('p_delta_mae_below_zero', 0) >= 0.8 else 'TESTED-REJECTED'}
- what: Twelve-bin at-risk LightGBM reporting hazard, conditional reporter-subset likelihood with six roster features, 8,192-draw large-roster decoding, and a gated facility meta-median.
- outcome: rolling gates {json.dumps(metrics.get('reporting_system', {}), sort_keys=True)}.
- takeaway: hyperparameters use 2017-2018 and the 2019 gate is opened once; the 2020 official labels remain report-only for Model A.
""")


# Orchestration

def main() -> None:
    debug = is_debug()
    cache = shared_cache_dir()
    output = run_data_dir()
    report(
        "versions",
        python=platform.python_version(),
        numpy=np.__version__,
        pandas=pd.__version__,
        sklearn=sklearn.__version__,
        lightgbm=lightgbm.__version__,
        duckdb=duckdb.__version__,
        debug=debug,
        cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    context = load_task()
    report("data_loaded", train=len(context.train), val=len(context.val), test=len(context.test))
    prediction_root = candidate_cache_path(cache, debug)
    if not debug and load_cached_candidate(prediction_root, output):
        return
    bundle, bundle_state = build_bundle(context, cache, debug)
    report(
        "trial_features",
        state=bundle_state,
        rows=len(bundle.pairs),
        columns=bundle.features.shape[1],
        replay=len(bundle.replay),
        rate=f"{bundle.table_coverage.get('feature_rows_per_minute', 0.0):.1f}",
    )
    registry_matrix, registry = cached_registry_matrix(context.db, bundle.pairs, cache)
    if registry.get("state") == "ready":
        registry_columns = [column for column in registry_matrix if column not in bundle.features]
        bundle.features = pd.concat([bundle.features.reset_index(drop=True), registry_matrix[registry_columns].reset_index(drop=True)], axis=1)
        for column in REGISTRY_CATEGORICAL:
            if column in registry_matrix:
                bundle.features[column] = registry_matrix[column].reset_index(drop=True)
            if column in bundle.features:
                bundle.features[column] = bundle.features[column].fillna("__missing__").astype("category")
    relative_columns = 0
    direct_poles, direct_gate = target_exact_direct_gate(bundle)
    bundle.direct_poles = direct_poles
    report(
        "snapshot_stage",
        state=registry.get("state"),
        columns=registry_matrix.shape[1],
        linked=sum(value.get("linked", 0) for value in registry.get("reports", {}).values()),
        direct_positive=sum(value.get("direct_positive", 0) for value in registry.get("reports", {}).values()),
        direct_negative=sum(value.get("direct_negative", 0) for value in registry.get("reports", {}).values()),
        poles=direct_poles,
        relative_columns=relative_columns,
    )
    rolling = ensure_rolling_origin(context, bundle, cache, debug)
    report("rolling_origin_2017", **rolling)
    facility_seeds, facility_features, facility_state = build_facility_features(context, bundle.event_info, cache, debug)
    report("facility_features", state=facility_state, rows=len(facility_seeds), columns=facility_features.shape[1])
    hosted = run_hosted_measurement(bundle, cache, debug)
    modernbert = modernbert_measurement(cache)
    report(
        "modernbert_stage",
        state=modernbert["state"],
        encoder="answerdotai/ModernBERT-large",
        rows=modernbert.get("rows", 0),
        auc=modernbert.get("auc"),
        blend_auc_delta=modernbert.get("blend_auc_delta"),
    )
    reporting_system, reporting_state, reporting_folds, conditional_calibrator = run_reporting_gates(
        bundle, context, facility_seeds, facility_features, cache, debug
    )
    report(
        "reporting_gates",
        rounds=reporting_system["selected_rounds"],
        ridge=reporting_system["selected_ridge"],
        binary_2019=f"{reporting_system['fold_mae']['2019']['binary']:.6f}",
        hazard_2019=f"{reporting_system['fold_mae']['2019']['hazard_raw']:.6f}",
        conditional_2019=f"{reporting_system['fold_mae']['2019']['hazard_conditional']:.6f}",
        gate_a=f"{reporting_system['gate_a']['p_delta_mae_below_zero']:.3f}",
        gate_b=f"{reporting_system['gate_b']['p_delta_mae_below_zero']:.3f}",
        gate_c=f"{reporting_system['gate_c']['p_delta_mae_below_zero']:.3f}",
    )
    validation_partial, validation_decoded, validation_fold, model_a_calibrator = final_reporting_prediction(
        bundle,
        facility_seeds,
        facility_features,
        context,
        reporting_state,
        reporting_folds,
        conditional_calibrator,
        2020,
        "val",
        debug,
    )
    validation_fit = "Model_A_monthly_replay_windows_closed_by_2020-01-01_and_2017-2019_train_roster_labels_only"
    validation_checksum = prediction_checksum(validation_partial)
    report("model_a_frozen", rows=len(validation_partial), checksum=validation_checksum, label_fit=validation_fit)
    test_partial, test_decoded, test_fold, model_b_calibrator = final_reporting_prediction(
        bundle,
        facility_seeds,
        facility_features,
        context,
        reporting_state,
        reporting_folds,
        conditional_calibrator,
        2021,
        "test",
        debug,
    )
    test_fit = "Model_B_monthly_replay_windows_closed_by_2021-01-01_plus_2020_validation_roster_labels"
    if debug:
        center = float(context.train.df[context.target_col].mean())
        validation = np.full(len(context.val), center, dtype=np.float64)
        test = np.full(len(context.test), center, dtype=np.float64)
        validation[:len(validation_partial)] = validation_partial
        test[:len(test_partial)] = test_partial
    else:
        validation = validation_partial.astype(np.float64)
        test = test_partial.astype(np.float64)
    validation = np.clip(validation, 0.0, 1.0)
    test = np.clip(test, 0.0, 1.0)
    save_predictions(validation, test)
    subprocess.run([sys.executable, "kapso_datasets/check_predictions.py"], check=True)
    diagnostics: dict[str, Any] = {
        "version": HAZARD_VERSION,
        "debug": debug,
        "hosted": hosted,
        "reporting_system": reporting_system,
        "model_a_conditional_calibrator": calibrator_record(model_a_calibrator),
        "model_b_conditional_calibrator": calibrator_record(model_b_calibrator),
        "validation_checksum": prediction_checksum(validation),
        "validation_prediction_label_fit": validation_fit,
        "test_prediction_label_fit": test_fit,
        "snapshot_state": registry,
        "snapshot_direct_gate": direct_gate,
        "modernbert": modernbert,
    }
    if not debug:
        labels = context.val.df[context.target_col].to_numpy(dtype=np.float64)
        incumbent_path = cache / "predictions" / "run_0002_snapshot_exact" / "val_predictions.npy"
        candidates = {
            "selected": validation,
            "lattice_median": validation_decoded["median"],
            "conditional_mean": validation_decoded["mean"],
            "direct_l1": validation_decoded["fallback"],
        }
        if incumbent_path.exists():
            candidates["run_0002_incumbent"] = np.load(incumbent_path, allow_pickle=False)
        diagnostics["resolution"] = connected_bootstrap(labels, candidates, bundle.edges[2020], draws=100)
        val_mask = facility_seeds["split"].eq("val") & facility_seeds["timestamp"].dt.year.eq(2020)
        val_features = facility_features.loc[val_mask].reset_index(drop=True)
        diagnostics["slices"] = slice_diagnostics(labels, validation, val_features, validation_decoded)
        train_2019 = context.train.df[context.train.df["timestamp"].dt.year.eq(2019)]
        diagnostics["representativeness"] = {
            "train_2019_rows": int(len(train_2019)),
            "train_2019_label_mean": float(train_2019[context.target_col].mean()),
            "validation_2020_rows": int(len(context.val)),
            "validation_2020_label_mean": float(labels.mean()),
            "test_2021_rows": int(len(context.test)),
            "test_prediction_mean": float(test.mean()),
        }
        report(
            "report_only_validation_diagnostics",
            bootstrap_se=f"{diagnostics['resolution']['bootstrap_standard_error']:.6f}",
            pairwise_rank_correlation=f"{diagnostics['resolution']['mean_pairwise_rank_correlation']:.6f}",
            selected_mae=f"{diagnostics['resolution']['candidate_mae']['selected']:.6f}",
            slices=len(diagnostics["slices"]),
        )
        record_campaign_memory(cache, bundle, diagnostics)
        prediction_root.mkdir(parents=True, exist_ok=True)
        np.save(prediction_root / "val_predictions.npy", validation)
        np.save(prediction_root / "test_predictions.npy", test)
        (prediction_root / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=str) + "\n")
        register_artifact(cache, {
            "name": "lane 1 READY quarterly target-exact replay",
            "path": str(Path("lane0_target_exact_stage1/bcd3d8470814da67")),
            "description": "Verified quarterly replay with 706,991 trial-origin pairs, 358,010 sampled replay rows, exact dates, all-table features, and cutoff-visible rosters",
            "content_key": "target_exact_quarterly_v3:bcd3d8470814da67",
            "rebuild_hint": "Reuse the READY directory; rebuild only if its content-key inputs change.",
        })
        register_artifact(cache, {
            "name": "lane 1 monthly hazard person periods and OOF vectors",
            "path": str(Path("hazard_lane1") / HAZARD_VERSION / "full"),
            "description": "Twelve-bin person periods, cached 2017-2019 OOF hazard shapes, and the physical 2017 roster extension",
            "content_key": f"{HAZARD_VERSION}:quarterly-v3",
            "rebuild_hint": "Run main.py with the READY quarterly bundle and verified registry cache.",
        })
        register_artifact(cache, {
            "name": f"lane 1 {HAZARD_VERSION} predictions",
            "path": str(prediction_root.relative_to(cache)),
            "description": "Frozen Model A validation predictions, Model B test predictions, rolling gates, slices, hazard shapes, and conditional recalibrator diagnostics",
            "content_key": f"{HAZARD_VERSION}:{prediction_root.name.rsplit('_', 1)[-1]}",
            "rebuild_hint": "Run main.py with matching source and artifact content keys.",
        })
        (prediction_root / "READY").write_text("ready\n")
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, sort_keys=True, default=str) + "\n")
    report(
        "complete",
        val_shape=validation.shape,
        test_shape=test.shape,
        val_mean=f"{validation.mean():.6f}",
        test_mean=f"{test.mean():.6f}",
        validation_checksum=prediction_checksum(validation),
    )


if __name__ == "__main__":
    main()
