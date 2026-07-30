from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from feature_pipeline import (
    CACHE_VERSION,
    build_historical_priors,
    build_local_warehouse,
    build_text_neighbor_features,
    build_text_features,
    cache_root,
    load_task_rows,
    register_artifact,
)
from modeling import (
    final_predictions,
    rolling_selection,
    stratified_diagnostics,
)


# Runtime

warnings.filterwarnings("ignore")
START = time.time()


def log_phase(event: str, **values) -> None:
    payload = {
        "event": event,
        "elapsed_seconds": round(time.time() - START, 2),
        **values,
    }
    print(f"[study_adverse] {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def _append_campaign_history(shared_cache: Path, selection) -> None:
    path = shared_cache / "features_history.md"
    lock_path = shared_cache / "features_history.lock"
    lock_path.touch(exist_ok=True)
    import fcntl

    marker = f"generic_exp_0_lane0_{CACHE_VERSION}_measured"
    block = (
        f"\n### All-table rolling ablation measured\n"
        f"- run/experiment: {marker} | status: TESTED-KEPT\n"
        f"- what: structured core versus completed-result cohort profiles versus 96-dimensional word/character TF-IDF SVD\n"
        f"- outcome: rolling MAE {json.dumps(selection.diagnostics['ablation_mae'], sort_keys=True)}; selected profiles={selection.use_profiles}, text={selection.use_text}, text-neighbors={selection.use_text_neighbors}\n"
        f"- takeaway: configuration was frozen exclusively from 2016-2019 train-only origins\n"
        f"\n### Empirical-Bayes prior strength measured\n"
        f"- run/experiment: {marker} | status: TESTED-KEPT\n"
        f"- what: pseudo-counts 5, 15, and 50 over sponsor, agency, condition, intervention, geography, phase, and type priors\n"
        f"- outcome: direct rolling prior MAE {json.dumps(selection.diagnostics['pseudo_count_mae'], sort_keys=True)}; selected {selection.pseudo_count}\n"
        f"- takeaway: use the selected support-dependent shrinkage strength in both frozen final models\n"
    )
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            existing = path.read_text() if path.exists() else ""
            if marker not in existing:
                with path.open("a") as handle:
                    handle.write(block)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    debug = "--debug" in sys.argv
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    shared_cache = Path(
        os.environ.get("KAPSO_SHARED_CACHE_DIR", "./output_data_generic_exp_0/shared_cache")
    )
    shared_cache.mkdir(parents=True, exist_ok=True)
    output = Path(
        os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_0")
    )
    output.mkdir(parents=True, exist_ok=True)
    log_phase(
        "start",
        debug=debug,
        dataset=dataset_name,
        task=task_name,
        cache_version=CACHE_VERSION,
    )

    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    database = dataset.get_db(upto_test_timestamp=True)
    rows = load_task_rows(task)
    log_phase(
        "data_loaded",
        train_rows=int((rows["split"] == "train").sum()),
        val_rows=int((rows["split"] == "val").sum()),
        test_rows=int((rows["split"] == "test").sum()),
        tables=len(database.table_dict),
    )

    warehouse = build_local_warehouse(database, rows, shared_cache, debug=debug)
    log_phase(
        "local_warehouse",
        structured_columns=warehouse["features"].shape[1],
        relation_builders=len(warehouse["memberships"]),
        result_profile_columns=warehouse["result_profiles"].shape[1],
    )

    priors = build_historical_priors(
        rows,
        warehouse["memberships"],
        warehouse["result_profiles"],
        shared_cache,
        debug=debug,
    )
    log_phase("historical_priors", columns=priors.shape[1])

    text_oof = build_text_features(
        warehouse["documents"], rows, shared_cache, mode="oof", debug=debug
    )
    log_phase("rolling_text", dimensions=text_oof.shape[1])
    text_neighbors_oof = build_text_neighbor_features(
        text_oof,
        rows,
        warehouse["features"]["study_enrollment"],
        shared_cache,
        mode="oof",
        debug=debug,
    )
    log_phase("rolling_text_neighbors", dimensions=text_neighbors_oof.shape[1])

    selection = rolling_selection(
        rows,
        warehouse["features"],
        priors,
        text_oof,
        text_neighbors_oof,
        debug,
        log_phase,
    )
    diagnostics = stratified_diagnostics(
        rows,
        warehouse["features"],
        priors,
        selection,
    )
    log_phase(
        "selection_frozen",
        pseudo_count=selection.pseudo_count,
        use_profiles=selection.use_profiles,
        use_text=selection.use_text,
        use_text_neighbors=selection.use_text_neighbors,
        blend_weights=selection.blend_weights,
        upper_cap=selection.upper_cap,
        rolling_metrics=selection.diagnostics["blend_metrics"],
    )

    if (selection.use_text or selection.use_text_neighbors) and not debug:
        text_a = build_text_features(
            warehouse["documents"], rows, shared_cache, mode="a", debug=False
        )
        log_phase("model_a_text", dimensions=text_a.shape[1])
        text_neighbors_a = build_text_neighbor_features(
            text_a,
            rows,
            warehouse["features"]["study_enrollment"],
            shared_cache,
            mode="a",
            debug=False,
        )
        log_phase("model_a_text_neighbors", dimensions=text_neighbors_a.shape[1])
        text_b = build_text_features(
            warehouse["documents"], rows, shared_cache, mode="b", debug=False
        )
        log_phase("model_b_text", dimensions=text_b.shape[1])
        text_neighbors_b = build_text_neighbor_features(
            text_b,
            rows,
            warehouse["features"]["study_enrollment"],
            shared_cache,
            mode="b",
            debug=False,
        )
        log_phase("model_b_text_neighbors", dimensions=text_neighbors_b.shape[1])
    else:
        text_a = text_oof
        text_b = text_oof
        text_neighbors_a = text_neighbors_oof
        text_neighbors_b = text_neighbors_oof

    val_prediction, test_prediction = final_predictions(
        rows,
        warehouse["features"],
        priors,
        text_a,
        text_b,
        text_neighbors_a,
        text_neighbors_b,
        selection,
        debug,
        log_phase,
    )
    expected_val = int((rows["split"] == "val").sum())
    expected_test = int((rows["split"] == "test").sum())
    if val_prediction.shape != (expected_val,) or test_prediction.shape != (expected_test,):
        raise RuntimeError(
            f"prediction shape mismatch val={val_prediction.shape} test={test_prediction.shape}"
        )
    if not np.all(np.isfinite(val_prediction)) or not np.all(np.isfinite(test_prediction)):
        raise RuntimeError("non-finite prediction")
    np.save(output / "val_predictions.npy", val_prediction)
    np.save(output / "test_predictions.npy", test_prediction)
    if not debug:
        prediction_cache = cache_root(shared_cache) / "model_predictions.npz"
        np.savez_compressed(
            prediction_cache,
            val_predictions=val_prediction,
            test_predictions=test_prediction,
        )
        register_artifact(
            shared_cache,
            f"{CACHE_VERSION}_model_predictions",
            prediction_cache,
            "Two-model frozen-pipeline validation and test ensemble predictions.",
        )
    metrics = {
        "debug": debug,
        "cache_version": CACHE_VERSION,
        "selection": {
            "pseudo_count": selection.pseudo_count,
            "use_profiles": selection.use_profiles,
            "use_text": selection.use_text,
            "use_text_neighbors": selection.use_text_neighbors,
            "blend_weights": selection.blend_weights,
            "rounds": selection.rounds,
            "upper_cap": selection.upper_cap,
        },
        "rolling": {
            key: value
            for key, value in selection.diagnostics.items()
            if key not in {"oof_indices", "oof_prediction"}
        },
        "strata": diagnostics,
        "prediction_summary": {
            "val_min": float(val_prediction.min()),
            "val_median": float(np.median(val_prediction)),
            "val_mean": float(val_prediction.mean()),
            "val_max": float(val_prediction.max()),
            "test_min": float(test_prediction.min()),
            "test_median": float(np.median(test_prediction)),
            "test_mean": float(test_prediction.mean()),
            "test_max": float(test_prediction.max()),
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    if not debug:
        _append_campaign_history(shared_cache, selection)
    log_phase(
        "complete",
        val_shape=val_prediction.shape,
        test_shape=test_prediction.shape,
        val_median=float(np.median(val_prediction)),
        test_median=float(np.median(test_prediction)),
    )


if __name__ == "__main__":
    main()
