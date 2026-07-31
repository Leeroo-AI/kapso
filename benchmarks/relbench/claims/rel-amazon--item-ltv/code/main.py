from __future__ import annotations

# Imports

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from feature_pipeline import (
    append_locked,
    build_feature_frame,
    data_paths,
    initialize_living_documents,
    load_static_products,
    load_task_frames,
    phase_log,
    record_living_outcome,
    reliability_diagnostics,
    verify_target_support,
)
from model_pipeline import train_prediction_chains


# Orchestration

def main() -> None:
    warnings.filterwarnings("ignore")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    started = time.time()
    debug = "--debug" in sys.argv
    mode = "debug" if debug else "full"
    cache_dir = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_0"))
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path("output_data_generic_exp_0")
    work_dir.mkdir(parents=True, exist_ok=True)
    initialize_living_documents(cache_dir, mode)
    paths = data_paths()
    train, val, test = load_task_frames(paths)
    phase_log("task tables loaded", started, f"train={len(train)} val={len(val)} test={len(test)} mode={mode}")
    static, product_text = load_static_products(paths)
    support = verify_target_support(train, static)
    diagnostics = reliability_diagnostics(paths)
    print(f"[measurement] target_support={json.dumps(support, sort_keys=True)}", flush=True)
    print(f"[measurement] reliability={json.dumps(diagnostics, sort_keys=True)}", flush=True)
    append_locked(
        cache_dir / "table_information.md",
        "\n### Lane 0 causal count-L1 measurement\n"
        f"- Target support: `{json.dumps(support, sort_keys=True)}`.\n"
        f"- Reliability: `{json.dumps(diagnostics, sort_keys=True)}`.\n"
        "- Both diagnostics determine exclusion of raw global-volume levels; causal product counts and level-invariant ranks, shares, and ratios remain.\n",
    )
    frame, feature_names, categorical_indices, selected_decay, decay_scores = build_feature_frame(
        train,
        val,
        test,
        static,
        product_text,
        paths,
        cache_dir,
        work_dir,
        debug,
    )
    phase_log("feature design frozen", started, f"decay={selected_decay} internal_decay_scores={decay_scores}")
    report = train_prediction_chains(frame, feature_names, categorical_indices, output_dir, debug)
    val_predictions = np.load(output_dir / "val_predictions.npy", allow_pickle=False)
    test_predictions = np.load(output_dir / "test_predictions.npy", allow_pickle=False)
    if val_predictions.shape != (len(val),) or test_predictions.shape != (len(test),):
        raise RuntimeError(
            f"prediction contract shape failure: val={val_predictions.shape} test={test_predictions.shape}"
        )
    if not np.all(np.isfinite(val_predictions)) or not np.all(np.isfinite(test_predictions)):
        raise RuntimeError("prediction contract finiteness failure")
    diagnostics_report = {
        "mode": mode,
        "support": support,
        "reliability": diagnostics,
        "selected_time_decay": selected_decay,
        "time_decay_internal_scores": decay_scores,
        "model": report,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(json.dumps(diagnostics_report, indent=2))
    if not debug:
        fold_values = [
            fold["rounded_mae"] if report["rounded_counts"] else fold["raw_mae"]
            for fold in report["folds"]
            if fold["training_decay"] == report["training_time_decay"]
        ]
        outcome = (
            f"five expanding-origin folds; median dollar MAE={float(np.median(fold_values)):.6f}, "
            f"dispersion={float(np.std(fold_values, ddof=1)):.6f}, iterations={report['final_iterations']}, "
            f"rounded={report['rounded_counts']}, history_decay={selected_decay}, "
            f"training_decay={report['training_time_decay']}"
        )
        record_living_outcome(
            cache_dir,
            "causal count-L1 all-table backbone internal result",
            outcome,
            "selection used training origins only; validation predictions stayed from Model A before Model B used validation labels",
        )
    phase_log(
        "predictions validated and saved",
        started,
        f"val={val_predictions.shape} test={test_predictions.shape} reserve=30m",
    )


if __name__ == "__main__":
    main()
