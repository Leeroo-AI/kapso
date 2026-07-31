from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.tasks import get_task

from feature_pipeline import FeatureBuilder, load_feature_frames, register_artifact
from model_pipeline import (
    bank_core_probe,
    feature_columns,
    final_predictions,
    gate_additional_models,
    run_lgb_selection,
    stratified_report,
)


warnings.filterwarnings("ignore")


def load_frames() -> tuple[object, dict[str, pd.DataFrame]]:
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    task = get_task(dataset_name, task_name, download=False)
    frames = {
        "train": task.get_table("train").df.copy(),
        "val": task.get_table("val").df.copy(),
        "test": task.get_table("test").df.copy(),
    }
    return task, frames


def json_safe_selection(selection: dict[str, object]) -> dict[str, object]:
    blocked = {"labels", "row_ids", "lgb_predictions", "columns"}
    return {key: value for key, value in selection.items() if key not in blocked}


def json_safe_gate(gate: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in gate.items() if key != "pooled_prediction"}


def write_outputs(
    frames: dict[str, pd.DataFrame],
    selected: dict[str, np.ndarray],
    selected_val_prediction: np.ndarray,
    selected_test_prediction: np.ndarray,
    debug: bool,
    metrics: dict[str, object],
) -> None:
    output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    base = float(frames["train"]["churn"].mean())
    val_prediction = np.full(len(frames["val"]), base, dtype=np.float64)
    test_prediction = np.full(len(frames["test"]), base, dtype=np.float64)
    val_prediction[selected["val"]] = selected_val_prediction
    test_prediction[selected["test"]] = selected_test_prediction
    val_prediction = np.clip(val_prediction, 1e-6, 1 - 1e-6)
    test_prediction = np.clip(test_prediction, 1e-6, 1 - 1e-6)
    np.save(output / "val_predictions.npy", val_prediction)
    np.save(output / "test_predictions.npy", test_prediction)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    local_output = Path("output_data_generic_exp_0")
    local_output.mkdir(exist_ok=True)
    (local_output / ("debug_metrics.json" if debug else "full_metrics.json")).write_text(
        json.dumps(metrics, indent=2)
    )
    result = subprocess.run(
        [sys.executable, "kapso_datasets/check_predictions.py"],
        check=True,
        text=True,
        capture_output=True,
    )
    for line in result.stdout.splitlines():
        print(line, flush=True)


def rolling_fallback(frames: dict[str, pd.DataFrame]) -> None:
    output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    probability = float(frames["train"]["churn"].mean())
    np.save(output / "test_predictions.npy", np.full(len(frames["test"]), probability, dtype=np.float64))
    print(f"[rolling] emitted {len(frames['test'])} causal-prior fallback predictions", flush=True)


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    task, frames = load_frames()
    print(
        f"[startup] debug={debug} train={len(frames['train'])} val={len(frames['val'])} test={len(frames['test'])}",
        flush=True,
    )
    if len(frames["val"]) == 0:
        rolling_fallback(frames)
        return
    builder = FeatureBuilder(frames, debug)
    core_path = builder.build_core()
    core_frame = load_feature_frames(core_path)
    core_probe = bank_core_probe(core_frame, builder.root, debug)
    core_columns = feature_columns(core_frame)
    del core_frame
    gc.collect()
    wide_path = builder.build_wide()
    priority_path = builder.build_priority()
    priority_two_path = builder.build_priority_two()
    builder.close()
    frame = load_feature_frames(core_path, [priority_path, priority_two_path])
    all_columns = feature_columns(frame)
    print(
        f"[features] rows={len(frame)} core={len(core_columns)} all={len(all_columns)}",
        flush=True,
    )
    selection = run_lgb_selection(frame, core_columns, all_columns, debug)
    gate = gate_additional_models(frame, selection, debug)
    slices = stratified_report(frame, selection, gate)
    selected_val_prediction, selected_test_prediction = final_predictions(frame, selection, gate, debug)
    metrics = {
        "debug": debug,
        "feature_content_key": builder.metadata()["content_key"],
        "core_probe": core_probe,
        "selection": json_safe_selection(selection),
        "gate": json_safe_gate(gate),
        "oof_slices": slices,
        "model_a_training_splits": ["train"],
        "model_b_training_splits": ["train", "val"],
        "elapsed_seconds": time.time() - started,
    }
    write_outputs(
        frames,
        builder.selected,
        selected_val_prediction,
        selected_test_prediction,
        debug,
        metrics,
    )
    register_artifact(Path(os.environ["KAPSO_SHARED_CACHE_DIR"]), builder.metadata())
    print(
        f"[complete] validation predictions frozen before Model B; elapsed={time.time() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
