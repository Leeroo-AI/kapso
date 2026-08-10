# Section: imports

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from relbench.tasks import get_task

from retrieval_pipeline import VERSION, run_solution


# Section: orchestration

def main() -> None:
    warnings.filterwarnings("ignore")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TQDM_DISABLE", "1")
    start = time.time()
    debug = "--debug" in sys.argv
    task = get_task(
        os.environ["RELBENCH_DATASET"],
        os.environ["RELBENCH_TASK"],
        download=False,
    )
    train = task.get_table("train").df.copy()
    val = task.get_table("val").df.copy()
    test = task.get_table("test").df.copy()
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / VERSION
    shared.mkdir(parents=True, exist_ok=True)
    local = Path("output_data_generic_exp_3")
    local.mkdir(parents=True, exist_ok=True)
    cache_name = "debug_predictions.npz" if debug else "full_predictions.npz"
    prediction_cache = shared / cache_name
    diagnostics_cache = shared / ("debug_diagnostics.json" if debug else "full_diagnostics.json")
    if prediction_cache.exists() and diagnostics_cache.exists():
        cached = np.load(prediction_cache, allow_pickle=False)
        val_prediction = cached["val"]
        test_prediction = cached["test"]
        diagnostics = json.loads(diagnostics_cache.read_text())
        print(f"[lane3] loaded cached outputs version={VERSION}", flush=True)
    else:
        val_prediction, test_prediction, diagnostics = run_solution(
            train,
            val,
            test,
            debug,
            shared,
        )
        np.savez_compressed(
            prediction_cache,
            val=val_prediction,
            test=test_prediction,
        )
        diagnostics_cache.write_text(json.dumps(diagnostics, indent=2))
    if val_prediction.shape != (len(val),):
        raise RuntimeError(f"validation shape {val_prediction.shape} != {(len(val),)}")
    if test_prediction.shape != (len(test),):
        raise RuntimeError(f"test shape {test_prediction.shape} != {(len(test),)}")
    if not np.all(np.isfinite(val_prediction)) or not np.all(np.isfinite(test_prediction)):
        raise RuntimeError("non-finite predictions")
    if (
        np.min(val_prediction) < 0.0
        or np.max(val_prediction) > 1.0
        or np.min(test_prediction) < 0.0
        or np.max(test_prediction) > 1.0
    ):
        raise RuntimeError("predictions outside probability range")
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_3"))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "val_predictions.npy", val_prediction)
    np.save(output / "test_predictions.npy", test_prediction)
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (local / ("debug_metrics.json" if debug else "full_metrics.json")).write_text(
        json.dumps(diagnostics, indent=2)
    )
    print(
        f"[lane3] wrote val{val_prediction.shape} test{test_prediction.shape} "
        f"debug={debug} total_seconds={time.time() - start:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
