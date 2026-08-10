# Imports

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from f1_model import FeatureStore, diagnostics_json, load_snapshot, predict_pipeline


# Execution

def _predict(
    store: FeatureStore,
    training: pd.DataFrame,
    target: pd.DataFrame,
    debug: bool,
    cutoff: pd.Timestamp,
):
    predictions, diagnostics = predict_pipeline(store, training, target, debug, cutoff)
    return np.asarray(predictions, dtype=np.float64), diagnostics


def _compact(diagnostics: dict) -> dict:
    if "validation" in diagnostics:
        return {name: _compact(value) for name, value in diagnostics.items()}
    candidates = {
        name: {
            key: values.get(key)
            for key in ("mean", "delta", "bootstrap_se", "wins", "passes")
        }
        for name, values in diagnostics.get("candidate_auc", {}).items()
    }
    return {
        "selected": diagnostics.get("selected"),
        "candidate_auc": candidates,
        "slices": diagnostics.get("slices", {}),
        "pseudo": diagnostics.get("pseudo", {}),
        "label_check": diagnostics.get("label_check", {}),
    }


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    cache_root = Path(os.environ["RELBENCH_CACHE_DIR"])
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    output = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_5"))
    output.mkdir(parents=True, exist_ok=True)
    database, splits = load_snapshot(cache_root, dataset_name, task_name)
    loaded = time.time()
    store = FeatureStore(database)
    featured = time.time()
    rolling = len(splits["val"]) == 0
    if rolling:
        target = splits["test"]
        cutoff = pd.Timestamp(target["date"].min())
        test_predictions, diagnostics = _predict(store, splits["train"], target, debug, cutoff)
        np.save(output / "test_predictions.npy", test_predictions)
        val_shape = "rolling"
    else:
        validation = splits["val"]
        val_cutoff = pd.Timestamp(validation["date"].min())
        val_predictions, val_diagnostics = _predict(
            store, splits["train"], validation, debug, val_cutoff
        )
        combined = pd.concat([splits["train"], splits["val"]], ignore_index=True)
        latest_database_date = max(
            pd.Timestamp(frame["date"].max())
            for frame in database.values()
            if "date" in frame and len(frame)
        )
        test_predictions, diagnostics = _predict(
            store, combined, splits["test"], debug, latest_database_date
        )
        np.save(output / "val_predictions.npy", val_predictions)
        np.save(output / "test_predictions.npy", test_predictions)
        val_shape = str(val_predictions.shape)
        diagnostics = {"validation": val_diagnostics, "test": diagnostics}
    finished = time.time()
    print(
        f"[f1] phase_seconds load={loaded-started:.3f} features={featured-loaded:.3f} "
        f"model={finished-featured:.3f} total={finished-started:.3f} val={val_shape} "
        f"test={test_predictions.shape} diagnostics={diagnostics_json(_compact(diagnostics))}"
    )


if __name__ == "__main__":
    main()
