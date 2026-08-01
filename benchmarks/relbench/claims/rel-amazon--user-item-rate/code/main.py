from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import numba
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numba"], check=True)

from solution import (
    FEATURE_NAMES,
    Snapshot,
    build_embeddings,
    build_semantic_index,
    build_snapshot,
    cache_paths,
    candidate_arrays,
    database_paths,
    debug_predictions,
    load_static,
    predict,
    predict_rrf,
    source_recall,
    train_ranker,
    training_matrix,
    union_recall,
)


# Utilities

def phase(start: float, name: str) -> None:
    print(f"[main] {name}: {time.time() - start:.1f}s elapsed", flush=True)


def task_frame(path: Path, labels: bool) -> pd.DataFrame:
    columns = "customer_id,product_id" if labels else "customer_id"
    return duckdb.sql(f"select {columns} from read_parquet('{path}')").df()


def combine(parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.concatenate([part[index] for part in parts]) for index in range(4))


def assert_predictions(prediction: np.ndarray, rows: int) -> None:
    if prediction.shape != (rows, 10):
        raise RuntimeError(f"wrong prediction shape {prediction.shape}")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise RuntimeError(f"wrong prediction dtype {prediction.dtype}")
    if prediction.min() < 0 or prediction.max() >= 506_012:
        raise RuntimeError("prediction ID outside product range")
    if any(len(np.unique(row)) != 10 for row in prediction):
        raise RuntimeError("duplicate product in prediction row")


def write_predictions(name: str, prediction: np.ndarray) -> None:
    output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / name, prediction.astype(np.int64))


# Pipeline

def main() -> None:
    start = time.time()
    paths = database_paths()
    val = task_frame(paths["val"], True)
    test = task_frame(paths["test"], False)
    debug = "--debug" in sys.argv
    _, cache_root = cache_paths()
    gate_path = cache_root / "train_only_gates.json"
    cached_gates = json.loads(gate_path.read_text()) if gate_path.exists() else None
    static = load_static(include_text=not debug and cached_gates is None)
    phase(start, "loaded task and static tables")

    if debug:
        val_prediction, test_prediction = debug_predictions(
            val["customer_id"].to_numpy(np.int32),
            test["customer_id"].to_numpy(np.int32),
            static,
        )
        assert_predictions(val_prediction, len(val))
        assert_predictions(test_prediction, len(test))
        write_predictions("val_predictions.npy", val_prediction)
        write_predictions("test_predictions.npy", test_prediction)
        phase(start, "debug predictions written")
        return

    if cached_gates is not None:
        use_als = bool(cached_gates["gates"]["als"]["kept"])
        use_semantic = bool(cached_gates["gates"]["semantic"]["kept"])
        val_snapshot = build_snapshot(
            "2015-10-01",
            val["customer_id"].to_numpy(np.int32),
            static,
            None,
            None,
            use_als or use_semantic,
        )
        val_prediction = predict_rrf(val_snapshot, use_als, use_semantic)
        assert_predictions(val_prediction, len(val))
        write_predictions("val_predictions.npy", val_prediction)
        print("[main] validation predictions frozen from train-only-selected RRF Model A", flush=True)
        test_snapshot = build_snapshot(
            "2016-01-01",
            test["customer_id"].to_numpy(np.int32),
            static,
            None,
            None,
            use_als or use_semantic,
        )
        test_prediction = predict_rrf(test_snapshot, use_als, use_semantic)
        assert_predictions(test_prediction, len(test))
        write_predictions("test_predictions.npy", test_prediction)
        output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
        (output / "metrics.json").write_text(json.dumps(cached_gates, indent=2))
        phase(start, "cached train-only gates and RRF predictions written")
        return

    embeddings = build_embeddings(static)
    semantic_index = build_semantic_index(embeddings)
    static.product_text = None
    phase(start, "product representation ready")

    con = duckdb.connect()
    folds = {}
    for cutoff in ["2015-01-01", "2015-07-02"]:
        frame = con.sql(
            f"select customer_id,product_id from read_parquet('{paths['train']}') where timestamp='{cutoff}'::timestamp"
        ).df()
        snapshot = build_snapshot(
            cutoff,
            frame["customer_id"].to_numpy(np.int32),
            static,
            embeddings,
            semantic_index,
            True,
        )
        folds[cutoff] = (frame, snapshot)
        phase(start, f"built forward fold {cutoff}")

    diagnostics = {"source_recall": {}, "union_recall": {}}
    fold_gains = {"als": [], "semantic": []}
    for cutoff, (frame, snapshot) in folds.items():
        recalls = source_recall(snapshot, frame["product_id"])
        base = union_recall(snapshot, frame["product_id"], False, False)
        with_als = union_recall(snapshot, frame["product_id"], True, False)
        with_semantic = union_recall(snapshot, frame["product_id"], True, True)
        diagnostics["source_recall"][cutoff] = recalls
        diagnostics["union_recall"][cutoff] = {
            "core": base,
            "core_als": with_als,
            "core_als_semantic": with_semantic,
        }
        fold_gains["als"].append(with_als - base)
        fold_gains["semantic"].append(with_semantic - with_als)
        print(f"[main] train-only recall {cutoff}: {json.dumps(diagnostics['union_recall'][cutoff])}", flush=True)
        print(f"[main] train-only source slices {cutoff}: {json.dumps(recalls)}", flush=True)
    use_als = float(np.mean(fold_gains["als"])) >= 0.015 and min(fold_gains["als"]) >= -0.005
    use_semantic = float(np.mean(fold_gains["semantic"])) >= 0.015 and min(fold_gains["semantic"]) >= -0.005
    diagnostics["gates"] = {
        "als": {"kept": use_als, "fold_gains": fold_gains["als"]},
        "semantic": {"kept": use_semantic, "fold_gains": fold_gains["semantic"]},
        "review_text": {"kept": False, "reason": "optional block frozen out before full-review encoding"},
    }
    phase(start, "completed predetermined source gates")

    january_frame, january_snapshot = folds["2015-01-01"]
    july_frame, july_snapshot = folds["2015-07-02"]
    january_train = training_matrix(
        january_snapshot, static, january_frame["product_id"], use_als, use_semantic, 60_000
    )
    july_validation = training_matrix(
        july_snapshot, static, july_frame["product_id"], use_als, use_semantic, 40_000
    )
    _, selected_trees = train_ranker(january_train, july_validation, None)
    selected_trees = int(max(40, min(800, selected_trees)))
    diagnostics["selected_trees"] = selected_trees
    model_a_data = combine([january_train, july_validation])
    model_a, _ = train_ranker(model_a_data, None, selected_trees)
    phase(start, f"trained Model A with {selected_trees} trees")

    val_snapshot = build_snapshot(
        "2015-10-01",
        val["customer_id"].to_numpy(np.int32),
        static,
        embeddings,
        semantic_index,
        use_als or use_semantic,
    )
    val_prediction = predict(val_snapshot, static, model_a, use_als, use_semantic)
    assert_predictions(val_prediction, len(val))
    write_predictions("val_predictions.npy", val_prediction)
    print("[main] validation predictions frozen from Model A", flush=True)
    phase(start, "generated and froze validation predictions")

    validation_train = training_matrix(
        val_snapshot, static, val["product_id"], use_als, use_semantic, 60_000
    )
    model_b_data = combine([january_train, july_validation, validation_train])
    model_b, _ = train_ranker(model_b_data, None, selected_trees)
    del january_train, july_validation, validation_train, model_a_data, model_b_data, model_a
    del january_snapshot, july_snapshot, val_snapshot, folds
    gc.collect()
    phase(start, "trained Model B after validation freeze")

    test_snapshot = build_snapshot(
        "2016-01-01",
        test["customer_id"].to_numpy(np.int32),
        static,
        embeddings,
        semantic_index,
        use_als or use_semantic,
    )
    test_prediction = predict(test_snapshot, static, model_b, use_als, use_semantic)
    assert_predictions(test_prediction, len(test))
    write_predictions("test_predictions.npy", test_prediction)
    output = Path(os.environ["KAPSO_RUN_DATA_DIR"])
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    phase(start, "test predictions and diagnostics written")


if __name__ == "__main__":
    main()
