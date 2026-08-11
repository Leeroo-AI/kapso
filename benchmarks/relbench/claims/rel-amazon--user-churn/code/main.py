from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from kapso_datasets.common import save_predictions
from renewal import (
    add_causal_relational_targets,
    add_customer_name_features,
    add_renewal_transforms,
    choose_rank_blend,
    ensure_features,
    expanding_oof,
    feature_columns,
    labels_frame,
    load_feature_frame,
    matrix,
    rank_values,
    split_path,
    temporal_weights,
    train_model,
    write_diagnostics,
)
from temporal_gnn import expanding_graph_oof, fit_graph_final, graph_smoke_test


def announce(phase: str, started: float, detail: str = "") -> None:
    elapsed = time.time() - started
    suffix = f" {detail}" if detail else ""
    print(f"[lane3] phase={phase} elapsed={elapsed:.1f}s{suffix}", flush=True)


def full_length(split: str) -> int:
    import duckdb

    return int(duckdb.sql(f"SELECT count(*) FROM read_parquet('{split_path(split)}')").fetchone()[0])


def debug_run(started: float, diagnostics: dict) -> tuple[np.ndarray, np.ndarray]:
    paths, timings = ensure_features("train", debug=True)
    frame = load_feature_frame(paths)
    labels = labels_frame()
    y = label_vector(labels, frame)
    frame, _, _ = add_causal_relational_targets(frame, y)
    frame = add_renewal_transforms(frame)
    frame, = add_customer_name_features(frame)
    columns = feature_columns(frame)
    renewal, renewal_mask, renewal_folds = expanding_oof(frame, labels, columns, debug=True)
    graph, graph_mask, graph_folds = expanding_graph_oof(frame, labels, debug=True)
    diagnostics["debug_feature_timings"] = timings
    diagnostics["debug_renewal_folds"] = renewal_folds
    diagnostics["debug_graph_folds"] = graph_folds
    diagnostics["debug_overlap"] = int((renewal_mask & graph_mask).sum())
    val = np.full(full_length("val"), 0.624, dtype=np.float32)
    test = np.full(full_length("test"), 0.624, dtype=np.float32)
    announce("debug_pipeline", started, f"seeds={len(frame)} renewal_oof={renewal_mask.sum()} graph_oof={graph_mask.sum()}")
    return val, test


def slice_scores(frame: pd.DataFrame, y: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> list[dict]:
    history = frame["history_label_n"].to_numpy()
    activity = frame["n_91"].to_numpy()
    groups = {
        "history_0": history == 0,
        "history_1_2": (history >= 1) & (history <= 2),
        "history_3_plus": history >= 3,
        "activity_1": activity == 1,
        "activity_2_3": (activity >= 2) & (activity <= 3),
        "activity_4_plus": activity >= 4,
    }
    rows = []
    for name, group in groups.items():
        selected = mask & np.asarray(group)
        if selected.sum() and np.unique(y[selected]).size == 2:
            rows.append({"stratum": name, "count": int(selected.sum()), "auc": float(roc_auc_score(y[selected], prediction[selected]))})
    return rows


def label_vector(labels: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    indexed = labels.set_index("row_id")["churn"]
    return indexed.loc[frame["row_id"]].to_numpy(np.int8)


def read_split_labels(split: str) -> np.ndarray:
    import duckdb

    return duckdb.sql(f"SELECT churn FROM read_parquet('{split_path(split)}')").fetchnumpy()["churn"].astype(np.int8)


def full_run(started: float, diagnostics: dict) -> tuple[np.ndarray, np.ndarray]:
    train_paths, train_timings = ensure_features("train")
    val_paths, val_timings = ensure_features("val")
    test_paths, test_timings = ensure_features("test")
    announce("feature_materialization", started, f"train_rows={sum(row['rows'] for row in train_timings)}")
    train_frame = load_feature_frame(train_paths)
    val_frame = load_feature_frame(val_paths)
    test_frame = load_feature_frame(test_paths)
    labels = labels_frame()
    y_train = label_vector(labels, train_frame)
    val_labels = read_split_labels("val")
    train_frame, val_frame, test_frame = add_causal_relational_targets(train_frame, y_train, val_frame, val_labels, test_frame)
    train_frame = add_renewal_transforms(train_frame)
    val_frame = add_renewal_transforms(val_frame)
    test_frame = add_renewal_transforms(test_frame)
    train_frame, val_frame, test_frame = add_customer_name_features(train_frame, val_frame, test_frame)
    columns = feature_columns(train_frame)
    announce("feature_load", started, f"features={len(columns)}")
    renewal_oof, renewal_mask, renewal_folds = expanding_oof(train_frame, labels, columns)
    announce("renewal_oof", started, f"folds={len(renewal_folds)}")
    graph_oof, graph_mask, graph_folds = expanding_graph_oof(train_frame, labels)
    overlap = renewal_mask & graph_mask
    blend_weight, blend_results = choose_rank_blend(
        y_train,
        train_frame["origin_index"].to_numpy(),
        renewal_oof,
        graph_oof,
        overlap,
    )
    announce("graph_referee", started, f"weight={blend_weight:.1f} folds={len(graph_folds)}")
    x_train = matrix(train_frame, columns)
    x_val = matrix(val_frame, columns)
    model_a = train_model(x_train, y_train, temporal_weights(train_frame["origin_index"].to_numpy()), rounds=520)
    val_renewal = model_a.predict_proba(x_val)[:, 1].astype(np.float32)
    frozen_val = val_renewal.copy()
    val_for_b = val_frame.copy()
    val_for_b["origin_index"] = int(train_frame["origin_index"].max()) + 1
    train_b = pd.concat([train_frame, val_for_b], ignore_index=True)
    y_b = np.concatenate([y_train, val_labels])
    x_b = matrix(train_b, columns)
    x_test = matrix(test_frame, columns)
    model_b = train_model(x_b, y_b, temporal_weights(train_b["origin_index"].to_numpy()), rounds=520)
    test_renewal = model_b.predict_proba(x_test)[:, 1].astype(np.float32)
    announce("two_model_renewal", started, f"model_a={len(y_train)} model_b={len(y_b)}")
    if blend_weight > 0:
        val_frame.attrs["timestamp"] = "2015-10-01 00:00:00"
        test_frame.attrs["timestamp"] = "2016-01-01 00:00:00"
        val_graph = fit_graph_final(train_frame, y_train, val_frame)
        graph_train_b = train_b.copy()
        test_graph = fit_graph_final(graph_train_b, y_b, test_frame)
        frozen_val = ((1.0 - blend_weight) * rank_values(frozen_val) + blend_weight * rank_values(val_graph)).astype(np.float32)
        test_renewal = ((1.0 - blend_weight) * rank_values(test_renewal) + blend_weight * rank_values(test_graph)).astype(np.float32)
        announce("two_model_graph", started, f"weight={blend_weight:.1f}")
    diagnostics.update(
        {
            "feature_timings": train_timings + val_timings + test_timings,
            "feature_count": len(columns),
            "renewal_folds": renewal_folds,
            "graph_folds": graph_folds,
            "blend_weight": blend_weight,
            "blend_results": blend_results,
            "oof_slices": slice_scores(train_frame, y_train, renewal_oof, renewal_mask),
            "model_a_validation_source": "train labels only",
            "model_b_test_source": "train plus validation labels",
        }
    )
    return frozen_val, test_renewal


def main() -> None:
    warnings.filterwarnings("ignore")
    started = time.time()
    debug = "--debug" in sys.argv
    output = Path("output_data_generic_exp_3")
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = {"debug": debug, "smoke": graph_smoke_test()}
    announce("graph_smoke", started, f"violations={diagnostics['smoke']['post_seed_violations']}")
    if debug:
        val, test = debug_run(started, diagnostics)
    else:
        val, test = full_run(started, diagnostics)
    val = np.clip(np.nan_to_num(val, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    test = np.clip(np.nan_to_num(test, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    save_predictions(val, test)
    diagnostics["elapsed_seconds"] = time.time() - started
    diagnostics["val_shape"] = list(val.shape)
    diagnostics["test_shape"] = list(test.shape)
    run_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", output))
    write_diagnostics(run_dir / "metrics.json", diagnostics)
    write_diagnostics(output / ("debug_metrics.json" if debug else "metrics.json"), diagnostics)
    announce("complete", started, f"val={val.shape} test={test.shape}")


if __name__ == "__main__":
    main()
