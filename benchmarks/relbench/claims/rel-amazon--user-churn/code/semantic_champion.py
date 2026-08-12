from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from numba import njit, prange, set_num_threads
from sklearn.metrics import roc_auc_score

from champion_runner import load_modules
from temporal_transformer import bootstrap_support, cache_root, rank_values


SEMANTIC_NAMES = ("reader_intent", "engagement", "sentiment", "specificity", "risk", "future_intent")


@njit(cache=True, parallel=True)
def summarize_semantics(starts, sizes, semantic):
    result = np.zeros((len(starts), 36), dtype=np.float32)
    for row in prange(len(starts)):
        end = starts[row] + sizes[row]
        if sizes[row] <= 0:
            continue
        for feature in range(6):
            result[row, feature] = semantic[end - 1, feature]
            for group, window in enumerate((4, 16, 64)):
                beginning = max(starts[row], end - window)
                total = 0.0
                for index in range(beginning, end):
                    total += semantic[index, feature]
                result[row, 6 + group * 6 + feature] = total / max(1, end - beginning)
            maximum = -1e9
            minimum = 1e9
            for index in range(starts[row], end):
                maximum = max(maximum, semantic[index, feature])
                minimum = min(minimum, semantic[index, feature])
            result[row, 24 + feature] = maximum
            result[row, 30 + feature] = minimum
    return result


def feature_names() -> list[str]:
    return (
        [f"llm_last_{name}" for name in SEMANTIC_NAMES]
        + [f"llm_mean4_{name}" for name in SEMANTIC_NAMES]
        + [f"llm_mean16_{name}" for name in SEMANTIC_NAMES]
        + [f"llm_mean64_{name}" for name in SEMANTIC_NAMES]
        + [f"llm_max64_{name}" for name in SEMANTIC_NAMES]
        + [f"llm_min64_{name}" for name in SEMANTIC_NAMES]
    )


def ensure_semantic_seed_features(split: str) -> np.ndarray:
    output = cache_root() / f"semantic_seed_{split}_ordered_v1.npy"
    seed_root = cache_root() / f"seed_arrays_{split}_full_ordered_v2"
    starts = np.load(seed_root / "starts.npy", mmap_mode="r")
    sizes = np.load(seed_root / "sizes.npy", mmap_mode="r")
    if output.exists():
        values = np.load(output, mmap_mode="r")
        if values.shape == (len(starts), 36):
            return values
    semantic = np.asarray(np.load(cache_root() / "event_semantics_full.npy", mmap_mode="r"), dtype=np.float32)
    set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    values = summarize_semantics(starts, sizes, semantic)
    temporary = output.with_name(f"semantic_seed_{split}_{os.getpid()}.npy")
    np.save(temporary, values)
    os.replace(temporary, output)
    return np.load(output, mmap_mode="r")


def add_semantics(frame: pd.DataFrame, values: np.ndarray) -> pd.DataFrame:
    semantic_frame = pd.DataFrame(np.asarray(values), columns=feature_names())
    return pd.concat([frame.reset_index(drop=True), semantic_frame], axis=1)


def label_vector(labels: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    return labels.set_index("row_id")["churn"].loc[frame["row_id"]].to_numpy(np.int8)


def semantic_oof(renewal, frame: pd.DataFrame, y: np.ndarray, columns: list[str]) -> tuple[np.ndarray, list[dict]]:
    path = cache_root() / "semantic_champion_oof_ordered_v1.npz"
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        return cached["prediction"], json.loads(str(cached["rows"]))
    x = renewal.matrix(frame, columns)
    origins = frame["origin_index"].to_numpy()
    weights = renewal.temporal_weights(origins)
    prediction = np.full(len(frame), np.nan, dtype=np.float32)
    rows = []
    for fold in renewal.OOF_ORIGIN_INDICES:
        train_mask = origins < fold
        hold_mask = origins == fold
        model = renewal.train_model(x[train_mask], y[train_mask], weights[train_mask], rounds=520)
        values = model.predict_proba(x[hold_mask])[:, 1].astype(np.float32)
        prediction[hold_mask] = values
        score = float(roc_auc_score(y[hold_mask], values))
        rows.append({"fold": fold, "train_n": int(train_mask.sum()), "hold_n": int(hold_mask.sum()), "auc": score})
        print(f"[lane2-semantic] fold={fold} auc={score:.6f}", flush=True)
    temporary = path.with_name(f"semantic_champion_oof_{os.getpid()}.npz")
    np.savez_compressed(temporary, prediction=prediction, rows=np.array(json.dumps(rows)))
    os.replace(temporary, path)
    return prediction, rows


def select_semantic_blend(y, origins, baseline, semantic):
    valid = np.isfinite(baseline) & np.isfinite(semantic)
    rows = []
    for weight in np.linspace(0, 1, 11):
        scores = []
        for fold in (20, 24, 30):
            mask = valid & (origins == fold)
            values = (1 - weight) * rank_values(baseline[mask]) + weight * rank_values(semantic[mask])
            scores.append(float(roc_auc_score(y[mask], values)))
        rows.append({"weight": float(weight), "fold_auc": scores, "median_auc": float(np.median(scores)), "worst_auc": float(np.min(scores))})
    rows.sort(key=lambda row: (row["median_auc"], row["worst_auc"]), reverse=True)
    proposed = rows[0]
    baseline_row = next(row for row in rows if row["weight"] == 0)
    supports = []
    errors = []
    for fold in (20, 24, 30):
        mask = valid & (origins == fold)
        base = rank_values(baseline[mask])
        challenger = (1 - proposed["weight"]) * base + proposed["weight"] * rank_values(semantic[mask])
        support, error = bootstrap_support(y[mask], challenger, base, 100)
        supports.append(support)
        errors.append(error)
    admitted = proposed["weight"] > 0 and proposed["median_auc"] > baseline_row["median_auc"] and np.median(supports) >= 0.8
    gate = {
        "admitted": bool(admitted),
        "weight": float(proposed["weight"] if admitted else 0),
        "proposed_weight": proposed["weight"],
        "median_support": float(np.median(supports)),
        "median_paired_se": float(np.median(errors)),
        "baseline_median": baseline_row["median_auc"],
        "proposed_median": proposed["median_auc"],
    }
    return gate, rows


def build_frames():
    renewal, _ = load_modules()
    train_paths, _ = renewal.ensure_features("train")
    val_paths, _ = renewal.ensure_features("val")
    test_paths, _ = renewal.ensure_features("test")
    train = renewal.load_feature_frame(train_paths)
    val = renewal.load_feature_frame(val_paths)
    test = renewal.load_feature_frame(test_paths)
    labels = renewal.labels_frame()
    y_train = label_vector(labels, train)
    val_y = duckdb.sql(f"SELECT churn FROM read_parquet('{renewal.split_path('val')}')").fetchnumpy()["churn"].astype(np.int8)
    train, val, test = renewal.add_causal_relational_targets(train, y_train, val, val_y, test)
    train = renewal.add_renewal_transforms(train)
    val = renewal.add_renewal_transforms(val)
    test = renewal.add_renewal_transforms(test)
    train, val, test = renewal.add_customer_name_features(train, val, test)
    train = add_semantics(train, ensure_semantic_seed_features("train"))
    val = add_semantics(val, ensure_semantic_seed_features("val"))
    test = add_semantics(test, ensure_semantic_seed_features("test"))
    return renewal, train, val, test, y_train, val_y


def run_experiment() -> dict:
    started = time.time()
    renewal, train, val, test, y_train, val_y = build_frames()
    columns = renewal.feature_columns(train)
    prediction, fold_rows = semantic_oof(renewal, train, y_train, columns)
    champion = np.load(Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / "lane2_champion_reproduction_v1" / "predictions_full.npz", allow_pickle=False)
    gate, blends = select_semantic_blend(y_train, train["origin_index"].to_numpy(), champion["oof"], prediction)
    diagnostics = {"gate": gate, "folds": fold_rows, "blends": blends, "elapsed_seconds": time.time() - started}
    path = cache_root() / "semantic_champion_experiment.json"
    path.write_text(json.dumps(diagnostics, indent=2))
    print(f"[lane2-semantic] gate={json.dumps(gate)}", flush=True)
    return diagnostics


def final_predictions(champion: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    experiment_path = cache_root() / "semantic_champion_experiment.json"
    if not experiment_path.exists():
        diagnostics = run_experiment()
    else:
        diagnostics = json.loads(experiment_path.read_text())
    gate = diagnostics["gate"]
    if not gate["admitted"]:
        return np.asarray(champion["val"], dtype=np.float32), np.asarray(champion["test"], dtype=np.float32), diagnostics
    output = cache_root() / "semantic_champion_final_ordered_v1.npz"
    if output.exists():
        cached = np.load(output, allow_pickle=False)
        semantic_val = cached["val"]
        semantic_test = cached["test"]
    else:
        renewal, train, val, test, y_train, val_y = build_frames()
        columns = renewal.feature_columns(train)
        model_a = renewal.train_model(
            renewal.matrix(train, columns),
            y_train,
            renewal.temporal_weights(train["origin_index"].to_numpy()),
            rounds=520,
        )
        semantic_val = model_a.predict_proba(renewal.matrix(val, columns))[:, 1].astype(np.float32).copy()
        val_for_b = val.copy()
        val_for_b["origin_index"] = int(train["origin_index"].max()) + 1
        train_b = pd.concat([train, val_for_b], ignore_index=True)
        y_b = np.concatenate([y_train, val_y])
        model_b = renewal.train_model(
            renewal.matrix(train_b, columns),
            y_b,
            renewal.temporal_weights(train_b["origin_index"].to_numpy()),
            rounds=520,
        )
        semantic_test = model_b.predict_proba(renewal.matrix(test, columns))[:, 1].astype(np.float32)
        temporary = output.with_name(f"semantic_champion_final_{os.getpid()}.npz")
        np.savez_compressed(temporary, val=semantic_val, test=semantic_test)
        os.replace(temporary, output)
    weight = float(gate["weight"])
    val_prediction = ((1 - weight) * rank_values(champion["val"]) + weight * rank_values(semantic_val)).astype(np.float32)
    test_prediction = ((1 - weight) * rank_values(champion["test"]) + weight * rank_values(semantic_test)).astype(np.float32)
    diagnostics["semantic_validation_prediction_correlation"] = float(np.corrcoef(rank_values(semantic_val), rank_values(champion["val"]))[0, 1])
    diagnostics["model_a_source"] = "train labels only"
    diagnostics["model_b_source"] = "train plus validation labels"
    return val_prediction, test_prediction, diagnostics


if __name__ == "__main__":
    run_experiment()
