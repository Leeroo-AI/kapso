from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from feature_pipeline import (
    FEATURE_VERSION,
    artifact_root,
    brute_force_checks,
    build_features,
    ensure_episode_cache,
    ensure_event_cache,
    register_artifact,
)
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from model_pipeline import build_b_blend_and_predict, select_and_build_a


def load_official_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = time.time()
    context = load_task()
    train = context.train.df[["timestamp", "user_id", context.target_col]].copy()
    validation = context.val.df[["timestamp", "user_id", context.target_col]].copy()
    test = context.test.df[["timestamp", "user_id"]].copy()
    validation["row_id"] = np.arange(len(validation), dtype=np.int32)
    test["row_id"] = np.arange(len(test), dtype=np.int32)
    del context
    gc.collect()
    print(f"[load] official train={len(train)} val={len(validation)} test={len(test)} seconds={time.time() - started:.1f}")
    return train, validation, test


def build_labeled_frame(train: pd.DataFrame, validation: pd.DataFrame, episode_path: Path, debug: bool) -> pd.DataFrame:
    dense = pd.read_parquet(episode_path)
    dense["kind"] = np.int8(1)
    train = train.copy()
    validation = validation.drop(columns="row_id").copy()
    train["kind"] = np.int8(0)
    validation["kind"] = np.int8(2)
    official_keys = pd.concat([train[["timestamp", "user_id"]], validation[["timestamp", "user_id"]]], ignore_index=True).drop_duplicates()
    dense = dense.merge(official_keys.assign(_official=np.int8(1)), on=["timestamp", "user_id"], how="left")
    dense = dense[dense["_official"].isna()].drop(columns="_official")
    if debug:
        june = dense[dense["timestamp"] == pd.Timestamp("2018-06-01")]
        february = dense[dense["timestamp"] == pd.Timestamp("2018-02-01")]
        remaining = max(0, 30_000 - len(june))
        if len(february) > remaining:
            february = february.sample(n=remaining, random_state=1337)
        dense = pd.concat([february, june], ignore_index=True)
        labeled = pd.concat([dense, validation], ignore_index=True)
    else:
        labeled = pd.concat([train, dense, validation], ignore_index=True)
    labeled["timestamp"] = pd.to_datetime(labeled["timestamp"])
    labeled["user_id"] = labeled["user_id"].astype(np.int32)
    labeled["num_ratings"] = labeled["num_ratings"].astype(np.float32)
    labeled["kind"] = labeled["kind"].astype(np.int8)
    duplicates = int(labeled.duplicated(["timestamp", "user_id"]).sum())
    if duplicates:
        raise RuntimeError(f"labeled episode deduplication failed: {duplicates} duplicate keys")
    print(f"[episodes] labeled={len(labeled)} train={int((labeled.kind == 0).sum())} dense={int((labeled.kind == 1).sum())} val={int((labeled.kind == 2).sum())}")
    return labeled


def feature_cache_path(shared_cache: Path) -> Path:
    return artifact_root(shared_cache) / "full_feature_matrix"


def create_feature_data(labeled: pd.DataFrame, test: pd.DataFrame, event_root: Path, shared_cache: Path, debug: bool):
    cache = feature_cache_path(shared_cache)
    ready = cache / "ready.json"
    if not debug and ready.exists():
        metadata = json.loads(ready.read_text())
        if metadata.get("labeled_rows") == len(labeled) and metadata.get("test_rows") == len(test):
            matrix = np.load(cache / "matrix.npy", mmap_mode="r")
            timestamps = np.load(cache / "timestamps.npy", mmap_mode="r")
            users = np.load(cache / "users.npy", mmap_mode="r")
            target = np.load(cache / "target.npy", mmap_mode="r")
            kinds = np.load(cache / "kinds.npy", mmap_mode="r")
            names = metadata["names"]
            groups = metadata["groups"]
            print(f"[cache] full feature matrix hit shape={matrix.shape}")
            return matrix, timestamps, users, target, kinds, names, groups
    combined = pd.concat(
        [
            labeled[["timestamp", "user_id"]],
            test[["timestamp", "user_id"]],
        ],
        ignore_index=True,
    )
    matrix, names, groups = build_features(combined, event_root, True)
    brute_force_checks(combined, matrix, names)
    timestamps = labeled["timestamp"].to_numpy(dtype="datetime64[ms]").astype(np.int64)
    users = labeled["user_id"].to_numpy(dtype=np.int32)
    target = labeled["num_ratings"].to_numpy(dtype=np.float32)
    kinds = labeled["kind"].to_numpy(dtype=np.int8)
    if not debug:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix="feature_build_", dir=cache.parent))
        np.save(temporary / "matrix.npy", matrix)
        np.save(temporary / "timestamps.npy", timestamps)
        np.save(temporary / "users.npy", users)
        np.save(temporary / "target.npy", target)
        np.save(temporary / "kinds.npy", kinds)
        metadata = {
            "feature_version": FEATURE_VERSION,
            "labeled_rows": len(labeled),
            "test_rows": len(test),
            "names": names,
            "groups": groups,
        }
        (temporary / "ready.json").write_text(json.dumps(metadata, indent=2))
        if cache.exists():
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, cache)
        register_artifact(shared_cache, f"lane0-full-feature-matrix-{FEATURE_VERSION}", cache, "Dense-origin censored core and exact rolling relational feature matrix.")
    return matrix, timestamps, users, target, kinds, names, groups


def main() -> None:
    overall_started = time.time()
    debug = is_debug()
    shared_cache = shared_cache_dir()
    print(f"[run] mode={'debug' if debug else 'full'} feature_version={FEATURE_VERSION}")
    train, validation, test = load_official_frames()
    event_root = ensure_event_cache(shared_cache)
    episode_path = ensure_episode_cache(shared_cache)
    labeled = build_labeled_frame(train, validation, episode_path, debug)
    del train
    gc.collect()
    matrix_all, timestamps, users, target, kinds, names, groups = create_feature_data(labeled, test, event_root, shared_cache, debug)
    labeled_rows = len(labeled)
    matrix = matrix_all[:labeled_rows]
    test_matrix = matrix_all[labeled_rows:]
    validation_indices = np.flatnonzero(kinds == 2)
    if len(validation_indices) != len(validation):
        raise RuntimeError(f"validation index count mismatch {len(validation_indices)} != {len(validation)}")
    structure, blend_a, validation_prediction, diagnostics_a = select_and_build_a(
        matrix,
        target,
        timestamps,
        kinds,
        names,
        groups,
        validation_indices,
        debug,
    )
    checksum = hashlib.sha256(np.asarray(validation_prediction, dtype=np.float64).tobytes()).hexdigest()
    print(f"[model_a] validation checksum={checksum}")
    blend_b, test_prediction, diagnostics_b = build_b_blend_and_predict(
        matrix,
        target,
        timestamps,
        structure,
        blend_a,
        test_matrix,
        debug,
    )
    validation_prediction = np.maximum(np.asarray(validation_prediction, dtype=np.float64), 0.0)
    test_prediction = np.maximum(np.asarray(test_prediction, dtype=np.float64), 0.0)
    if validation_prediction.shape != (len(validation),) or test_prediction.shape != (len(test),):
        raise RuntimeError(f"prediction shape failure val={validation_prediction.shape} test={test_prediction.shape}")
    save_predictions(validation_prediction, test_prediction)
    metrics = {
        "feature_version": FEATURE_VERSION,
        "feature_count": len(names),
        "selected_wide": structure.use_wide,
        "leaves": structure.leaves,
        "recency_half_life": structure.half_life,
        "tweedie_power": structure.tweedie_power,
        "trees": structure.trees,
        "blend_a": {"names": blend_a.component_names, "coefficients": blend_a.coefficients.tolist(), "intercept": blend_a.intercept},
        "blend_b": {"names": blend_b.component_names, "coefficients": blend_b.coefficients.tolist(), "intercept": blend_b.intercept},
        "validation_checksum": checksum,
        "internal_a": diagnostics_a,
        "internal_b": diagnostics_b,
        "elapsed_seconds": time.time() - overall_started,
    }
    (run_data_dir() / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[run] complete elapsed={time.time() - overall_started:.1f}s")


if __name__ == "__main__":
    main()
