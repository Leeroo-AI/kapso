from __future__ import annotations

import fcntl
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from relbench.datasets import get_dataset
from relbench.tasks import get_task

from feature_layer import FEATURE_VERSION, FeatureBuilder, build_features, model_feature_columns
from modeling import MODEL_VERSION, adaptive_blend, base_frame, fit_all_bases, fixed_blend, select_half_life


warnings.filterwarnings("ignore")


def is_debug() -> bool:
    return "--debug" in sys.argv


def shared_cache() -> Path:
    path = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def output_directory() -> Path:
    path = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./output_data_generic_exp_0"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def register_cache(root: Path) -> None:
    registry = root / "artifacts.json"
    lock_path = root / "artifacts.lock"
    entry = {
        "name": "driver-dnf lane0 causal and prequential cache",
        "path": FEATURE_VERSION,
        "description": "Target-free per-origin all-nine-table causal features and leakage-safe prequential base predictions",
        "content_key": f"{FEATURE_VERSION}:{MODEL_VERSION}",
        "rebuild_hint": "Run the rolling candidate chronologically; each tick extends missing origins",
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            records = json.loads(registry.read_text()) if registry.exists() else []
        except Exception:
            records = []
        if not any(record.get("content_key") == entry["content_key"] for record in records):
            records.append(entry)
            temporary = Path(str(registry) + f".{os.getpid()}.tmp")
            temporary.write_text(json.dumps(records, indent=2))
            os.replace(temporary, registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.reindex(columns=columns).astype(np.float32)
    return output.replace([np.inf, -np.inf], np.nan).reset_index(drop=True)


def load_context():
    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db = dataset.get_db(upto_test_timestamp=False)
    train = task.get_table("train", mask_input_cols=False).df.copy()
    val = task.get_table("val", mask_input_cols=False).df.copy()
    test = task.get_table("test").df.copy()
    for frame in [train, val, test]:
        frame["date"] = pd.to_datetime(frame["date"])
    return task, db, train, val, test


def prediction_cache_paths(root: Path, origin: pd.Timestamp) -> tuple[Path, Path]:
    directory = root / FEATURE_VERSION / "prequential" / MODEL_VERSION
    directory.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp(origin).strftime("%Y%m%dT%H%M%S")
    return directory / f"{stamp}.npz", directory / f"{stamp}.json"


def read_prequential_cache(root: Path, origin: pd.Timestamp, drivers: np.ndarray) -> pd.DataFrame | None:
    data_path, meta_path = prediction_cache_paths(root, origin)
    if not data_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("feature_version") != FEATURE_VERSION or meta.get("model_version") != MODEL_VERSION:
            return None
        if pd.Timestamp(meta["source_cutoff"]) > pd.Timestamp(origin):
            return None
        if pd.Timestamp(meta["auxiliary_cutoff"]) > pd.Timestamp(origin):
            return None
        if pd.Timestamp(meta["task_label_cutoff"]) + pd.Timedelta(days=30) > pd.Timestamp(origin):
            return None
        with np.load(data_path, allow_pickle=False) as values:
            stored_drivers = values["driverId"].astype(int)
            requested_drivers = drivers.astype(int)
            if len(np.unique(stored_drivers)) != len(stored_drivers) or len(stored_drivers) != len(requested_drivers):
                return None
            order = pd.Index(stored_drivers).get_indexer(requested_drivers)
            if np.any(order < 0):
                return None
            return pd.DataFrame({name: values[name][order] for name in ["m1", "m2", "m3", "m4", "disagreement_std", "disagreement_range", "calendar_khat", "rookie", "new_team"]})
    except Exception:
        return None


def write_prequential_cache(root: Path, origin: pd.Timestamp, drivers: np.ndarray, frame: pd.DataFrame, task_cutoff: pd.Timestamp, auxiliary_cutoff: pd.Timestamp) -> None:
    data_path, meta_path = prediction_cache_paths(root, origin)
    temp_data = Path(str(data_path) + f".{os.getpid()}.tmp")
    temp_meta = Path(str(meta_path) + f".{os.getpid()}.tmp")
    arrays = {name: frame[name].to_numpy(dtype=float) for name in ["m1", "m2", "m3", "m4", "disagreement_std", "disagreement_range", "calendar_khat", "rookie", "new_team"]}
    arrays["driverId"] = drivers.astype(int)
    with temp_data.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    meta = {
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "origin": pd.Timestamp(origin).isoformat(),
        "source_cutoff": pd.Timestamp(origin).isoformat(),
        "task_label_cutoff": pd.Timestamp(task_cutoff).isoformat(),
        "auxiliary_cutoff": pd.Timestamp(auxiliary_cutoff).isoformat(),
        "contains_targets": False,
    }
    temp_meta.write_text(json.dumps(meta))
    os.replace(temp_data, data_path)
    os.replace(temp_meta, meta_path)


def generate_prequential(train: pd.DataFrame, train_features: pd.DataFrame, event: pd.DataFrame, event_features: pd.DataFrame, columns: list[str], current_origin: pd.Timestamp, cache_root: Path) -> tuple[pd.DataFrame, dict]:
    closed_limit = pd.Timestamp(current_origin) - pd.Timedelta(days=30)
    trailing_limit = pd.Timestamp(current_origin) - pd.DateOffset(months=30)
    origins = [pd.Timestamp(value) for value in sorted(train.loc[(train["date"] <= closed_limit) & (train["date"] >= trailing_limit), "date"].unique())]
    parts = []
    created = 0
    for origin in origins:
        target_mask = train["date"].eq(origin).to_numpy()
        drivers = train.loc[target_mask, "driverId"].to_numpy(dtype=int)
        cached = read_prequential_cache(cache_root, origin, drivers)
        if cached is None:
            task_mask = (train["date"] + pd.Timedelta(days=30) <= origin).to_numpy()
            event_mask = (event["date"] <= origin).to_numpy()
            if task_mask.sum() < 200 or np.unique(train.loc[task_mask, "did_not_finish"]).size < 2 or event_mask.sum() < 500:
                continue
            task_x = matrix(train_features.loc[task_mask], columns)
            event_x = matrix(event_features.loc[event_mask], columns)
            prediction_x = matrix(train_features.loc[target_mask], columns)
            half_life, _ = select_half_life(task_x, train.loc[task_mask, "did_not_finish"].to_numpy(dtype=int), train.loc[task_mask, "date"].reset_index(drop=True), origin, cache_root, False)
            predictions = fit_all_bases(
                task_x,
                train.loc[task_mask, "did_not_finish"].to_numpy(dtype=int),
                train.loc[task_mask, "date"].reset_index(drop=True),
                event_x,
                event.loc[event_mask, "outcome_class"].to_numpy(dtype=int),
                event.loc[event_mask, "date"].reset_index(drop=True),
                prediction_x,
                train_features.loc[target_mask].reset_index(drop=True),
                origin,
                half_life,
                False,
                True,
            )
            cached = base_frame(predictions, train_features.loc[target_mask].reset_index(drop=True))
            task_cutoff = train.loc[task_mask, "date"].max()
            auxiliary_cutoff = event.loc[event_mask, "date"].max()
            write_prequential_cache(cache_root, origin, drivers, cached, task_cutoff, auxiliary_cutoff)
            created += 1
        cached = cached.copy()
        cached["date"] = origin
        cached["driverId"] = drivers
        cached["target"] = train.loc[target_mask, "did_not_finish"].to_numpy(dtype=int)
        parts.append(cached)
    result = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return result, {"origins": len(origins), "usable_origins": int(result["date"].nunique()) if len(result) else 0, "created": created}


def run_rolling(builder: FeatureBuilder, train: pd.DataFrame, test: pd.DataFrame, cache_root: Path, debug: bool) -> tuple[np.ndarray, dict]:
    phase = time.time()
    origin = pd.Timestamp(test["date"].max())
    model_train = train[train["date"] >= origin - pd.DateOffset(years=2)].reset_index(drop=True) if debug else train.reset_index(drop=True)
    train_features = build_features(builder, model_train[["date", "driverId"]], cache_root, "task", True, None if debug else "full_task")
    test_features = build_features(builder, test[["date", "driverId"]], cache_root, "task", True)
    event = builder.results[["date", "driverId", "outcome_class"]].copy().reset_index(drop=True)
    if debug:
        event = event[event["date"] >= origin - pd.DateOffset(years=2)].reset_index(drop=True)
    event_features = build_features(builder, event[["date", "driverId"]], cache_root, "event", True, None if debug else "full_event")
    columns = model_feature_columns(train_features)
    print(f"[candidate] features rows={len(train_features)}/{len(event_features)}/{len(test_features)} cols={len(columns)} elapsed={time.time() - phase:.2f}s")
    half_life, recency_diagnostics = (8.0, {"selected": "8", "mode": "debug"}) if debug else select_half_life(matrix(train_features, columns), model_train["did_not_finish"].to_numpy(dtype=int), model_train["date"].reset_index(drop=True), origin, cache_root, False)
    phase = time.time()
    predictions = fit_all_bases(
        matrix(train_features, columns),
        model_train["did_not_finish"].to_numpy(dtype=int),
        model_train["date"].reset_index(drop=True),
        matrix(event_features, columns),
        event["outcome_class"].to_numpy(dtype=int),
        event["date"].reset_index(drop=True),
        matrix(test_features, columns),
        test_features,
        origin,
        half_life,
        debug,
        not debug,
    )
    current = base_frame(predictions, test_features)
    print(f"[candidate] bases stage_b={not debug} half_life={half_life} elapsed={time.time() - phase:.2f}s")
    if debug:
        final = fixed_blend(current)
        adaptation = {"enabled": False, "reason": "debug_fixed_blend"}
        prequential_diagnostics = {"origins": 0, "usable_origins": 0, "created": 0}
    else:
        phase = time.time()
        current_drivers = test["driverId"].to_numpy(dtype=int)
        if read_prequential_cache(cache_root, origin, current_drivers) is None:
            write_prequential_cache(cache_root, origin, current_drivers, current, model_train["date"].max(), event["date"].max())
        prequential, prequential_diagnostics = generate_prequential(model_train, train_features, event, event_features, columns, origin, cache_root)
        final, adaptation = adaptive_blend(prequential, current)
        print(f"[candidate] prequential {prequential_diagnostics} adaptive={adaptation.get('enabled', False)} elapsed={time.time() - phase:.2f}s")
    diagnostics = {"rolling": True, "debug": debug, "features": len(columns), "half_life": half_life, "recency": recency_diagnostics, "prequential": prequential_diagnostics, "adaptation": adaptation}
    return final, diagnostics


def fit_static_chain(train: pd.DataFrame, train_features: pd.DataFrame, prediction_features: pd.DataFrame, event: pd.DataFrame, event_features: pd.DataFrame, event_cutoff: pd.Timestamp, origin: pd.Timestamp, cache_root: Path, debug: bool) -> tuple[np.ndarray, dict, list[str]]:
    columns = model_feature_columns(train_features)
    event_mask = (event["date"] <= event_cutoff).to_numpy()
    half_life, recency = (8.0, {"selected": "8", "mode": "debug"}) if debug else select_half_life(matrix(train_features, columns), train["did_not_finish"].to_numpy(dtype=int), train["date"].reset_index(drop=True), origin, cache_root, False)
    predictions = fit_all_bases(
        matrix(train_features, columns),
        train["did_not_finish"].to_numpy(dtype=int),
        train["date"].reset_index(drop=True),
        matrix(event_features.loc[event_mask], columns),
        event.loc[event_mask, "outcome_class"].to_numpy(dtype=int),
        event.loc[event_mask, "date"].reset_index(drop=True),
        matrix(prediction_features, columns),
        prediction_features,
        origin,
        half_life,
        debug,
        not debug,
    )
    return fixed_blend(base_frame(predictions, prediction_features)), {"half_life": half_life, "recency": recency}, columns


def run_static(builder: FeatureBuilder, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cache_root: Path, debug: bool) -> tuple[np.ndarray, np.ndarray, dict]:
    phase = time.time()
    train_features = build_features(builder, train[["date", "driverId"]], cache_root, "task", False)
    val_features = build_features(builder, val[["date", "driverId"]], cache_root, "task", False)
    test_features = build_features(builder, test[["date", "driverId"]], cache_root, "task", False)
    event = builder.results[["date", "driverId", "outcome_class"]].copy().reset_index(drop=True)
    event_features = build_features(builder, event[["date", "driverId"]], cache_root, "event", False)
    first_val = pd.Timestamp(val["date"].min())
    val_prediction, val_diagnostics, columns = fit_static_chain(train, train_features, val_features, event, event_features, first_val, first_val, cache_root, debug)
    combined = pd.concat([train, val], ignore_index=True)
    combined_features = pd.concat([train_features, val_features], ignore_index=True)
    first_test = pd.Timestamp(test["date"].min())
    test_prediction, test_diagnostics, _ = fit_static_chain(combined, combined_features, test_features, event, event_features, pd.Timestamp(builder.db_max_date), first_test, cache_root, debug)
    diagnostics = {"rolling": False, "debug": debug, "features": len(columns), "val_chain": val_diagnostics, "test_chain": test_diagnostics, "elapsed": time.time() - phase}
    return val_prediction, test_prediction, diagnostics


def main() -> None:
    started = time.time()
    debug = is_debug()
    cache_root = shared_cache()
    register_cache(cache_root)
    task, db, train, val, test = load_context()
    rolling = len(val) == 0
    print(f"[candidate] mode={'rolling' if rolling else 'static'} debug={debug} train={len(train)} val={len(val)} test={len(test)}")
    phase = time.time()
    builder = FeatureBuilder(db)
    print(f"[candidate] database prepared elapsed={time.time() - phase:.2f}s max_date={builder.db_max_date}")
    output = output_directory()
    if rolling:
        prediction, diagnostics = run_rolling(builder, train, test, cache_root, debug)
        np.save(output / "test_predictions.npy", np.asarray(prediction, dtype=float))
    else:
        val_prediction, test_prediction, diagnostics = run_static(builder, train, val, test, cache_root, debug)
        np.save(output / "val_predictions.npy", np.asarray(val_prediction, dtype=float))
        np.save(output / "test_predictions.npy", np.asarray(test_prediction, dtype=float))
    diagnostics["total_elapsed"] = time.time() - started
    (output / "metrics.json").write_text(json.dumps(diagnostics, default=str))
    print(f"[candidate] saved test={len(test)} elapsed={time.time() - started:.2f}s")


if __name__ == "__main__":
    main()
