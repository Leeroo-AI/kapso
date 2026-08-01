from __future__ import annotations

import fcntl
import gc
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from features import BEHAVIOR_VERSION, FrozenLabelBuilder, build_behavior_store
from kapso_datasets.common import is_debug, load_task, run_data_dir, save_predictions, shared_cache_dir
from modeling import apply_router, fit_router, predict_experts, safe_auc, slice_diagnostics, train_experts


SOLUTION_VERSION = "generic_exp_1_coverage_router_v1"


def _log(start: float, phase: str) -> None:
    print(f"[pipeline] {phase}: elapsed={time.time() - start:.1f}s", flush=True)


def _label_columns() -> list[str]:
    return [
        "_row_id",
        "_target",
        "SearchID",
        "SearchDate",
        "UserID",
        "IPID",
        "hour_band",
        "CategoryID",
        "LocationID",
        "device_cross",
        "search_parent_category",
        "search_region",
        "category_location_cross",
        "device_category_cross",
        "hour_category_cross",
        "query_present",
    ]


def _debug_sample(indices: np.ndarray, limit: int, seed: int) -> np.ndarray:
    if len(indices) <= limit:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, limit, replace=False))


def _daily_features(
    builder: FrozenLabelBuilder,
    rows: pd.DataFrame,
    indices: np.ndarray,
    name: str,
    debug: bool,
) -> pd.DataFrame:
    path = shared_cache_dir() / f"{SOLUTION_VERSION}_{name}_{'debug' if debug else 'full'}.pkl"
    if path.exists():
        cached = pd.read_pickle(path)
        if len(cached) == len(indices) and np.array_equal(cached.index.to_numpy(), indices):
            print(f"[labels] loaded {name} daily frozen features rows={len(cached)}", flush=True)
            return cached
    features = builder.daily_transform(rows.loc[pd.Index(indices)])
    features = features.loc[pd.Index(indices)]
    features.to_pickle(path)
    print(f"[labels] built {name} daily frozen features rows={len(features)}", flush=True)
    return features


def _origin_predictions(
    rows: pd.DataFrame,
    labels: np.ndarray,
    builder: FrozenLabelBuilder,
    daily_features: pd.DataFrame,
    daily_indices: np.ndarray,
    origins: list[tuple[pd.Timestamp, pd.Timestamp]],
    base_columns: list[str],
    categorical_columns: list[str],
    debug: bool,
) -> dict:
    all_labels = []
    all_repeat = []
    all_cold = []
    all_features = []
    all_origins = []
    all_indices = []
    for number, (origin, end) in enumerate(origins):
        train_indices = daily_indices[rows.loc[pd.Index(daily_indices), "SearchDate"].to_numpy() < np.datetime64(origin)]
        hold_mask = (rows["SearchDate"] >= origin) & (rows["SearchDate"] < end) & rows["_target"].notna()
        hold_indices = np.flatnonzero(hold_mask.to_numpy())
        frozen = builder.transform(rows.loc[pd.Index(hold_indices)], origin)
        experts = train_experts(
            rows,
            daily_features,
            train_indices,
            labels,
            base_columns,
            categorical_columns,
            debug,
            1337 + number * 101,
            origin_model=True,
        )
        repeat, cold = predict_experts(experts, rows, frozen, hold_indices, base_columns)
        all_labels.append(labels[hold_indices])
        all_repeat.append(repeat)
        all_cold.append(cold)
        all_features.append(frozen.reset_index(drop=True))
        all_origins.append(np.full(len(hold_indices), origin.strftime("%Y-%m-%d"), dtype="U10"))
        all_indices.append(hold_indices)
        print(
            f"[origins] origin={origin.date()} train={len(train_indices)} hold={len(hold_indices)} "
            f"covered={(frozen['user_history_count'].to_numpy() > 0).mean():.4f}",
            flush=True,
        )
        del experts, frozen
        gc.collect()
    return {
        "labels": np.concatenate(all_labels),
        "repeat": np.concatenate(all_repeat),
        "cold": np.concatenate(all_cold),
        "features": pd.concat(all_features, ignore_index=True),
        "origins": np.concatenate(all_origins),
        "indices": np.concatenate(all_indices),
    }


def _prediction_bank(debug: bool) -> Path:
    path = shared_cache_dir() / f"{SOLUTION_VERSION}_{'debug' if debug else 'full'}_predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_bank(path: Path, name: str, values: np.ndarray) -> None:
    np.save(path / f"{name}.npy", np.asarray(values, dtype=np.float32))


def _register_artifact(path: Path) -> None:
    root = shared_cache_dir()
    registry = root / "artifacts.json"
    lock_path = root / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if registry.exists():
            try:
                records = json.loads(registry.read_text())
            except json.JSONDecodeError:
                records = []
        else:
            records = []
        relative = str(path.relative_to(root))
        if not any(record.get("path") == relative for record in records):
            records.append(
                {
                    "name": SOLUTION_VERSION,
                    "path": relative,
                    "description": "Auditable forward-origin and final specialist prediction bank",
                    "content_key": f"{SOLUTION_VERSION}:{BEHAVIOR_VERSION}",
                    "rebuild_hint": "Run main.py with the RelBench sanitized cache",
                }
            )
            temporary = root / f"artifacts.{os.getpid()}.tmp"
            temporary.write_text(json.dumps(records, indent=2))
            temporary.replace(registry)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    warnings.filterwarnings("ignore")
    start = time.time()
    debug = is_debug()
    ctx = load_task(upto_test_timestamp=False)
    if ctx.target_col != "IsUserLoggedOn":
        raise RuntimeError(f"Unexpected task target {ctx.target_col}")
    rows, base_columns, categorical_columns = build_behavior_store(ctx)
    labels = rows["_target"].to_numpy(dtype=np.float64)
    train_indices = np.flatnonzero((rows["_split"] == 0).to_numpy())
    val_indices = np.flatnonzero((rows["_split"] == 1).to_numpy())
    test_indices = np.flatnonzero((rows["_split"] == 2).to_numpy())
    combined_indices = np.concatenate([train_indices, val_indices])
    if (len(val_indices), len(test_indices)) != (695590, 592133):
        raise RuntimeError(f"Unexpected split shapes val={len(val_indices)} test={len(test_indices)}")
    _log(start, f"loaded inputs and {len(base_columns)} behavioral features")

    train_source = rows.loc[pd.Index(train_indices), _label_columns()]
    train_builder = FrozenLabelBuilder(train_source)
    chain_a_daily_indices = _debug_sample(train_indices, 50_000, 1337) if debug else train_indices
    chain_a_daily = _daily_features(train_builder, rows, chain_a_daily_indices, "chain_a", debug)
    origin_definitions = (
        [(pd.Timestamp("2015-05-02"), pd.Timestamp("2015-05-08"))]
        if debug
        else [
            (pd.Timestamp("2015-04-30"), pd.Timestamp("2015-05-02")),
            (pd.Timestamp("2015-05-02"), pd.Timestamp("2015-05-08")),
        ]
    )
    forward = _origin_predictions(
        rows,
        labels,
        train_builder,
        chain_a_daily,
        chain_a_daily_indices,
        origin_definitions,
        base_columns,
        categorical_columns,
        debug,
    )
    router_a = fit_router(
        forward["labels"],
        forward["repeat"],
        forward["cold"],
        forward["features"],
        forward["origins"],
    )
    _log(start, "completed forward-origin expert and router fitting")

    val_frozen = train_builder.transform(rows.loc[pd.Index(val_indices)], pd.Timestamp("2015-05-08"))
    experts_a = train_experts(
        rows,
        chain_a_daily,
        chain_a_daily_indices,
        labels,
        base_columns,
        categorical_columns,
        debug,
        2027,
    )
    val_repeat, val_cold = predict_experts(experts_a, rows, val_frozen, val_indices, base_columns)
    val_hard = np.where(val_frozen["user_history_count"].to_numpy() > 0, val_repeat, val_cold)
    val_predictions = apply_router(router_a, val_repeat, val_cold, val_frozen).astype(np.float32)
    val_predictions = np.clip(val_predictions, 1e-6, 1 - 1e-6)
    bank = _prediction_bank(debug)
    _save_bank(bank, "chain_a_val_repeat", val_repeat)
    _save_bank(bank, "chain_a_val_cold", val_cold)
    _save_bank(bank, "chain_a_val_hard", val_hard)
    _save_bank(bank, "chain_a_val_final", val_predictions)
    _save_bank(bank, "origin_repeat", forward["repeat"])
    _save_bank(bank, "origin_cold", forward["cold"])
    _log(start, "banked legal Chain A validation predictions")

    combined_source = rows.loc[pd.Index(combined_indices), _label_columns()]
    combined_builder = FrozenLabelBuilder(combined_source)
    chain_b_daily_indices = _debug_sample(combined_indices, 50_000, 7331) if debug else combined_indices
    chain_b_daily = _daily_features(combined_builder, rows, chain_b_daily_indices, "chain_b", debug)
    router_b_labels = np.concatenate([forward["labels"], labels[val_indices]])
    router_b_repeat = np.concatenate([forward["repeat"], val_repeat])
    router_b_cold = np.concatenate([forward["cold"], val_cold])
    router_b_features = pd.concat([forward["features"], val_frozen.reset_index(drop=True)], ignore_index=True)
    router_b_origins = np.concatenate(
        [forward["origins"], np.full(len(val_indices), "2015-05-08", dtype="U10")]
    )
    router_b = fit_router(
        router_b_labels,
        router_b_repeat,
        router_b_cold,
        router_b_features,
        router_b_origins,
    )
    test_frozen = combined_builder.transform(rows.loc[pd.Index(test_indices)], pd.Timestamp("2015-05-14"))
    experts_b = train_experts(
        rows,
        chain_b_daily,
        chain_b_daily_indices,
        labels,
        base_columns,
        categorical_columns,
        debug,
        3037,
    )
    test_repeat, test_cold = predict_experts(experts_b, rows, test_frozen, test_indices, base_columns)
    test_hard = np.where(test_frozen["user_history_count"].to_numpy() > 0, test_repeat, test_cold)
    test_predictions = apply_router(router_b, test_repeat, test_cold, test_frozen).astype(np.float32)
    test_predictions = np.clip(test_predictions, 1e-6, 1 - 1e-6)
    _save_bank(bank, "chain_b_test_repeat", test_repeat)
    _save_bank(bank, "chain_b_test_cold", test_cold)
    _save_bank(bank, "chain_b_test_hard", test_hard)
    _save_bank(bank, "chain_b_test_final", test_predictions)
    _register_artifact(bank)
    _log(start, "completed and banked legal Chain B test predictions")

    internal_predictions = apply_router(
        router_a,
        forward["repeat"],
        forward["cold"],
        forward["features"],
    )
    diagnostics = {
        "solution_version": SOLUTION_VERSION,
        "debug": debug,
        "elapsed_seconds": time.time() - start,
        "feature_count": len(base_columns),
        "forward_auc": safe_auc(forward["labels"], internal_predictions),
        "forward_slices": slice_diagnostics(forward["labels"], internal_predictions, forward["features"]),
        "chain_a_router": router_a.diagnostics,
        "chain_b_router": router_b.diagnostics,
        "validation_coverage": float((val_frozen["user_history_count"].to_numpy() > 0).mean()),
        "test_coverage": float((test_frozen["user_history_count"].to_numpy() > 0).mean()),
    }
    output = run_data_dir()
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=True))
    save_predictions(val_predictions, test_predictions)
    _log(start, "saved final aligned predictions")


if __name__ == "__main__":
    main()
