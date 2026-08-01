from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from feature_factory import T_TIME, V_TIME, VERSION, build_or_load_features, load_events
from kapso_datasets.common import save_predictions
from modeling import Selection, apply_blend, assemble, fit_lgb_final, fit_xgb_final, internal_select, predict_lgb, predict_xgb


def _logger(start: float):
    def log(message: str) -> None:
        print(f"[lane0 elapsed={time.time() - start:.1f}s] {message}", flush=True)
    return log


def _r2(y: np.ndarray, prediction: np.ndarray) -> float:
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - prediction) ** 2) / denominator) if denominator > 0 else 0.0


def _diagnostics(events, blocks, prediction: np.ndarray) -> dict:
    y = events.val_y
    frozen = np.asarray(blocks.frozen_val)
    user_cold = frozen[:, 0] == 0
    beer_cold = frozen[:, 13] == 0
    categories = {
        "warm_warm": ~user_cold & ~beer_cold,
        "user_cold_only": user_cold & ~beer_cold,
        "beer_cold_only": ~user_cold & beer_cold,
        "both_cold": user_cold & beer_cold,
    }
    result = {
        "selected_validation": {
            "count": len(y),
            "r2": _r2(y, prediction),
            "mae": float(np.mean(np.abs(y - prediction))),
            "rmse": float(np.sqrt(np.mean((y - prediction) ** 2))),
        },
        "cold_slices": {},
        "horizon_slices": {},
    }
    for name, mask in categories.items():
        result["cold_slices"][name] = {"count": int(mask.sum()), "r2": _r2(y[mask], prediction[mask]) if mask.sum() > 1 else None}
    val_time = events.time[events.n_train:events.n_tv]
    horizon_month = np.maximum((val_time - V_TIME) // (90 * 86400), 0)
    for bucket in np.unique(horizon_month):
        mask = horizon_month == bucket
        result["horizon_slices"][f"quarter_{int(bucket)}"] = {"count": int(mask.sum()), "r2": _r2(y[mask], prediction[mask]) if mask.sum() > 1 else None}
    return result


def _register_predictions(shared: Path, prediction_dir: Path, key: str) -> None:
    import fcntl
    registry = shared / "artifacts.json"
    lock_path = shared / "artifacts.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            data = json.loads(registry.read_text()) if registry.exists() else []
            name = f"ratebeer causal ensemble predictions lane0 {key}"
            if not any(item.get("name") == name for item in data):
                data.append({"name": name, "path": str(prediction_dir.relative_to(shared)), "description": "Out-of-sample state-A validation and state-B test predictions", "content_key": key, "rebuild_hint": "Run main.py with matching source and feature content"})
                registry.write_text(json.dumps(data, indent=2))
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _cache_predictions(selection: Selection, val_prediction: np.ndarray, test_prediction: np.ndarray, debug: bool) -> None:
    shared = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    payload = json.dumps({"version": VERSION, "debug": debug, "selection": selection.__dict__}, sort_keys=True)
    key = hashlib.sha256(payload.encode()).hexdigest()[:20]
    directory = shared / f"lane0_predictions_{key}"
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / "val_predictions.npy", val_prediction)
    np.save(directory / "test_predictions.npy", test_prediction)
    (directory / "selection.json").write_text(json.dumps(selection.__dict__, indent=2))
    _register_predictions(shared, directory, key)


def main() -> None:
    warnings.filterwarnings("ignore")
    start = time.time()
    log = _logger(start)
    debug = "--debug" in sys.argv
    model_dir = Path("output_data_generic_exp_0") / ("debug" if debug else "full")
    model_dir.mkdir(parents=True, exist_ok=True)
    log(f"start debug={debug} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")
    events = load_events(debug)
    log(f"loaded canonical projection train={events.n_train}/{events.full_train_n} val={events.n_val}/{events.full_val_n} test={events.n_test}/{events.full_test_n}")
    blocks = build_or_load_features(events, debug, log)
    log(f"features base_cols={blocks.base.shape[1]} label_cols={blocks.strict_tv.shape[1]} total_cols={len(blocks.names)}")
    selection_cache = Path(os.environ["KAPSO_SHARED_CACHE_DIR"]) / f"lane0_internal_selection_{VERSION}_{'debug' if debug else 'full'}.json"
    if selection_cache.exists():
        selection = Selection.load(selection_cache)
        log(f"internal selection cache hit path={selection_cache.name} fold_records={selection.fold_records}")
    else:
        selection = internal_select(events, blocks, debug, log)
        selection.save(selection_cache)
    selection.save(model_dir / "internal_selection.json")
    train_index = np.arange(events.n_train, dtype=np.int64)
    val_base_slice = slice(events.n_train, events.n_tv)
    train_x = assemble(blocks.base, np.asarray(blocks.strict_tv)[:events.n_train], train_index)
    val_x = assemble(blocks.base, np.asarray(blocks.frozen_val), val_base_slice)
    log(f"model A matrices train={train_x.shape} validation={val_x.shape}")
    phase = time.time()
    xgb_a = fit_xgb_final(train_x, events.train_y, events.time[:events.n_train], selection.xgb_rounds, selection.recency_weighted, model_dir / "xgb_A.json")
    xgb_val = predict_xgb(xgb_a, val_x)
    log(f"model A xgboost complete rounds={selection.xgb_rounds} seconds={time.time() - phase:.1f}")
    phase = time.time()
    lgb_a = fit_lgb_final(train_x, events.train_y, events.time[:events.n_train], selection.lgb_rounds, selection.recency_weighted, model_dir / "lightgbm_A.txt")
    lgb_val = predict_lgb(lgb_a, val_x)
    val_selected = apply_blend(selection, xgb_val, lgb_val, events.time[events.n_train:events.n_tv], V_TIME)
    np.save(model_dir / "val_predictions_A_checkpoint.npy", val_selected)
    log(f"model A lightgbm and checkpoint complete rounds={selection.lgb_rounds} seconds={time.time() - phase:.1f}")
    del xgb_a, lgb_a, xgb_val, lgb_val, train_x, val_x
    gc.collect()
    tv_y = np.concatenate((events.train_y, events.val_y)).astype(np.float32)
    tv_index = np.arange(events.n_tv, dtype=np.int64)
    test_base_slice = slice(events.n_tv, len(events.time))
    tv_x = assemble(blocks.base, np.asarray(blocks.strict_tv), tv_index)
    test_x = assemble(blocks.base, np.asarray(blocks.frozen_test), test_base_slice)
    log(f"model B matrices train={tv_x.shape} test={test_x.shape}")
    phase = time.time()
    xgb_b = fit_xgb_final(tv_x, tv_y, events.time[:events.n_tv], selection.xgb_rounds, selection.recency_weighted, model_dir / "xgb_B.json")
    xgb_test = predict_xgb(xgb_b, test_x)
    log(f"model B xgboost complete rounds={selection.xgb_rounds} seconds={time.time() - phase:.1f}")
    phase = time.time()
    lgb_b = fit_lgb_final(tv_x, tv_y, events.time[:events.n_tv], selection.lgb_rounds, selection.recency_weighted, model_dir / "lightgbm_B.txt")
    lgb_test = predict_lgb(lgb_b, test_x)
    test_selected = apply_blend(selection, xgb_test, lgb_test, events.time[events.n_tv:], T_TIME)
    log(f"model B lightgbm complete rounds={selection.lgb_rounds} seconds={time.time() - phase:.1f}")
    del xgb_b, lgb_b, xgb_test, lgb_test, tv_x, test_x
    gc.collect()
    center = float(events.train_y.mean())
    val_prediction = np.full(events.full_val_n, center, np.float32)
    test_prediction = np.full(events.full_test_n, center, np.float32)
    val_prediction[events.val_index] = val_selected
    test_prediction[events.test_index] = test_selected
    save_predictions(val_prediction, test_prediction)
    diagnostics = _diagnostics(events, blocks, val_selected)
    output_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "kapso_output"))
    (output_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (model_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    _cache_predictions(selection, val_prediction, test_prediction, debug)
    log(f"diagnostics={json.dumps(diagnostics, separators=(',', ':'))}")
    log(f"complete val_shape={val_prediction.shape} test_shape={test_prediction.shape}")


if __name__ == "__main__":
    main()
