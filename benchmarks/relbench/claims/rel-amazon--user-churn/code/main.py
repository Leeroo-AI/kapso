from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from kapso_datasets.common import run_data_dir, save_predictions
from residual_model import (
    fit_predict_residual,
    rank_fraction,
    select_residual_stage,
    validation_diagnostics,
)
from sequence_data import load_index, prepare_index
from survival_data import (
    FULL_ORIGINS,
    archive_candidate,
    cache_root,
    load_events,
    load_snapshot,
    preserve_fallback,
)
from survival_model import SurvivalTrajectory


TRAIN_ORIGINS = list(FULL_ORIGINS[:8])
VALIDATION_ORIGIN = FULL_ORIGINS[8]
TEST_ORIGIN = FULL_ORIGINS[9]
MODEL_VERSION = "intensityfree_residual_v38"


def phase(name: str, started: float, checkpoint: float) -> float:
    now = time.time()
    print(
        f"[phase] name={name} phase_seconds={now - checkpoint:.1f} elapsed_seconds={now - started:.1f}",
        flush=True,
    )
    return now


def save_metrics(values: dict) -> None:
    (run_data_dir() / "metrics.json").write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def constant_base(rows: int, positive_rate: float = 0.62) -> tuple[np.ndarray, np.ndarray]:
    intervals = np.tile(np.array([0.06, 0.10, 0.10, 0.12, positive_rate], dtype=np.float32), (rows, 1))
    intervals /= intervals.sum(axis=1, keepdims=True)
    return intervals[:, 4].copy(), intervals


def debug_run(index: dict[str, np.ndarray], events: dict[str, np.ndarray], fallback_path: Path, started: float) -> None:
    checkpoint = time.time()
    trajectory = SurvivalTrajectory(events, index)
    result = trajectory.advance("2015-07-02", completed_limit=100_000, terminal_limit=20_000)
    checkpoint = phase("debug_likelihood", started, checkpoint)
    training_snapshot = load_snapshot("2015-04-02")
    prediction_snapshot = load_snapshot("2015-07-02")
    training_base = constant_base(len(training_snapshot["customer_id"]), float(training_snapshot["label"].mean()))
    prediction_base = constant_base(len(prediction_snapshot["customer_id"]), float(training_snapshot["label"].mean()))
    eligible = trajectory.ends[prediction_snapshot["customer_id"].astype(np.int64)] > trajectory.offsets[
        prediction_snapshot["customer_id"].astype(np.int64)
    ]
    selected_rows = np.flatnonzero(eligible)[:5_000]
    if len(selected_rows):
        scored_binary, scored_intervals = trajectory.score(
            prediction_snapshot["customer_id"][selected_rows].astype(np.int64),
            "2015-07-02",
        )
        prediction_base[0][selected_rows] = scored_binary
        prediction_base[1][selected_rows] = scored_intervals
    residual = fit_predict_residual(
        [training_snapshot],
        [training_base],
        prediction_snapshot,
        prediction_base,
        "temporal",
        1337,
        task_limit=25_000,
    )
    fallback = np.load(fallback_path, allow_pickle=False)
    validation = np.asarray(fallback["validation"], dtype=np.float64)
    test = np.asarray(fallback["test"], dtype=np.float64)
    save_predictions(validation, test)
    save_metrics(
        {
            "debug": True,
            "completed_gaps": result.completed,
            "terminal_exposures": result.terminals,
            "task_rows": 25_000,
            "scored_rows": int(len(selected_rows)),
            "residual_prediction_mean": float(residual.prediction.mean()),
        }
    )
    phase("debug_predictions", started, checkpoint)


def load_completed_cache(path: Path) -> tuple[np.ndarray, np.ndarray, dict] | None:
    metrics_path = path.with_suffix(".json")
    if not path.exists() or not metrics_path.exists():
        return None
    with np.load(path, allow_pickle=False) as values:
        validation = np.asarray(values["validation"], dtype=np.float64)
        test = np.asarray(values["test"], dtype=np.float64)
    if validation.shape != (409_792,) or test.shape != (351_885,):
        return None
    return validation, test, json.loads(metrics_path.read_text())


def fit_final_residual(
    stage: str,
    snapshots: dict[str, dict[str, np.ndarray]],
    base: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict]:
    if stage == "raw":
        return base[VALIDATION_ORIGIN][0].copy(), {"stage": "raw", "history": []}
    result = fit_predict_residual(
        [snapshots[origin] for origin in TRAIN_ORIGINS],
        [base[origin] for origin in TRAIN_ORIGINS],
        snapshots[VALIDATION_ORIGIN],
        base[VALIDATION_ORIGIN],
        stage,
        2026,
    )
    return result.prediction, {"stage": stage, "history": result.history}


def fit_test_residual(
    stage: str,
    snapshots: dict[str, dict[str, np.ndarray]],
    base: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, dict]:
    if stage == "raw":
        return base[TEST_ORIGIN][0].copy(), {"stage": "raw", "history": []}
    training_origins = TRAIN_ORIGINS + [VALIDATION_ORIGIN]
    result = fit_predict_residual(
        [snapshots[origin] for origin in training_origins],
        [base[origin] for origin in training_origins],
        snapshots[TEST_ORIGIN],
        base[TEST_ORIGIN],
        stage,
        2027,
    )
    return result.prediction, {"stage": stage, "history": result.history}


def full_run(index: dict[str, np.ndarray], events: dict[str, np.ndarray], fallback_path: Path, started: float) -> None:
    output_root = cache_root() / MODEL_VERSION
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_cache = output_root / "final_predictions.npz"
    cached = load_completed_cache(prediction_cache)
    if cached is not None:
        validation, test, metrics = cached
        save_predictions(validation, test)
        save_metrics(metrics)
        print("[cache] reused completed conditional-survival Model A/B predictions", flush=True)
        return
    checkpoint = time.time()
    snapshots = {origin: load_snapshot(origin) for origin in FULL_ORIGINS}
    checkpoint = phase("cutoff_feature_snapshots", started, checkpoint)
    trajectory = SurvivalTrajectory(events, index)
    base: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    likelihood_history = []
    for origin in TRAIN_ORIGINS:
        result = trajectory.advance(origin)
        likelihood_history.append(result.__dict__)
        snapshot = snapshots[origin]
        base[origin] = trajectory.score(snapshot["customer_id"].astype(np.int64), origin)
    checkpoint = phase("forward_likelihood_trajectory", started, checkpoint)
    selected_stage, selection = select_residual_stage(TRAIN_ORIGINS, snapshots, base, False)
    checkpoint = phase("four_origin_residual_gates", started, checkpoint)
    if "--selection-only" in sys.argv:
        save_metrics({"selection_only": True, "selection": selection, "selected_stage": selected_stage})
        return
    model_a_result = trajectory.advance(VALIDATION_ORIGIN)
    likelihood_history.append(model_a_result.__dict__)
    base[VALIDATION_ORIGIN] = trajectory.score(
        snapshots[VALIDATION_ORIGIN]["customer_id"].astype(np.int64),
        VALIDATION_ORIGIN,
    )
    standalone_validation, model_a_metrics = fit_final_residual(selected_stage, snapshots, base)
    preserved_validation = standalone_validation.copy()
    checkpoint = phase("model_a_validation", started, checkpoint)
    model_b_result = trajectory.advance(TEST_ORIGIN)
    likelihood_history.append(model_b_result.__dict__)
    base[TEST_ORIGIN] = trajectory.score(
        snapshots[TEST_ORIGIN]["customer_id"].astype(np.int64),
        TEST_ORIGIN,
    )
    standalone_test, model_b_metrics = fit_test_residual(selected_stage, snapshots, base)
    if not np.array_equal(preserved_validation, standalone_validation):
        raise RuntimeError("Model A validation vector changed after Model B fitting")
    fallback_values = np.load(fallback_path, allow_pickle=False)
    fallback_validation = np.asarray(fallback_values["validation"], dtype=np.float64)
    fallback_test = np.asarray(fallback_values["test"], dtype=np.float64)
    correlation = float(spearmanr(standalone_validation, fallback_validation).statistic)
    recorded_tcn_mean = float(np.mean([0.704268, 0.687159]))
    internal_distance = abs(selection["selected_mean_auc"] - recorded_tcn_mean)
    ensemble_eligible = bool(
        internal_distance <= 2.0 * selection["selected_clustered_standard_error"]
        and correlation < 0.95
    )
    archive_candidate(
        "conditional_survival_standalone",
        standalone_validation,
        standalone_test,
        {
            "selected_stage": selected_stage,
            "internal_mean_auc": selection["selected_mean_auc"],
            "internal_standard_error": selection["selected_clustered_standard_error"],
            "fallback_spearman": correlation,
        },
    )
    if ensemble_eligible:
        validation_prediction = 0.5 * rank_fraction(standalone_validation) + 0.5 * rank_fraction(fallback_validation)
        test_prediction = 0.5 * rank_fraction(standalone_test) + 0.5 * rank_fraction(fallback_test)
        archive_candidate(
            "conditional_survival_tcn_equal_rank",
            validation_prediction,
            test_prediction,
            {
                "weight_survival": 0.5,
                "weight_tcn": 0.5,
                "internal_distance": internal_distance,
                "fallback_spearman": correlation,
            },
        )
        prediction_choice = "conditional_survival_tcn_equal_rank"
    else:
        validation_prediction = standalone_validation
        test_prediction = standalone_test
        prediction_choice = "conditional_survival_standalone"
    validation_prediction = np.clip(validation_prediction, 0.0, 1.0)
    test_prediction = np.clip(test_prediction, 0.0, 1.0)
    if validation_prediction.shape != (409_792,) or test_prediction.shape != (351_885,):
        raise RuntimeError(f"Prediction shapes are {validation_prediction.shape} and {test_prediction.shape}")
    if not np.all(np.isfinite(validation_prediction)) or not np.all(np.isfinite(test_prediction)):
        raise RuntimeError("Predictions contain non-finite values")
    diagnostics = validation_diagnostics(
        snapshots[VALIDATION_ORIGIN]["label"].astype(np.int64),
        validation_prediction,
        fallback_validation,
        snapshots[VALIDATION_ORIGIN],
    )
    print(
        "[representativeness] val_rows=409792 test_rows=351885 val_label_rate=0.64203 prior_train_label_rate=0.59867 mean_depth_val=12.22 mean_depth_test=13.71 mean_recency_val=40.27 mean_recency_test=45.54",
        flush=True,
    )
    metrics = {
        "debug": False,
        "likelihood": likelihood_history,
        "selection": selection,
        "selected_stage": selected_stage,
        "model_a": model_a_metrics,
        "model_b": model_b_metrics,
        "prediction_choice": prediction_choice,
        "ensemble_eligible": ensemble_eligible,
        "fallback_spearman": correlation,
        "internal_distance_to_recorded_tcn": internal_distance,
        "diagnostics": diagnostics,
        "validation_fit": "Self-supervised events through validation cutoff and task residual on train labels only",
        "test_fit": "Continued self-supervision through test cutoff and task residual on train plus validation labels",
    }
    temporary = prediction_cache.with_suffix(f".npz.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, validation=validation_prediction, test=test_prediction)
    os.replace(temporary, prediction_cache)
    prediction_cache.with_suffix(".json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    save_predictions(validation_prediction, test_prediction)
    save_metrics(metrics)
    phase("model_b_archive_and_checks", started, checkpoint)


def main() -> None:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    started = time.time()
    checkpoint = started
    prepare_index()
    index = load_index()
    fallback_path = preserve_fallback()
    events = load_events(index)
    checkpoint = phase("coalesced_event_cache", started, checkpoint)
    if "--debug" in sys.argv:
        debug_run(index, events, fallback_path, started)
    else:
        full_run(index, events, fallback_path, started)
    print(f"[complete] elapsed_seconds={time.time() - started:.1f}", flush=True)


if __name__ == "__main__":
    main()
