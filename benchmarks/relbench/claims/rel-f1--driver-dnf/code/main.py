from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from model_pipeline import (
    CalibrationChoice,
    fit_oof_calibrator,
    fit_sidecars,
    predict_sidecars,
    save_diagnostics,
    select_blend,
    sidecar_oof,
)
from relational_features import FEATURE_NAMES, FEATURE_VERSION, build_all_features, read_snapshot


warnings.filterwarnings("ignore")


@dataclass
class Selection:
    policy: str
    weights: np.ndarray
    sidecar_weights: np.ndarray
    calibration_enabled: bool
    calibration_coefficient: float
    calibration_intercept: float
    sidecar_calibration_enabled: bool
    sidecar_calibration_coefficient: float
    sidecar_calibration_intercept: float
    diagnostics: dict


def elapsed(start: float, phase: str) -> None:
    print(f"[lane1] phase={phase} elapsed={time.monotonic() - start:.2f}s", flush=True)


def temporal_slice_metrics(predictions: np.ndarray, labels: np.ndarray, dates: pd.Series) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    years = pd.to_datetime(dates).dt.year
    for first in range(int(years.min() // 10 * 10), int(years.max() // 10 * 10) + 1, 10):
        mask = (years >= first) & (years < first + 10) & np.isfinite(predictions)
        if mask.sum() and len(np.unique(labels[mask])) > 1:
            output[str(first)] = {"count": int(mask.sum()), "roc_auc": float(roc_auc_score(labels[mask], predictions[mask]))}
    return output


def calibration_values(choice: CalibrationChoice) -> tuple[bool, float, float]:
    if not choice.enabled or choice.model is None:
        return False, 1.0, 0.0
    return True, float(choice.model.coef_[0, 0]), float(choice.model.intercept_[0])


def apply_calibration(values: np.ndarray, enabled: bool, coefficient: float, intercept: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    if not enabled:
        return probabilities
    logits = np.log(probabilities / (1.0 - probabilities))
    transformed = coefficient * logits + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(transformed, -30.0, 30.0)))


def load_selection(path: Path) -> Selection | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Selection(
            policy=str(data["policy"]),
            weights=np.asarray(data["weights"], dtype=np.float64),
            sidecar_weights=np.asarray(data["sidecar_weights"], dtype=np.float64),
            calibration_enabled=bool(data["calibration_enabled"]),
            calibration_coefficient=float(data["calibration_coefficient"]),
            calibration_intercept=float(data["calibration_intercept"]),
            sidecar_calibration_enabled=bool(data["sidecar_calibration_enabled"]),
            sidecar_calibration_coefficient=float(data["sidecar_calibration_coefficient"]),
            sidecar_calibration_intercept=float(data["sidecar_calibration_intercept"]),
            diagnostics=dict(data["diagnostics"]),
        )
    except Exception:
        return None


def save_selection(path: Path, selection: Selection) -> None:
    data = {
        "policy": selection.policy,
        "weights": selection.weights.tolist(),
        "sidecar_weights": selection.sidecar_weights.tolist(),
        "calibration_enabled": selection.calibration_enabled,
        "calibration_coefficient": selection.calibration_coefficient,
        "calibration_intercept": selection.calibration_intercept,
        "sidecar_calibration_enabled": selection.sidecar_calibration_enabled,
        "sidecar_calibration_coefficient": selection.sidecar_calibration_coefficient,
        "sidecar_calibration_intercept": selection.sidecar_calibration_intercept,
        "diagnostics": selection.diagnostics,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.json")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True))
    os.replace(temporary, path)


def tabpfn_oof(
    runner: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
    policy: str,
    maximum: int,
    debug: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(len(labels), np.nan, dtype=np.float64)
    fold_ids = np.full(len(labels), -1, dtype=np.int32)
    for fold_id, (train_idx, valid_idx) in enumerate(folds):
        predictions[valid_idx] = runner.predict(
            matrix[train_idx],
            labels[train_idx],
            dates.iloc[train_idx],
            matrix[valid_idx],
            policy,
            maximum,
            3137 + fold_id,
            debug,
        )
        fold_ids[valid_idx] = fold_id
    return predictions, fold_ids


def build_selection(
    runner: object,
    train_matrix: np.ndarray,
    labels: np.ndarray,
    dates: pd.Series,
    sidecar_predictions: np.ndarray,
    fold_ids: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    sidecar_choice: object,
    sidecar_calibrator: CalibrationChoice,
    maximum: int,
    debug: bool,
) -> Selection:
    policy_candidates: dict[str, tuple[np.ndarray, object]] = {}
    for policy in ("balanced", "last"):
        predictions, tab_fold_ids = tabpfn_oof(
            runner, train_matrix, labels, dates, folds, policy, maximum, debug
        )
        choice = select_blend(predictions.reshape(-1, 1), labels, tab_fold_ids)
        policy_candidates[policy] = (predictions, choice)
    chosen_policy = max(policy_candidates, key=lambda name: policy_candidates[name][1].score)
    tab_predictions, tab_choice = policy_candidates[chosen_policy]
    stacked = np.column_stack([sidecar_predictions, tab_predictions])
    blend_choice = select_blend(stacked, labels, fold_ids)
    blend_oof = stacked @ blend_choice.weights
    calibrator = fit_oof_calibrator(blend_oof, labels, fold_ids)
    calibration_enabled, calibration_coefficient, calibration_intercept = calibration_values(calibrator)
    sidecar_enabled, sidecar_coefficient, sidecar_intercept = calibration_values(sidecar_calibrator)
    diagnostics = {
        "context_policy_scores": {name: value[1].score for name, value in policy_candidates.items()},
        "context_policy_fold_auc": {name: value[1].fold_scores for name, value in policy_candidates.items()},
        "blend_objective": blend_choice.score,
        "blend_fold_auc": blend_choice.fold_scores,
        "sidecar_objective": sidecar_choice.score,
        "sidecar_fold_auc": sidecar_choice.fold_scores,
        "calibration_brier_raw": calibrator.brier_raw,
        "calibration_brier_candidate": calibrator.brier_calibrated,
        "sidecar_calibration_brier_raw": sidecar_calibrator.brier_raw,
        "sidecar_calibration_brier_candidate": sidecar_calibrator.brier_calibrated,
        "oof_slices": temporal_slice_metrics(blend_oof, labels, dates),
    }
    return Selection(
        policy=chosen_policy,
        weights=blend_choice.weights,
        sidecar_weights=sidecar_choice.weights,
        calibration_enabled=calibration_enabled,
        calibration_coefficient=calibration_coefficient,
        calibration_intercept=calibration_intercept,
        sidecar_calibration_enabled=sidecar_enabled,
        sidecar_calibration_coefficient=sidecar_coefficient,
        sidecar_calibration_intercept=sidecar_intercept,
        diagnostics=diagnostics,
    )


def write_metrics(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def main() -> None:
    start = time.monotonic()
    debug = "--debug" in sys.argv
    run_directory = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "output_data_generic_exp_1"))
    run_directory.mkdir(parents=True, exist_ok=True)
    shared = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "output_data_generic_exp_1/shared"))
    tables, train, val, test = read_snapshot()
    rolling = len(val) == 0
    elapsed(start, "load_snapshot")
    train_matrix, val_matrix, test_matrix = build_all_features(tables, [train, val, test], shared)
    if train_matrix.shape[1] != len(FEATURE_NAMES) or len(FEATURE_NAMES) != 60:
        raise RuntimeError(f"feature contract mismatch: {train_matrix.shape[1]} columns, {len(FEATURE_NAMES)} names")
    elapsed(start, "build_60_features")
    labels = train["did_not_finish"].to_numpy(dtype=np.int32)
    target_matrix = test_matrix if rolling else val_matrix
    selection_path = shared / "lane1_tabpfn_v2" / "selection" / f"{FEATURE_VERSION}_{'debug' if debug else 'full'}.json"
    selection = load_selection(selection_path)
    sidecar_predictions: np.ndarray | None = None
    fold_ids: np.ndarray | None = None
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None
    if selection is None:
        sidecar_predictions, fold_ids, folds = sidecar_oof(train_matrix, labels, train["date"], debug, 1337)
        sidecar_choice = select_blend(sidecar_predictions, labels, fold_ids)
        sidecar_oof_blend = sidecar_predictions @ sidecar_choice.weights
        sidecar_calibrator = fit_oof_calibrator(sidecar_oof_blend, labels, fold_ids)
        fallback_weights = sidecar_choice.weights
        fallback_calibration = calibration_values(sidecar_calibrator)
    else:
        fallback_weights = selection.sidecar_weights
        fallback_calibration = (
            selection.sidecar_calibration_enabled,
            selection.sidecar_calibration_coefficient,
            selection.sidecar_calibration_intercept,
        )
    model_a = fit_sidecars(train_matrix, labels, debug, 1737)
    target_sidecars = predict_sidecars(model_a, target_matrix)
    fallback_target = apply_calibration(target_sidecars @ fallback_weights, *fallback_calibration)
    if rolling:
        fallback_test = fallback_target
        fallback_validation = None
        model_b = None
        test_sidecars = None
        combined_matrix = None
        combined_labels = None
        combined_dates = None
    else:
        fallback_validation = fallback_target
        combined_matrix = np.vstack([train_matrix, val_matrix])
        combined_labels = np.concatenate([labels, val["did_not_finish"].to_numpy(dtype=np.int32)])
        combined_dates = pd.concat([train["date"], val["date"]], ignore_index=True)
        model_b = fit_sidecars(combined_matrix, combined_labels, debug, 2137)
        test_sidecars = predict_sidecars(model_b, test_matrix)
        fallback_test = apply_calibration(test_sidecars @ fallback_weights, *fallback_calibration)
    elapsed(start, "sidecar_ready")
    if fallback_validation is not None:
        np.save(run_directory / "val_predictions.npy", np.clip(fallback_validation, 1e-5, 1.0 - 1e-5).astype(np.float64))
    np.save(run_directory / "test_predictions.npy", np.clip(fallback_test, 1e-5, 1.0 - 1e-5).astype(np.float64))
    validation_prediction = fallback_validation
    test_prediction = fallback_test
    target_components = target_sidecars
    tabpfn_status = "fallback_disabled"
    checkpoint_seconds = 0.0
    tabpfn_seconds = 0.0
    debug_probe_origin = bool(
        len(test)
        and pd.Timestamp(test["date"].iloc[0]).month == 3
        and pd.Timestamp(test["date"].iloc[0]).day == 2
    )
    run_tabpfn = os.environ.get("KAPSO_DISABLE_TABPFN") != "1" and not (debug and rolling and not debug_probe_origin)
    if debug and rolling and not debug_probe_origin:
        tabpfn_status = "debug_sidecar_cut"
    if run_tabpfn:
        tab_start = time.monotonic()
        try:
            from tabpfn_support import TabPFNRunner, ensure_tabpfn

            checkpoint, checkpoint_seconds = ensure_tabpfn(shared)
            runner = TabPFNRunner(checkpoint, 2 if debug else 4, 2737)
            maximum = 2000 if debug else 9500
            if selection is None:
                if sidecar_predictions is None or fold_ids is None or folds is None:
                    raise RuntimeError("forward selection state unavailable")
                sidecar_choice = select_blend(sidecar_predictions, labels, fold_ids)
                sidecar_calibrator = fit_oof_calibrator(sidecar_predictions @ sidecar_choice.weights, labels, fold_ids)
                selection = build_selection(
                    runner,
                    train_matrix,
                    labels,
                    train["date"],
                    sidecar_predictions,
                    fold_ids,
                    folds,
                    sidecar_choice,
                    sidecar_calibrator,
                    maximum,
                    debug,
                )
                save_selection(selection_path, selection)
            tab_target = runner.predict(
                train_matrix,
                labels,
                train["date"],
                target_matrix,
                selection.policy,
                maximum,
                4137,
                debug,
            )
            target_components = np.column_stack([target_sidecars, tab_target])
            target_raw = target_components @ selection.weights
            target_final = apply_calibration(
                target_raw,
                selection.calibration_enabled,
                selection.calibration_coefficient,
                selection.calibration_intercept,
            )
            if rolling:
                test_prediction = target_final
            else:
                validation_prediction = target_final
                if combined_matrix is None or combined_labels is None or combined_dates is None or test_sidecars is None:
                    raise RuntimeError("static model B state unavailable")
                tab_test = runner.predict(
                    combined_matrix,
                    combined_labels,
                    combined_dates,
                    test_matrix,
                    selection.policy,
                    maximum,
                    5137,
                    debug,
                )
                test_components = np.column_stack([test_sidecars, tab_test])
                test_raw = test_components @ selection.weights
                test_prediction = apply_calibration(
                    test_raw,
                    selection.calibration_enabled,
                    selection.calibration_coefficient,
                    selection.calibration_intercept,
                )
            tabpfn_seconds = time.monotonic() - tab_start
            tabpfn_status = "pinned_v2"
        except Exception as error:
            tabpfn_seconds = time.monotonic() - tab_start
            tabpfn_status = f"sidecar_fallback:{type(error).__name__}:{str(error)[:160]}"
            print(f"[lane1] TabPFN fallback activated: {type(error).__name__}: {str(error)[:160]}", flush=True)
    validation_prediction = None if validation_prediction is None else np.clip(validation_prediction, 1e-5, 1.0 - 1e-5)
    test_prediction = np.clip(test_prediction, 1e-5, 1.0 - 1e-5)
    elapsed(start, "tabpfn_or_fallback")
    if validation_prediction is not None:
        np.save(run_directory / "val_predictions.npy", validation_prediction.astype(np.float64))
    np.save(run_directory / "test_predictions.npy", test_prediction.astype(np.float64))
    diagnostic_frame = test if rolling else val
    origin = pd.Timestamp(diagnostic_frame["date"].min()).strftime("%Y%m%dT%H%M%S") if len(diagnostic_frame) else "empty"
    diagnostic_path = shared / "lane1_tabpfn_v2" / "diagnostics" / f"{FEATURE_VERSION}_{'debug' if debug else 'full'}_{origin}.npz"
    diagnostic_payload = {
        "dates_ns": pd.to_datetime(diagnostic_frame["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64),
        "driver_ids": diagnostic_frame["driverId"].to_numpy(dtype=np.int64),
        "logistic": target_sidecars[:, 0],
        "lightgbm": target_sidecars[:, 1],
        "blend": test_prediction if rolling else validation_prediction,
    }
    if target_components.shape[1] == 3:
        diagnostic_payload["tabpfn"] = target_components[:, 2]
    save_diagnostics(diagnostic_path, diagnostic_payload)
    metrics = {
        "feature_version": FEATURE_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "rolling": rolling,
        "debug": debug,
        "train_rows": len(train),
        "train_origins": int(train["date"].nunique()),
        "tabpfn_status": tabpfn_status,
        "tabpfn_checkpoint_seconds": checkpoint_seconds,
        "tabpfn_total_seconds": tabpfn_seconds,
        "selection_policy": None if selection is None else selection.policy,
        "blend_weights": None if selection is None else selection.weights.tolist(),
        "selection_diagnostics": {} if selection is None else selection.diagnostics,
        "elapsed_seconds": time.monotonic() - start,
    }
    write_metrics(run_directory, metrics)
    elapsed(start, "write_predictions")
    weights = fallback_weights.tolist() if selection is None else selection.weights.tolist()
    print(
        f"[lane1] wrote {'rolling ' if rolling else ''}predictions test{test_prediction.shape} "
        f"weights={weights} tabpfn={tabpfn_status}",
        flush=True,
    )


if __name__ == "__main__":
    main()
