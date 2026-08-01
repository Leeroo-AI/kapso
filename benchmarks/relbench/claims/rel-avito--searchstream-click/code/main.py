from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from feature_factory import (
    FeatureBundle,
    build_base_features,
    build_spine,
    collect_label_payload,
    render_label_features,
)
from kapso_datasets.common import is_debug, run_data_dir, save_predictions, shared_cache_dir
from modeling import blend_prediction, slice_metrics, train_catboost, train_fixed, train_fold, write_diagnostics


FOLDS = [
    (pd.Timestamp("2015-04-30"), pd.Timestamp("2015-05-03")),
    (pd.Timestamp("2015-05-03"), pd.Timestamp("2015-05-06")),
    (pd.Timestamp("2015-05-06"), pd.Timestamp("2015-05-08")),
]


def feature_rows(base: pd.DataFrame, labels: pd.DataFrame, indices: np.ndarray | slice) -> pd.DataFrame:
    left = base.iloc[indices].reset_index(drop=True)
    right = labels.iloc[indices].reset_index(drop=True)
    return pd.concat([left, right], axis=1, copy=False)


def debug_run(bundle: FeatureBundle, base: pd.DataFrame) -> None:
    started = time.time()
    rng = np.random.default_rng(1337)
    train_indices = np.sort(rng.choice(bundle.n_train, size=min(100_000, bundle.n_train), replace=False))
    val_slice = slice(bundle.n_train, bundle.n_train + bundle.n_val)
    model_a = train_fixed(base.iloc[train_indices], bundle.labels_train[train_indices], 50)
    val_prediction = model_a.predict(base.iloc[val_slice])
    combined_labels = np.concatenate([bundle.labels_train, bundle.labels_val])
    combined_size = len(combined_labels)
    combined_indices = np.sort(rng.choice(combined_size, size=min(100_000, combined_size), replace=False))
    model_b = train_fixed(base.iloc[combined_indices], combined_labels[combined_indices], 50)
    test_prediction = model_b.predict(base.iloc[combined_size:])
    save_predictions(
        np.clip(val_prediction, 1e-7, 1 - 1e-7).astype(np.float64),
        np.clip(test_prediction, 1e-7, 1 - 1e-7).astype(np.float64),
    )
    print(f"[candidate] debug two-chain build complete: {time.time() - started:.1f}s", flush=True)


def smoothing_selection(
    bundle: FeatureBundle,
    base_train: pd.DataFrame,
) -> tuple[str, list[dict], list[dict]]:
    modes = ["light", "support", "heavy"]
    scores = {mode: [] for mode in modes}
    fold_records: list[dict] = []
    fold_payloads: list[dict] = []
    train_frame = bundle.spine.iloc[: bundle.n_train].reset_index(drop=True)
    dates = train_frame["SearchDate"]
    hist_logit = base_train["hist_ctr_logit"].to_numpy(dtype=np.float64)
    for fold_index, (score_start, score_end) in enumerate(FOLDS):
        fit_indices = np.flatnonzero((dates < score_start).to_numpy())
        score_indices = np.flatnonzero(((dates >= score_start) & (dates < score_end)).to_numpy())
        fit_frame = train_frame.iloc[fit_indices].reset_index(drop=True)
        score_frame = train_frame.iloc[score_indices].reset_index(drop=True)
        payload = collect_label_payload(
            fit_frame,
            bundle.labels_train[fit_indices],
            score_frame,
            include_causal_fit=False,
        )
        fold_payloads.append(
            {
                "fit_indices": fit_indices,
                "score_indices": score_indices,
                "payload": payload,
                "start": str(score_start.date()),
                "end": str((score_end - pd.Timedelta(days=1)).date()),
            }
        )
        record = {"fold": fold_index + 1, "fit_count": len(fit_indices), "score_count": len(score_indices), "smoothing": {}}
        for mode in modes:
            rendered = render_label_features(payload, mode)
            label_logits = np.column_stack(
                [
                    rendered["label_ad_logit"],
                    rendered["label_user_logit"],
                    rendered["label_ip_logit"],
                    rendered["label_user_ad_logit"],
                    rendered["label_ip_ad_logit"],
                    rendered["label_ad_position_logit"],
                ]
            ).mean(axis=1)
            proxy = 0.45 * hist_logit[score_indices] + 0.55 * label_logits
            score = float(roc_auc_score(bundle.labels_train[score_indices], proxy))
            scores[mode].append(score)
            record["smoothing"][mode] = score
        fold_records.append(record)
        print(f"[candidate] smoothing fold {fold_index + 1}: {json.dumps(record['smoothing'])}", flush=True)
    summaries = {
        mode: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "stable": float(np.mean(values) - 0.25 * np.std(values, ddof=1)),
        }
        for mode, values in scores.items()
    }
    best_mode = max(modes, key=lambda mode: summaries[mode]["stable"])
    best_stable = summaries[best_mode]["stable"]
    if best_mode != "support" and best_stable - summaries["support"]["stable"] < 0.0005:
        best_mode = "support"
    print(f"[candidate] smoothing selection {best_mode}: {json.dumps(summaries)}", flush=True)
    return best_mode, fold_records, fold_payloads


def full_run(
    bundle: FeatureBundle,
    base: pd.DataFrame,
    internal_only: bool = False,
    test_catboost: bool = False,
    position_experts: bool = False,
    seed_ensemble: bool = False,
) -> None:
    overall_start = time.time()
    n_train = bundle.n_train
    n_val = bundle.n_val
    n_fit_a = n_train + n_val
    base_train = base.iloc[:n_train].reset_index(drop=True)
    mode, smoothing_records, fold_payloads = smoothing_selection(bundle, base_train)
    train_frame = bundle.spine.iloc[:n_train].reset_index(drop=True)
    val_frame = bundle.spine.iloc[n_train:n_fit_a].reset_index(drop=True)
    payload_a = collect_label_payload(
        train_frame,
        bundle.labels_train,
        val_frame,
        include_causal_fit=True,
    )
    labels_a = render_label_features(payload_a, mode)
    del payload_a
    gc.collect()
    best_iterations: list[int] = []
    fold_predictions: list[np.ndarray] = []
    fold_hist: list[np.ndarray] = []
    fold_labels: list[np.ndarray] = []
    fold_ad_counts: list[np.ndarray] = []
    fold_slices: list[dict] = []
    fold_auc: list[float] = []
    fold_catboost_predictions: list[np.ndarray] = []
    fold_catboost_auc: list[float] = []
    fold_single_seed_predictions: list[np.ndarray] = []
    position_iterations: dict[str, list[int]] = {"1": [], "7": []}
    if test_catboost and position_experts:
        raise RuntimeError("CatBoost and position-expert experiments must be evaluated separately")
    for fold_index, fold in enumerate(fold_payloads):
        fit_indices = fold["fit_indices"]
        score_indices = fold["score_indices"]
        fold_label_score = render_label_features(fold["payload"], mode)
        train_features = feature_rows(base, labels_a, fit_indices)
        score_base = base.iloc[score_indices].reset_index(drop=True)
        score_features = pd.concat([score_base, fold_label_score], axis=1, copy=False)
        if position_experts:
            prediction = np.zeros(len(score_indices), dtype=np.float64)
            fold_iteration_values: list[int] = []
            fit_positions = bundle.spine.iloc[fit_indices]["Position"].to_numpy(dtype=np.float32)
            score_positions = bundle.spine.iloc[score_indices]["Position"].to_numpy(dtype=np.float32)
            for position in [1, 7]:
                fit_mask = fit_positions == position
                score_mask = score_positions == position
                expert, expert_prediction = train_fold(
                    train_features.loc[fit_mask].reset_index(drop=True),
                    bundle.labels_train[fit_indices][fit_mask],
                    score_features.loc[score_mask].reset_index(drop=True),
                    bundle.labels_train[score_indices][score_mask],
                )
                expert_iteration = int(expert.best_iteration or 1600)
                prediction[score_mask] = expert_prediction
                position_iterations[str(position)].append(expert_iteration)
                fold_iteration_values.append(expert_iteration)
                del expert, expert_prediction
            iteration = int(np.median(fold_iteration_values))
        else:
            seeds = [1337, 2027, 4099] if seed_ensemble else [1337]
            seed_predictions: list[np.ndarray] = []
            seed_iterations: list[int] = []
            for seed in seeds:
                model, seed_prediction = train_fold(
                    train_features,
                    bundle.labels_train[fit_indices],
                    score_features,
                    bundle.labels_train[score_indices],
                    seed=seed,
                )
                seed_predictions.append(seed_prediction)
                seed_iterations.append(int(model.best_iteration or 1600))
                del model
            fold_single_seed_predictions.append(seed_predictions[0])
            prediction = np.mean(np.column_stack(seed_predictions), axis=1)
            iteration = int(np.median(seed_iterations))
        score = float(roc_auc_score(bundle.labels_train[score_indices], prediction))
        best_iterations.append(iteration)
        fold_predictions.append(prediction)
        fold_hist.append(np.clip(bundle.spine.iloc[score_indices]["HistCTR"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7))
        fold_labels.append(bundle.labels_train[score_indices])
        fold_ad_counts.append(fold_label_score["label_ad_count"].to_numpy(dtype=np.float32))
        fold_auc.append(score)
        fold_slices.append(
            slice_metrics(
                bundle.labels_train[score_indices],
                prediction,
                bundle.spine.iloc[score_indices].reset_index(drop=True),
                fold_label_score,
            )
        )
        print(f"[candidate] LightGBM fold {fold_index + 1}: auc={score:.8f} iteration={iteration}", flush=True)
        if test_catboost:
            catboost_model, catboost_prediction = train_catboost(
                train_features,
                bundle.labels_train[fit_indices],
                score_features,
            )
            catboost_score = float(roc_auc_score(bundle.labels_train[score_indices], catboost_prediction))
            fold_catboost_predictions.append(catboost_prediction)
            fold_catboost_auc.append(catboost_score)
            print(f"[candidate] CatBoost fold {fold_index + 1}: auc={catboost_score:.8f}", flush=True)
            del catboost_model
        del train_features, score_features, score_base, fold_label_score
        gc.collect()
    blend_weights = [0.0, 0.10, 0.20, 0.30]
    blend_results: dict[str, dict] = {}
    for weight in blend_weights:
        values = [
            float(roc_auc_score(labels, blend_prediction(prediction, hist, weight)))
            for labels, prediction, hist in zip(fold_labels, fold_predictions, fold_hist)
        ]
        blend_results[str(weight)] = {
            "folds": values,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "stable": float(np.mean(values) - 0.25 * np.std(values, ddof=1)),
        }
    selected_weight = max(blend_weights, key=lambda weight: blend_results[str(weight)]["stable"])
    if selected_weight != 0.0:
        improvement = blend_results[str(selected_weight)]["mean"] - blend_results["0.0"]["mean"]
        standard_error = np.std(
            np.array(blend_results[str(selected_weight)]["folds"]) - np.array(blend_results["0.0"]["folds"]),
            ddof=1,
        ) / np.sqrt(len(FOLDS))
        if improvement <= standard_error:
            selected_weight = 0.0
    gating_results: dict[str, dict] = {}
    selected_gate_threshold = -1
    selected_gate_weight = 0.0
    for threshold in [0, 10, 50]:
        for weight in [0.05, 0.10, 0.20, 0.30]:
            values: list[float] = []
            for labels, prediction, hist, ad_count in zip(
                fold_labels,
                fold_predictions,
                fold_hist,
                fold_ad_counts,
            ):
                gated = prediction.copy()
                cold = ad_count <= threshold
                gated[cold] = blend_prediction(prediction[cold], hist[cold], weight)
                values.append(float(roc_auc_score(labels, gated)))
            key = f"count_le_{threshold}_weight_{weight}"
            gating_results[key] = {
                "folds": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "stable": float(np.mean(values) - 0.25 * np.std(values, ddof=1)),
            }
    candidate_gate = max(gating_results, key=lambda key: gating_results[key]["stable"])
    gate_differences = np.array(gating_results[candidate_gate]["folds"]) - np.array(
        blend_results["0.0"]["folds"]
    )
    gate_improvement = float(gate_differences.mean())
    gate_standard_error = float(gate_differences.std(ddof=1) / np.sqrt(len(FOLDS)))
    if gate_improvement > gate_standard_error:
        parts = candidate_gate.split("_")
        selected_gate_threshold = int(parts[2])
        selected_gate_weight = float(parts[4])
    print(
        f"[candidate] warm/cold gate threshold={selected_gate_threshold} weight={selected_gate_weight:.2f} improvement={gate_improvement:.8f} se={gate_standard_error:.8f}",
        flush=True,
    )
    catboost_blends: dict[str, dict] = {}
    selected_catboost_weight = 0.0
    if test_catboost:
        for weight in [0.10, 0.20, 0.30, 0.50]:
            values = [
                float(
                    roc_auc_score(
                        labels,
                        blend_prediction(lightgbm_prediction, catboost_prediction, weight),
                    )
                )
                for labels, lightgbm_prediction, catboost_prediction in zip(
                    fold_labels,
                    fold_predictions,
                    fold_catboost_predictions,
                )
            ]
            catboost_blends[str(weight)] = {
                "folds": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "stable": float(np.mean(values) - 0.25 * np.std(values, ddof=1)),
            }
        candidate_weight = max(
            [0.10, 0.20, 0.30, 0.50],
            key=lambda weight: catboost_blends[str(weight)]["stable"],
        )
        differences = np.array(catboost_blends[str(candidate_weight)]["folds"]) - np.array(
            blend_results["0.0"]["folds"]
        )
        improvement = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / np.sqrt(len(FOLDS)))
        if improvement > standard_error:
            selected_catboost_weight = candidate_weight
        print(
            f"[candidate] CatBoost gate weight={selected_catboost_weight:.2f} improvement={improvement:.8f} se={standard_error:.8f} blends={json.dumps(catboost_blends)}",
            flush=True,
        )
    selected_seed_ensemble = False
    seed_ensemble_gate: dict[str, float | list[float]] = {}
    if seed_ensemble:
        single_scores = [
            float(roc_auc_score(labels, prediction))
            for labels, prediction in zip(fold_labels, fold_single_seed_predictions)
        ]
        ensemble_scores = blend_results["0.0"]["folds"]
        differences = np.array(ensemble_scores) - np.array(single_scores)
        improvement = float(differences.mean())
        standard_error = float(differences.std(ddof=1) / np.sqrt(len(FOLDS)))
        selected_seed_ensemble = improvement > standard_error
        seed_ensemble_gate = {
            "single_scores": single_scores,
            "ensemble_scores": ensemble_scores,
            "improvement": improvement,
            "standard_error": standard_error,
        }
        print(
            f"[candidate] seed ensemble selected={selected_seed_ensemble} improvement={improvement:.8f} se={standard_error:.8f}",
            flush=True,
        )
    rounds = int(np.median(best_iterations))
    rounds = min(max(rounds, 100), 1600)
    expert_rounds = {
        position: min(max(int(np.median(iterations)), 100), 1600)
        for position, iterations in position_iterations.items()
        if iterations
    }
    print(
        f"[candidate] selected rounds={rounds} hist_logit_weight={selected_weight:.2f} blends={json.dumps(blend_results)}",
        flush=True,
    )
    if internal_only:
        diagnostics = {
            "smoothing_mode": mode,
            "smoothing_folds": smoothing_records,
            "lightgbm_fold_auc": fold_auc,
            "lightgbm_fold_mean": float(np.mean(fold_auc)),
            "lightgbm_fold_std": float(np.std(fold_auc, ddof=1)),
            "best_iterations": best_iterations,
            "selected_iterations": rounds,
            "position_expert_iterations": position_iterations,
            "selected_position_expert_iterations": expert_rounds,
            "blend_results": blend_results,
            "selected_hist_logit_weight": selected_weight,
            "warm_cold_gating": gating_results,
            "selected_gate_threshold": selected_gate_threshold,
            "selected_gate_weight": selected_gate_weight,
            "catboost_fold_auc": fold_catboost_auc,
            "catboost_blends": catboost_blends,
            "selected_catboost_weight": selected_catboost_weight,
            "seed_ensemble_gate": seed_ensemble_gate,
            "selected_seed_ensemble": selected_seed_ensemble,
            "fold_slices": fold_slices,
            "feature_count": int(base.shape[1] + labels_a.shape[1]),
            "elapsed_seconds": time.time() - overall_start,
        }
        fallback_dir = Path("output_data_generic_exp_0")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        write_diagnostics(fallback_dir / "internal_widening_diagnostics.json", diagnostics)
        print(f"[candidate] internal-only forward evaluation complete: {time.time() - overall_start:.1f}s", flush=True)
        return
    all_train_features = feature_rows(base, labels_a, np.arange(n_train, dtype=np.int64))
    val_features = feature_rows(base, labels_a, np.arange(n_train, n_fit_a, dtype=np.int64))
    if position_experts:
        val_prediction = np.zeros(bundle.n_val, dtype=np.float64)
        train_positions = bundle.spine.iloc[:n_train]["Position"].to_numpy(dtype=np.float32)
        val_positions = bundle.spine.iloc[n_train:n_fit_a]["Position"].to_numpy(dtype=np.float32)
        for position in [1, 7]:
            train_mask = train_positions == position
            val_mask = val_positions == position
            model_a = train_fixed(
                all_train_features.loc[train_mask].reset_index(drop=True),
                bundle.labels_train[train_mask],
                expert_rounds[str(position)],
            )
            val_prediction[val_mask] = model_a.predict(val_features.loc[val_mask].reset_index(drop=True))
            del model_a
    else:
        final_seeds = [1337, 2027, 4099] if selected_seed_ensemble else [1337]
        val_parts: list[np.ndarray] = []
        for seed in final_seeds:
            model_a = train_fixed(all_train_features, bundle.labels_train, rounds, seed=seed)
            val_parts.append(model_a.predict(val_features))
            del model_a
        val_prediction = np.mean(np.column_stack(val_parts), axis=1)
    val_hist = np.clip(bundle.spine.iloc[n_train:n_fit_a]["HistCTR"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    val_prediction = np.clip(blend_prediction(val_prediction, val_hist, selected_weight), 1e-7, 1 - 1e-7)
    if selected_gate_weight > 0:
        val_ad_count = labels_a.iloc[n_train:n_fit_a]["label_ad_count"].to_numpy(dtype=np.float32)
        cold = val_ad_count <= selected_gate_threshold
        val_prediction[cold] = blend_prediction(
            val_prediction[cold],
            val_hist[cold],
            selected_gate_weight,
        )
    if selected_catboost_weight > 0:
        catboost_a, catboost_val = train_catboost(all_train_features, bundle.labels_train, val_features)
        val_prediction = np.clip(
            blend_prediction(val_prediction, catboost_val, selected_catboost_weight),
            1e-7,
            1 - 1e-7,
        )
        del catboost_a, catboost_val
    fallback_dir = Path("output_data_generic_exp_0")
    fallback_dir.mkdir(parents=True, exist_ok=True)
    np.save(fallback_dir / "cached_val_predictions.npy", val_prediction.astype(np.float64))
    cache_dir = shared_cache_dir() / "generic_exp_0_lane0_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "model_a_val_predictions.npy", val_prediction.astype(np.float64))
    print(f"[candidate] cached out-of-sample Model A validation predictions at {time.time() - overall_start:.1f}s", flush=True)
    del all_train_features, val_features, labels_a
    gc.collect()
    combined_frame = bundle.spine.iloc[:n_fit_a].reset_index(drop=True)
    test_frame = bundle.spine.iloc[n_fit_a:].reset_index(drop=True)
    combined_labels = np.concatenate([bundle.labels_train, bundle.labels_val])
    payload_b = collect_label_payload(
        combined_frame,
        combined_labels,
        test_frame,
        include_causal_fit=True,
    )
    labels_b = render_label_features(payload_b, mode)
    del payload_b
    gc.collect()
    model_b_features = feature_rows(base, labels_b, np.arange(n_fit_a, dtype=np.int64))
    test_features = feature_rows(base, labels_b, np.arange(n_fit_a, n_fit_a + bundle.n_test, dtype=np.int64))
    if position_experts:
        test_prediction = np.zeros(bundle.n_test, dtype=np.float64)
        combined_positions = bundle.spine.iloc[:n_fit_a]["Position"].to_numpy(dtype=np.float32)
        test_positions = bundle.spine.iloc[n_fit_a:]["Position"].to_numpy(dtype=np.float32)
        for position in [1, 7]:
            train_mask = combined_positions == position
            test_mask = test_positions == position
            model_b = train_fixed(
                model_b_features.loc[train_mask].reset_index(drop=True),
                combined_labels[train_mask],
                expert_rounds[str(position)],
            )
            test_prediction[test_mask] = model_b.predict(test_features.loc[test_mask].reset_index(drop=True))
            del model_b
    else:
        final_seeds = [1337, 2027, 4099] if selected_seed_ensemble else [1337]
        test_parts: list[np.ndarray] = []
        for seed in final_seeds:
            model_b = train_fixed(model_b_features, combined_labels, rounds, seed=seed)
            test_parts.append(model_b.predict(test_features))
            del model_b
        test_prediction = np.mean(np.column_stack(test_parts), axis=1)
    test_hist = np.clip(bundle.spine.iloc[n_fit_a:]["HistCTR"].to_numpy(dtype=np.float64), 1e-7, 1 - 1e-7)
    test_prediction = np.clip(blend_prediction(test_prediction, test_hist, selected_weight), 1e-7, 1 - 1e-7)
    if selected_gate_weight > 0:
        test_ad_count = labels_b.iloc[n_fit_a:]["label_ad_count"].to_numpy(dtype=np.float32)
        cold = test_ad_count <= selected_gate_threshold
        test_prediction[cold] = blend_prediction(
            test_prediction[cold],
            test_hist[cold],
            selected_gate_weight,
        )
    if selected_catboost_weight > 0:
        catboost_b, catboost_test = train_catboost(model_b_features, combined_labels, test_features)
        test_prediction = np.clip(
            blend_prediction(test_prediction, catboost_test, selected_catboost_weight),
            1e-7,
            1 - 1e-7,
        )
        del catboost_b, catboost_test
    if val_prediction.shape != (bundle.n_val,) or test_prediction.shape != (bundle.n_test,):
        raise RuntimeError("prediction shape assertion failed")
    if not np.isfinite(val_prediction).all() or not np.isfinite(test_prediction).all():
        raise RuntimeError("non-finite prediction assertion failed")
    save_predictions(val_prediction.astype(np.float64), test_prediction.astype(np.float64))
    np.save(cache_dir / "model_b_test_predictions.npy", test_prediction.astype(np.float64))
    diagnostics = {
        "label_source_model_a": "train only",
        "label_source_model_b": "train plus validation",
        "smoothing_mode": mode,
        "smoothing_folds": smoothing_records,
        "lightgbm_fold_auc": fold_auc,
        "lightgbm_fold_mean": float(np.mean(fold_auc)),
        "lightgbm_fold_std": float(np.std(fold_auc, ddof=1)),
        "best_iterations": best_iterations,
        "selected_iterations": rounds,
        "position_expert_iterations": position_iterations,
        "selected_position_expert_iterations": expert_rounds,
        "blend_results": blend_results,
        "selected_hist_logit_weight": selected_weight,
        "warm_cold_gating": gating_results,
        "selected_gate_threshold": selected_gate_threshold,
        "selected_gate_weight": selected_gate_weight,
        "catboost_fold_auc": fold_catboost_auc,
        "catboost_blends": catboost_blends,
        "selected_catboost_weight": selected_catboost_weight,
        "seed_ensemble_gate": seed_ensemble_gate,
        "selected_seed_ensemble": selected_seed_ensemble,
        "fold_slices": fold_slices,
        "feature_count": int(base.shape[1] + labels_b.shape[1]),
        "elapsed_seconds": time.time() - overall_start,
    }
    write_diagnostics(run_data_dir() / "metrics.json", diagnostics)
    write_diagnostics(fallback_dir / "train_only_diagnostics.json", diagnostics)
    print(f"[candidate] full two-chain build complete: {time.time() - overall_start:.1f}s", flush=True)


def main() -> None:
    started = time.time()
    debug = is_debug()
    bundle = build_spine()
    base = build_base_features(
        bundle,
        debug,
        context_extension="--test-context-history" in sys.argv,
        query_extension="--test-query-features" in sys.argv,
    )
    print(f"[candidate] base matrix rows={len(base)} columns={base.shape[1]} elapsed={time.time() - started:.1f}s", flush=True)
    if debug:
        debug_run(bundle, base)
    else:
        full_run(
            bundle,
            base,
            internal_only="--internal-only" in sys.argv,
            test_catboost="--test-catboost" in sys.argv,
            position_experts="--test-position-experts" in sys.argv,
            seed_ensemble=True,
        )


if __name__ == "__main__":
    main()
