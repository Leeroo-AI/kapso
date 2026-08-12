from __future__ import annotations

import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import modeling as champion_modeling
from temporal_features import cache_root, register_artifact

DAY = 86400


# Hazard

HazardHead = champion_modeling.HazardHead
predict_hazard = champion_modeling.predict_hazard
fit_meta = champion_modeling.fit_meta
predict_meta = champion_modeling.predict_meta
predict_gbdt = champion_modeling.predict_gbdt


def fit_hazard(features: np.ndarray, bins: np.ndarray, indices: np.ndarray, tag: str, epochs: int, debug: bool) -> HazardHead:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = HazardHead(features.shape[1]).to(device=device, dtype=dtype)
    checkpoint = cache_root() / f"hazard_{tag}_{'debug' if debug else 'full'}_topic_v1_{features.shape[1]}.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return model
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    positives = float((bins[indices] < 13).mean())
    positive_weight = min(12.0, max(1.0, (1.0 - positives) / max(positives, 1e-5)))
    rng = np.random.default_rng(1701)
    batch_size = 16384
    for epoch in range(epochs):
        model.train()
        total = 0.0
        batches = 0
        started = time.time()
        for batch_indices in champion_modeling._blocks(indices, batch_size, rng):
            values = torch.as_tensor(np.asarray(features[batch_indices], dtype=np.float32), device=device, dtype=dtype)
            targets = torch.as_tensor(bins[batch_indices], device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = champion_modeling._hazard_loss(logits, targets, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        rate = len(indices) / max(1e-6, time.time() - started)
        print(f"[hazard] tag={tag} epoch={epoch + 1}/{epochs} loss={total / max(1, batches):.6f} rows_per_s={rate:.0f}", flush=True)
    torch.save(model.state_dict(), checkpoint)
    register_artifact(
        f"lane3 topic hazard {tag}",
        checkpoint,
        "Champion hazard head widened with gated topic/content features.",
        f"rel-stack-user-badge-lane3-topic-hazard-{tag}-{features.shape[1]}-v1",
    )
    return model


# Gradient boosting

def _classifier(rounds: int, random_state: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=rounds,
        num_leaves=63,
        min_child_samples=1500,
        learning_rate=0.04,
        max_bin=127,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.15,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "11")),
        verbosity=-1,
        random_state=random_state,
    )


def fit_gbdt(features: np.ndarray, bins: np.ndarray, indices: np.ndarray, tag: str, debug: bool, rounds: int, random_state: int = 1907) -> lgb.Booster:
    path = cache_root() / f"gbdt_{tag}_{'debug' if debug else 'full'}_topic_v1_{features.shape[1]}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    model = _classifier(rounds, random_state)
    model.fit(np.asarray(features[indices], dtype=np.float32), (bins[indices] < 13).astype(np.uint8), callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(str(path))
    register_artifact(
        f"lane3 topic LightGBM {tag}",
        path,
        "Champion LightGBM settings on the accepted topic/content matrix.",
        f"rel-stack-user-badge-lane3-topic-gbdt-{tag}-{features.shape[1]}-v1",
    )
    return model.booster_


def _fit_gbdt_parts(parts: list[np.ndarray], bins: np.ndarray, indices: np.ndarray, tag: str, debug: bool, rounds: int) -> lgb.Booster:
    columns = sum(part.shape[1] for part in parts)
    path = cache_root() / f"gate_{tag}_{'debug' if debug else 'full'}_v1_{columns}.txt"
    if path.exists():
        return lgb.Booster(model_file=str(path))
    values = np.column_stack([np.asarray(part[indices], dtype=np.float32) for part in parts])
    model = _classifier(rounds, 1907)
    model.fit(values, (bins[indices] < 13).astype(np.uint8), callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(str(path))
    return model.booster_


def _predict_parts(model: lgb.Booster, parts: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), 100000):
        stop = min(len(indices), start + 100000)
        values = np.column_stack([np.asarray(part[indices[start:stop]], dtype=np.float32) for part in parts])
        output[start:stop] = model.predict(values)
    return np.clip(output, 1e-6, 1 - 1e-6)


# Forward folds

def _safe_auc(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> float | None:
    if mask.sum() < 2 or len(np.unique(target[mask])) < 2:
        return None
    return float(roc_auc_score(target[mask], prediction[mask]))


def _slice_masks(compact: np.ndarray, names: list[str], indices: np.ndarray) -> dict[str, np.ndarray]:
    lifetime = np.zeros(len(indices), dtype=np.float32)
    recencies = []
    for group in (0, 1, 3, 5):
        lifetime += np.expm1(np.asarray(compact[indices, names.index(f"lifetime_g{group}")], dtype=np.float32))
        recencies.append(np.expm1(np.asarray(compact[indices, names.index(f"recency_g{group}")], dtype=np.float32)))
    recency = np.min(np.column_stack(recencies), axis=1)
    return {
        "never_active": lifetime == 0,
        "dormant": (lifetime > 0) & (recency > 365),
        "stale": (lifetime > 0) & (recency > 92) & (recency <= 365),
        "active": (lifetime > 0) & (recency <= 92),
    }


def gate_blocks(
    base: np.ndarray,
    tag: np.ndarray,
    content_by_fold: dict[int, np.ndarray],
    bins: np.ndarray,
    train_times: np.ndarray,
    compact: np.ndarray,
    compact_names: list[str],
    debug: bool,
) -> tuple[str, dict]:
    unique_times = np.unique(train_times)
    fold_times = unique_times[-3:]
    blocks = {"tag": lambda fold: [base, tag], "content": lambda fold: [base, content_by_fold[int(fold)]], "both": lambda fold: [base, tag, content_by_fold[int(fold)]]}
    report: dict[str, dict] = {name: {"fold_improvements": [], "dormant_improvements": [], "folds": {}} for name in blocks}
    content_oof_prediction = []
    content_oof_target = []
    content_oof_index = []
    rounds = 80 if debug else 500
    for fold_number, fold_time in enumerate(fold_times):
        train_limit = fold_time - 92 * DAY
        train_indices = np.flatnonzero(train_times < train_limit)
        eligible = np.unique(train_times[train_indices])
        train_indices = train_indices[np.isin(train_times[train_indices], eligible[-16:])]
        valid_indices = np.flatnonzero(train_times == fold_time)
        if debug:
            train_indices = train_indices[-250000:]
        target = (bins[valid_indices] < 13).astype(np.uint8)
        baseline_model = _fit_gbdt_parts([base], bins, train_indices, f"baseline_fold{fold_number}", debug, rounds)
        baseline_prediction = _predict_parts(baseline_model, [base], valid_indices)
        baseline_auc = float(roc_auc_score(target, baseline_prediction))
        masks = _slice_masks(compact, compact_names, valid_indices)
        baseline_slices = {name: _safe_auc(target, baseline_prediction, mask) for name, mask in masks.items()}
        for name, parts_for_fold in blocks.items():
            parts = parts_for_fold(fold_time)
            model = _fit_gbdt_parts(parts, bins, train_indices, f"{name}_fold{fold_number}", debug, rounds)
            prediction = _predict_parts(model, parts, valid_indices)
            auc = float(roc_auc_score(target, prediction))
            slices = {slice_name: _safe_auc(target, prediction, mask) for slice_name, mask in masks.items()}
            improvement = auc - baseline_auc
            dormant_improvement = None
            if slices["dormant"] is not None and baseline_slices["dormant"] is not None:
                dormant_improvement = slices["dormant"] - baseline_slices["dormant"]
            report[name]["fold_improvements"].append(improvement)
            if dormant_improvement is not None:
                report[name]["dormant_improvements"].append(dormant_improvement)
            report[name]["folds"][str(fold_number)] = {
                "origin": str(pd.to_datetime(fold_time, unit="s")),
                "count": int(len(valid_indices)),
                "baseline_auc": baseline_auc,
                "auc": auc,
                "improvement": improvement,
                "baseline_slices": baseline_slices,
                "slices": slices,
            }
            if name == "content":
                content_oof_prediction.append(prediction)
                content_oof_target.append(target)
                content_oof_index.append(valid_indices)
            print(f"[gate] fold={fold_number} block={name} baseline_auc={baseline_auc:.6f} auc={auc:.6f} delta={improvement:+.6f} dormant_delta={dormant_improvement}", flush=True)
    passing = []
    for name, values in report.items():
        folds = np.asarray(values["fold_improvements"], dtype=np.float64)
        dormant = np.asarray(values["dormant_improvements"], dtype=np.float64)
        passed = bool((folds > 0).sum() >= 2 and folds.mean() > 0 and len(dormant) >= 2 and (dormant >= 0).sum() >= 2 and dormant.mean() >= 0)
        values["mean_improvement"] = float(folds.mean())
        values["mean_dormant_improvement"] = float(dormant.mean()) if len(dormant) else None
        values["passed"] = passed
        if passed:
            passing.append(name)
    selected = max(passing, key=lambda name: report[name]["mean_improvement"]) if passing else "none"
    report["selected"] = selected
    np.savez(
        cache_root() / f"solo_content_gate_oof_{'debug' if debug else 'full'}_v1.npz",
        prediction=np.concatenate(content_oof_prediction),
        target=np.concatenate(content_oof_target),
        index=np.concatenate(content_oof_index),
    )
    print(f"[gate] selected={selected} report={json.dumps(report, separators=(',', ':'))}", flush=True)
    return selected, report


def train_oof(features_by_fold: dict[int, np.ndarray], bins: np.ndarray, train_times: np.ndarray, debug: bool, feature_tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    unique_times = np.unique(train_times)
    fold_times = unique_times[-3:]
    neural_parts = []
    gbdt_parts = []
    target_parts = []
    metrics = {}
    for fold_number, fold_time in enumerate(fold_times):
        features = features_by_fold[int(fold_time)]
        train_limit = fold_time - 92 * DAY
        train_indices = np.flatnonzero(train_times < train_limit)
        eligible_origins = np.unique(train_times[train_indices])
        tree_indices = train_indices[np.isin(train_times[train_indices], eligible_origins[-16:])]
        valid_indices = np.flatnonzero(train_times == fold_time)
        if debug:
            train_indices = train_indices[-250000:]
            tree_indices = tree_indices[-250000:]
        hazard = fit_hazard(features, bins, train_indices, f"{feature_tag}_fold{fold_number}", 1 if debug else 3, debug)
        booster = fit_gbdt(features, bins, tree_indices, f"{feature_tag}_fold{fold_number}", debug, 80 if debug else 500)
        neural = predict_hazard(hazard, features, valid_indices)
        tree = predict_gbdt(booster, features, valid_indices)
        target = (bins[valid_indices] < 13).astype(np.uint8)
        neural_parts.append(neural)
        gbdt_parts.append(tree)
        target_parts.append(target)
        metrics[f"fold{fold_number}_neural_auc"] = float(roc_auc_score(target, neural))
        metrics[f"fold{fold_number}_gbdt_auc"] = float(roc_auc_score(target, tree))
        print(f"[oof] fold={fold_number} rows={len(valid_indices)} neural_auc={metrics[f'fold{fold_number}_neural_auc']:.6f} gbdt_auc={metrics[f'fold{fold_number}_gbdt_auc']:.6f}", flush=True)
    return np.concatenate(neural_parts), np.concatenate(gbdt_parts), np.concatenate(target_parts), metrics
