from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
from numpy.lib.format import open_memmap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from temporal_features import cache_root, register_artifact


# Matrix

def combine_features(compact: np.ndarray, memory: np.ndarray, debug: bool) -> np.memmap:
    path = cache_root() / ("combined_debug_v1.npy" if debug else "combined_full_v1.npy")
    shape = (len(compact), compact.shape[1] + memory.shape[1])
    if path.exists():
        matrix = np.load(path, mmap_mode="r")
        if matrix.shape == shape:
            return matrix
    matrix = open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    block = 100000
    for start in range(0, len(compact), block):
        stop = min(len(compact), start + block)
        matrix[start:stop, : compact.shape[1]] = compact[start:stop]
        matrix[start:stop, compact.shape[1] :] = memory[start:stop]
    matrix.flush()
    register_artifact(
        "lane3 compact plus temporal memory matrix",
        path,
        "Concatenated compact statistics and frozen chronological user memory.",
        "rel-stack-user-badge-lane3-combined-debug-v1" if debug else "rel-stack-user-badge-lane3-combined-full-v1",
    )
    return np.load(path, mmap_mode="r")


# Hazard head

class HazardHead(nn.Module):
    def __init__(self, features: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(features, 192),
            nn.LayerNorm(192),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Linear(96, 13),
        )
        nn.init.constant_(self.network[-1].bias, -6.0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def _hazard_loss(logits: torch.Tensor, bins: torch.Tensor, positive_weight: float) -> torch.Tensor:
    positive = bins < 13
    log_survival = F.logsigmoid(-logits.float())
    log_hazard = F.logsigmoid(logits.float())
    cumulative = torch.cumsum(log_survival, dim=1)
    rows = torch.arange(len(bins), device=bins.device)
    before = torch.where(bins > 0, cumulative[rows, (bins - 1).clamp(0, 12)], torch.zeros(len(bins), device=bins.device))
    event = log_hazard[rows, bins.clamp(0, 12)]
    event_nll = -(before + event)
    censor_nll = -cumulative[:, -1]
    nll = torch.where(positive, event_nll, censor_nll).mean()
    risk = 1.0 - torch.exp(cumulative[:, -1])
    target = positive.float()
    weights = torch.where(positive, torch.full_like(target, positive_weight), torch.ones_like(target))
    any_loss = F.binary_cross_entropy(risk.clamp(1e-6, 1 - 1e-6), target, weight=weights)
    return any_loss + 0.25 * nll


def _blocks(indices: np.ndarray, block_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    order = np.arange(0, len(indices), block_size)
    rng.shuffle(order)
    return [indices[start : min(len(indices), start + block_size)] for start in order]


def fit_hazard(features: np.ndarray, bins: np.ndarray, indices: np.ndarray, tag: str, epochs: int, debug: bool) -> HazardHead:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = HazardHead(features.shape[1]).to(device=device, dtype=dtype)
    checkpoint = cache_root() / f"hazard_{tag}_{'debug' if debug else 'full'}_v3.pt"
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
        for batch_indices in _blocks(indices, batch_size, rng):
            values = torch.as_tensor(np.asarray(features[batch_indices], dtype=np.float32), device=device, dtype=dtype)
            targets = torch.as_tensor(bins[batch_indices], device=device, dtype=torch.int64)
            optimizer.zero_grad(set_to_none=True)
            logits = model(values)
            loss = _hazard_loss(logits, targets, positive_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total += float(loss.detach())
            batches += 1
        rate = len(indices) / max(1e-6, time.time() - started)
        print(f"[hazard] tag={tag} epoch={epoch + 1}/{epochs} loss={total / max(1, batches):.6f} rows_per_s={rate:.0f}", flush=True)
    torch.save(model.state_dict(), checkpoint)
    return model


def predict_hazard(model: HazardHead, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    model.eval()
    output = np.empty(len(indices), dtype=np.float32)
    batch_size = 32768
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            stop = min(len(indices), start + batch_size)
            values = torch.as_tensor(np.asarray(features[indices[start:stop]], dtype=np.float32), device=device, dtype=dtype)
            logits = model(values)
            survival = torch.sigmoid(-logits.float()).prod(1)
            output[start:stop] = (1.0 - survival).cpu().numpy()
    return np.clip(output, 1e-6, 1 - 1e-6)


# Gradient boosting

def fit_gbdt(features: np.ndarray, bins: np.ndarray, indices: np.ndarray, tag: str, debug: bool, rounds: int, random_state: int = 1907) -> lgb.Booster:
    path = cache_root() / f"gbdt_{tag}_{'debug' if debug else 'full'}_v5.txt"
    model = lgb.LGBMClassifier(
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
    if path.exists():
        return lgb.Booster(model_file=str(path))
    values = np.asarray(features[indices], dtype=np.float32)
    target = (bins[indices] < 13).astype(np.uint8)
    model.fit(values, target, callbacks=[lgb.log_evaluation(0)])
    model.booster_.save_model(str(path))
    return model.booster_


def predict_gbdt(model: lgb.Booster, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    output = np.empty(len(indices), dtype=np.float32)
    block = 100000
    for start in range(0, len(indices), block):
        stop = min(len(indices), start + block)
        output[start:stop] = model.predict(np.asarray(features[indices[start:stop]], dtype=np.float32))
    return np.clip(output, 1e-6, 1 - 1e-6)


# Meta model

def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-6, 1 - 1e-6)
    return np.log(values) - np.log1p(-values)


def fit_meta(neural: np.ndarray, gbdt: np.ndarray, target: np.ndarray) -> LogisticRegression:
    values = np.column_stack([logit(neural), logit(gbdt)])
    model = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=300)
    model.fit(values, target)
    return model


def predict_meta(model: LogisticRegression, neural: np.ndarray, gbdt: np.ndarray) -> np.ndarray:
    values = np.column_stack([logit(neural), logit(gbdt)])
    return np.clip(model.predict_proba(values)[:, 1].astype(np.float32), 1e-6, 1 - 1e-6)


def train_oof(features: np.ndarray, bins: np.ndarray, train_times: np.ndarray, debug: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    unique_times = np.unique(train_times)
    fold_times = unique_times[-2:]
    neural_parts = []
    gbdt_parts = []
    target_parts = []
    metrics = {}
    for fold_number, fold_time in enumerate(fold_times):
        train_limit = fold_time - 92 * 86400
        train_indices = np.flatnonzero(train_times < train_limit)
        eligible_origins = np.unique(train_times[train_indices])
        tree_indices = train_indices[np.isin(train_times[train_indices], eligible_origins[-16:])]
        valid_indices = np.flatnonzero(train_times == fold_time)
        if debug:
            train_indices = train_indices[-250000:]
            tree_indices = tree_indices[-250000:]
        hazard = fit_hazard(features, bins, train_indices, f"fold{fold_number}", 1 if debug else 3, debug)
        booster = fit_gbdt(features, bins, tree_indices, f"fold{fold_number}", debug, 80 if debug else 500)
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
