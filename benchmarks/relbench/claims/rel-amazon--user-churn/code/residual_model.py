from __future__ import annotations

import gc
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float64)
    output[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / max(len(values), 1)
    return output


def depth_gate(depth: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(math.log(6.0) - np.log(np.maximum(depth, 1.0))) / 0.35))


def percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    output = np.empty(len(values), dtype=np.float32)
    output[order] = (np.arange(len(values), dtype=np.float32) + 0.5) / max(len(values), 1)
    return output


def compose_features(
    snapshot: dict[str, np.ndarray],
    base_binary: np.ndarray,
    base_intervals: np.ndarray,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    ego = np.asarray(snapshot["ego"], dtype=np.float32)
    customer_id = np.asarray(snapshot["customer_id"], dtype=np.int64)
    customer_bucket = np.zeros((len(customer_id), 32), dtype=np.float32)
    customer_bucket[np.arange(len(customer_id)), np.minimum(customer_id * 32 // 1_850_193, 31)] = 1.0
    base_binary = np.clip(np.asarray(base_binary, dtype=np.float32), 1e-6, 1.0 - 1e-6)
    base_intervals = np.clip(np.asarray(base_intervals, dtype=np.float32), 1e-7, 1.0)
    temporal = np.column_stack(
        (
            base_binary,
            np.log(base_binary) - np.log1p(-base_binary),
            np.log(base_intervals),
            ego,
            np.asarray(snapshot["customer_static"][:, :12], dtype=np.float32),
            customer_id.astype(np.float32) / 1_850_192.0,
            np.log1p(customer_id.astype(np.float32)) / math.log(1_850_193.0),
            customer_bucket,
            np.asarray(snapshot["calendar_context"][:, 1:2], dtype=np.float32),
        )
    ).astype(np.float32)
    if stage == "temporal":
        matrix = temporal
    elif stage == "latent":
        matrix = np.column_stack(
            (
                temporal,
                np.asarray(snapshot["implicit"][:, :32], dtype=np.float32),
                np.asarray(snapshot["content"], dtype=np.float32),
                np.asarray(snapshot["trajectory"], dtype=np.float32),
                np.asarray(snapshot["neighborhood"], dtype=np.float32),
            )
        ).astype(np.float32)
    else:
        raise ValueError(stage)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=12.0, neginf=-12.0)
    gate = depth_gate(ego[:, 5]).astype(np.float32)
    return matrix, gate


class AdditiveResidual(torch.nn.Module):
    def __init__(self, dimensions: int) -> None:
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(dimensions, 256),
            torch.nn.LayerNorm(256),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(256, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.15),
        )
        self.interval = torch.nn.Linear(128, 5)
        self.binary = torch.nn.Linear(128, 1)
        with torch.no_grad():
            self.interval.weight.mul_(0.05)
            self.interval.bias.zero_()
            self.binary.weight.mul_(0.05)
            self.binary.bias.zero_()

    def forward(
        self,
        values: torch.Tensor,
        gate: torch.Tensor,
        base_intervals: torch.Tensor,
        base_binary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.layers(values)
        strength = 8.0 + gate
        interval_logits = torch.log(torch.clamp(base_intervals, min=1e-7)) + strength[:, None] * self.interval(hidden)
        base_logit = torch.logit(torch.clamp(base_binary, min=1e-6, max=1.0 - 1e-6))
        binary_logit = base_logit + strength * self.binary(hidden).squeeze(-1)
        return interval_logits, binary_logit


@dataclass
class ResidualPrediction:
    prediction: np.ndarray
    interval_prediction: np.ndarray
    binary_prediction: np.ndarray
    history: list[dict]


def fit_predict_residual(
    train_snapshots: list[dict[str, np.ndarray]],
    train_base: list[tuple[np.ndarray, np.ndarray]],
    prediction_snapshot: dict[str, np.ndarray],
    prediction_base: tuple[np.ndarray, np.ndarray],
    stage: str,
    seed: int,
    task_limit: int | None = None,
) -> ResidualPrediction:
    train_matrices = []
    train_gates = []
    train_intervals = []
    train_binary = []
    train_base_intervals = []
    train_base_binary = []
    for snapshot, base in zip(train_snapshots, train_base):
        matrix, gate = compose_features(snapshot, base[0], base[1], stage)
        train_matrices.append(matrix)
        train_gates.append(gate)
        train_intervals.append(np.asarray(snapshot["interval"], dtype=np.int64))
        train_binary.append(np.asarray(snapshot["label"], dtype=np.float32))
        train_base_binary.append(np.asarray(base[0], dtype=np.float32))
        train_base_intervals.append(np.asarray(base[1], dtype=np.float32))
    values = np.concatenate(train_matrices)
    gates = np.concatenate(train_gates)
    intervals = np.concatenate(train_intervals)
    binary = np.concatenate(train_binary)
    base_binary = np.concatenate(train_base_binary)
    base_intervals = np.concatenate(train_base_intervals)
    if task_limit is not None and len(values) > task_limit:
        generator = np.random.default_rng(seed)
        rows = np.sort(generator.choice(len(values), size=task_limit, replace=False))
        values = values[rows]
        gates = gates[rows]
        intervals = intervals[rows]
        binary = binary[rows]
        base_binary = base_binary[rows]
        base_intervals = base_intervals[rows]
    prediction_values, prediction_gates = compose_features(
        prediction_snapshot,
        prediction_base[0],
        prediction_base[1],
        stage,
    )
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    values = np.clip((values - mean) / scale, -12.0, 12.0)
    prediction_values = np.clip((prediction_values - mean) / scale, -12.0, 12.0)
    torch.manual_seed(seed)
    model = AdditiveResidual(values.shape[1]).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-4)
    values_tensor = torch.from_numpy(values).cuda()
    gates_tensor = torch.from_numpy(gates).cuda()
    intervals_tensor = torch.from_numpy(intervals).cuda()
    binary_tensor = torch.from_numpy(binary).cuda()
    base_binary_tensor = torch.from_numpy(base_binary).cuda()
    base_intervals_tensor = torch.from_numpy(base_intervals).cuda()
    batch_size = 1024
    history = []
    for epoch in range(2):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed + epoch)
        order = torch.randperm(len(values_tensor), generator=generator, device="cuda")
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                interval_logits, binary_logits = model(
                    values_tensor[rows],
                    gates_tensor[rows],
                    base_intervals_tensor[rows],
                    base_binary_tensor[rows],
                )
                interval_loss = torch.nn.functional.cross_entropy(interval_logits, intervals_tensor[rows])
                binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, binary_tensor[rows])
                loss = interval_loss + 0.3 * binary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
        record = {"epoch": epoch + 1, "loss": total_loss / len(values_tensor), "rows": len(values_tensor)}
        history.append(record)
        print(f"[residual] stage={stage} seed={seed} epoch={epoch + 1} rows={len(values_tensor):,} loss={record['loss']:.6f}", flush=True)
    model.eval()
    output_intervals = np.empty((len(prediction_values), 5), dtype=np.float32)
    output_binary = np.empty(len(prediction_values), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(prediction_values), 65_536):
            stop = min(start + 65_536, len(prediction_values))
            prediction_tensor = torch.from_numpy(prediction_values[start:stop]).cuda()
            gate_tensor = torch.from_numpy(prediction_gates[start:stop]).cuda()
            interval_tensor = torch.from_numpy(np.asarray(prediction_base[1][start:stop], dtype=np.float32)).cuda()
            binary_base_tensor = torch.from_numpy(np.asarray(prediction_base[0][start:stop], dtype=np.float32)).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                interval_logits, binary_logits = model(
                    prediction_tensor,
                    gate_tensor,
                    interval_tensor,
                    binary_base_tensor,
                )
            output_intervals[start:stop] = torch.softmax(interval_logits.float(), dim=1).cpu().numpy()
            output_binary[start:stop] = torch.sigmoid(binary_logits.float()).cpu().numpy()
    prediction = output_intervals[:, 4].copy()
    del model, optimizer, values_tensor, gates_tensor, intervals_tensor, binary_tensor
    del base_binary_tensor, base_intervals_tensor
    gc.collect()
    torch.cuda.empty_cache()
    return ResidualPrediction(prediction, output_intervals, output_binary, history)


def clustered_difference(
    labels: np.ndarray,
    challenger: np.ndarray,
    incumbent: np.ndarray,
    customers: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    unique, groups = np.unique(customers, return_inverse=True)
    generator = np.random.default_rng(seed)
    differences = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        group_weight = generator.poisson(1.0, size=len(unique)).astype(np.float32)
        weights = group_weight[groups]
        if weights[labels == 0].sum() == 0 or weights[labels == 1].sum() == 0:
            differences[draw] = 0.0
        else:
            differences[draw] = roc_auc_score(labels, challenger, sample_weight=weights) - roc_auc_score(
                labels,
                incumbent,
                sample_weight=weights,
            )
    return float(np.mean(differences > 0)), float(np.std(differences, ddof=1))


def select_residual_stage(
    origins: list[str],
    snapshots: dict[str, dict[str, np.ndarray]],
    base: dict[str, tuple[np.ndarray, np.ndarray]],
    debug: bool,
) -> tuple[str, dict]:
    fold_origins = origins[-1:] if debug else origins[-4:]
    predictions: dict[str, list[np.ndarray]] = {"raw": [], "temporal": [], "latent": []}
    labels_by_fold = []
    customers_by_fold = []
    depth_by_fold = []
    histories: dict[str, list] = {"temporal": [], "latent": []}
    for fold_index, origin in enumerate(fold_origins):
        earlier = [value for value in origins if value < origin]
        prediction_snapshot = snapshots[origin]
        predictions["raw"].append(base[origin][0])
        labels_by_fold.append(np.asarray(prediction_snapshot["label"], dtype=np.int64))
        customers_by_fold.append(np.asarray(prediction_snapshot["customer_id"], dtype=np.int64))
        depth_by_fold.append(np.asarray(prediction_snapshot["ego"][:, 5], dtype=np.float32))
        for stage_index, stage in enumerate(("temporal", "latent")):
            result = fit_predict_residual(
                [snapshots[value] for value in earlier],
                [base[value] for value in earlier],
                prediction_snapshot,
                base[origin],
                stage,
                1337 + 100 * fold_index + stage_index,
                25_000 if debug else None,
            )
            predictions[stage].append(result.prediction)
            histories[stage].append(result.history)
            print(
                f"[forward] origin={origin} stage={stage} train_origins={len(earlier)} auc={roc_auc_score(labels_by_fold[-1], result.prediction):.6f}",
                flush=True,
            )
    labels = np.concatenate(labels_by_fold)
    customers = np.concatenate(customers_by_fold)
    depths = np.concatenate(depth_by_fold)
    joined = {stage: np.concatenate(values) for stage, values in predictions.items()}
    fold_scores = {
        stage: [
            float(roc_auc_score(label, prediction))
            for label, prediction in zip(labels_by_fold, predictions[stage])
        ]
        for stage in predictions
    }
    selected = "raw"
    gates = []
    for stage_index, challenger in enumerate(("temporal", "latent")):
        mean_delta = float(np.mean(fold_scores[challenger]) - np.mean(fold_scores[selected]))
        probability, paired_se = clustered_difference(
            labels,
            joined[challenger],
            joined[selected],
            customers,
            10 if debug else 20,
            4000 + stage_index,
        )
        rich = depths >= 6
        rich_delta = float(
            roc_auc_score(labels[rich], joined[challenger][rich])
            - roc_auc_score(labels[rich], joined[selected][rich])
        )
        _, rich_se = clustered_difference(
            labels[rich],
            joined[challenger][rich],
            joined[selected][rich],
            customers[rich],
            10 if debug else 20,
            5000 + stage_index,
        )
        accepted = bool(mean_delta > 0 and probability >= 0.8 and rich_delta >= -rich_se)
        gate = {
            "challenger": challenger,
            "incumbent": selected,
            "mean_delta": mean_delta,
            "probability_positive": probability,
            "paired_standard_error": paired_se,
            "rich_delta": rich_delta,
            "rich_standard_error": rich_se,
            "accepted": accepted,
        }
        gates.append(gate)
        print(f"[residual_gate] {gate}", flush=True)
        if accepted:
            selected = challenger
    diagnostics = {
        "fold_origins": fold_origins,
        "fold_scores": fold_scores,
        "gates": gates,
        "selected": selected,
        "selected_mean_auc": float(np.mean(fold_scores[selected])),
        "histories": histories,
    }
    selected_bootstrap = np.empty(20 if not debug else 10, dtype=np.float64)
    generator = np.random.default_rng(7331)
    unique, groups = np.unique(customers, return_inverse=True)
    for draw in range(len(selected_bootstrap)):
        weights = generator.poisson(1.0, size=len(unique)).astype(np.float32)[groups]
        selected_bootstrap[draw] = roc_auc_score(labels, joined[selected], sample_weight=weights)
    diagnostics["selected_clustered_standard_error"] = float(selected_bootstrap.std(ddof=1))
    print(f"[selection] residual_stage={selected} fold_scores={fold_scores[selected]}", flush=True)
    return selected, diagnostics


def validation_diagnostics(
    labels: np.ndarray,
    prediction: np.ndarray,
    fallback: np.ndarray,
    snapshot: dict[str, np.ndarray],
) -> dict:
    generator = np.random.default_rng(1337)
    bootstrap = np.empty(100, dtype=np.float64)
    for draw in range(100):
        weights = generator.poisson(1.0, size=len(labels)).astype(np.float32)
        bootstrap[draw] = roc_auc_score(labels, prediction, sample_weight=weights)
    correlation = float(spearmanr(prediction, fallback).statistic)
    ego = np.asarray(snapshot["ego"])
    depth = ego[:, 5]
    recency = ego[:, 0]
    multiplicity_proxy = ego[:, 2] - ego[:, 6]
    slices = {
        "depth_1_2": depth <= 2,
        "depth_3_5": (depth >= 3) & (depth <= 5),
        "depth_6_plus": depth >= 6,
        "recency_0_30": recency <= 30,
        "recency_31_60": (recency > 30) & (recency <= 60),
        "recency_61_91": recency > 60,
        "repeat_sparse": multiplicity_proxy <= 0,
        "repeat_present": multiplicity_proxy > 0,
    }
    output_slices = {}
    for name, mask in slices.items():
        output_slices[name] = {
            "count": int(mask.sum()),
            "positive_rate": float(labels[mask].mean()),
            "roc_auc": float(roc_auc_score(labels[mask], prediction[mask])),
        }
        print(f"[validation_slice] name={name} values={output_slices[name]}", flush=True)
    return {
        "roc_auc": float(roc_auc_score(labels, prediction)),
        "bootstrap_standard_error": float(bootstrap.std(ddof=1)),
        "fallback_roc_auc": float(roc_auc_score(labels, fallback)),
        "fallback_spearman": correlation,
        "slices": output_slices,
    }
