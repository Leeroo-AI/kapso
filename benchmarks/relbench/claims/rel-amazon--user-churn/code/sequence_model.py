from __future__ import annotations

import gc
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

from sequence_data import SeedSet, atomic_json, cache_root


MODEL_VERSION = "sequence_survival_gru_v4"


def rank_fraction(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = (np.arange(len(values), dtype=np.float64) + 0.5) / len(values)
    return result


def fixed_sample(seed_set: SeedSet, limit: int, seed: int) -> SeedSet:
    if len(seed_set.customer) <= limit:
        return seed_set
    generator = np.random.default_rng(seed)
    rows = np.sort(generator.choice(len(seed_set.customer), size=limit, replace=False))
    return seed_set.subset(rows, seed_set.name + "_sample")


def concatenate_seed_sets(first: SeedSet, second: SeedSet, name: str) -> SeedSet:
    return SeedSet(
        name,
        np.concatenate((first.customer, second.customer)),
        np.concatenate((first.day, second.day)),
        np.concatenate((first.target, second.target)),
        np.concatenate((first.auxiliary, second.auxiliary)),
        np.concatenate((first.origin, second.origin)),
        np.concatenate((first.last, second.last)),
        np.concatenate((first.depth, second.depth)),
        np.concatenate((first.context, second.context)),
    )


class FeatureStore:
    def __init__(self, index: dict[str, np.ndarray], paths: dict[str, Path], gate: dict, panel: dict[str, np.ndarray]):
        self.customer = np.asarray(index["customer"])
        self.day = np.asarray(index["day"])
        self.product = np.asarray(index["product"])
        self.rating = np.asarray(index["rating"])
        self.verified = np.asarray(index["verified"])
        self.text_missing = np.asarray(index["text_missing"])
        self.gap = np.asarray(index["gap"])
        self.multiplicity = np.asarray(index["multiplicity"])
        self.offsets = np.asarray(index["offsets"])
        self.price = np.asarray(index["product_price"])
        self.category = np.asarray(index["product_category"])
        self.brand = np.asarray(index["product_brand"])
        self.product_text_missing = np.asarray(index["product_text_missing"])
        self.popularity = np.asarray(index["product_popularity"])
        self.doc_hash = np.asarray(index["doc_hash"])
        self.review_embeddings = np.load(paths["review"])
        self.product_embeddings = np.load(paths["product"])
        self.event_doc = np.load(paths["event_doc"])
        valid = panel["attributes"][:, 0] >= 0
        hashes = panel["doc_hash"][valid]
        attributes = panel["attributes"][valid]
        order = np.argsort(hashes, kind="mergesort")
        hashes = hashes[order]
        attributes = attributes[order]
        unique = np.concatenate(([True], hashes[1:] != hashes[:-1])) if len(hashes) else np.empty(0, dtype=np.bool_)
        self.attribute_hashes = hashes[unique]
        self.attributes = attributes[unique].astype(np.float32) / 4.0
        self.keep_attributes = bool(gate.get("kept", False))

    def event_attributes(self, hashes: np.ndarray) -> np.ndarray:
        output = np.zeros(hashes.shape + (5,), dtype=np.float32)
        output[..., 4] = 1.0
        if not self.keep_attributes or not len(self.attribute_hashes):
            return output
        positions = np.searchsorted(self.attribute_hashes, hashes)
        clipped = np.minimum(positions, len(self.attribute_hashes) - 1)
        matches = self.attribute_hashes[clipped] == hashes
        output[matches, :4] = self.attributes[clipped[matches]]
        output[matches, 4] = 0.0
        return output

    def batch(self, seeds: SeedSet, rows: np.ndarray, sequence_length: int) -> dict[str, torch.Tensor]:
        customers = seeds.customer[rows]
        last = seeds.last[rows]
        starts = self.offsets[customers]
        event_positions = last[:, None] - np.arange(sequence_length - 1, -1, -1, dtype=np.int64)[None, :]
        valid = event_positions >= starts[:, None]
        event_positions = np.maximum(event_positions, starts[:, None])
        products = self.product[event_positions]
        ages = seeds.day[rows, None] - self.day[event_positions]
        review = self.review_embeddings[self.event_doc[event_positions]].astype(np.float32)
        product_embedding = self.product_embeddings[products].astype(np.float32)
        popularity = self.popularity[seeds.origin[rows, None], products]
        numeric = np.stack(
            (
                self.rating[event_positions] / 5.0,
                self.verified[event_positions].astype(np.float32),
                self.price[products].astype(np.float32) / 5.0,
                self.text_missing[event_positions].astype(np.float32),
                self.product_text_missing[products].astype(np.float32),
                np.log1p(ages).astype(np.float32) / 6.0,
                np.log1p(self.gap[event_positions].astype(np.float32)) / 7.0,
                np.log1p(self.multiplicity[event_positions].astype(np.float32)) / 3.0,
                np.log1p(popularity.astype(np.float32)) / 12.0,
            ),
            axis=-1,
        )
        attributes = self.event_attributes(self.doc_hash[event_positions])
        pool_length = 128
        pool_positions = last[:, None] - np.arange(pool_length - 1, -1, -1, dtype=np.int64)[None, :]
        pool_valid = pool_positions >= starts[:, None]
        pool_positions = np.maximum(pool_positions, starts[:, None])
        pool_products = self.product[pool_positions]
        pool_ages = seeds.day[rows, None] - self.day[pool_positions]
        pool_review = self.review_embeddings[self.event_doc[pool_positions]].astype(np.float32)
        summary_review = np.empty((len(rows), 5, 64), dtype=np.float32)
        summary_numeric = np.empty((len(rows), 5, 7), dtype=np.float32)
        windows = [30, 91, 182, 365, 100000]
        for window_index, window in enumerate(windows):
            mask = pool_valid & (pool_ages <= window)
            weights = np.exp(-pool_ages.astype(np.float32) / (window if window < 100000 else 365.0)) * mask
            denominator = np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
            summary_review[:, window_index] = (pool_review * weights[:, :, None]).sum(axis=1) / denominator
            counts = mask.sum(axis=1).astype(np.float32)
            if window_index == 4:
                counts = seeds.depth[rows].astype(np.float32)
            masked_products = np.where(mask, pool_products, -1)
            sorted_products = np.sort(masked_products, axis=1)
            distinct = ((sorted_products[:, 1:] != sorted_products[:, :-1]) & (sorted_products[:, 1:] >= 0)).sum(axis=1)
            distinct += (sorted_products[:, 0] >= 0)
            masked_count = np.maximum(mask.sum(axis=1), 1)
            mean_gap = (self.gap[pool_positions].astype(np.float32) * mask).sum(axis=1) / masked_count
            mean_rating = (self.rating[pool_positions] * mask).sum(axis=1) / masked_count
            verified_share = (self.verified[pool_positions].astype(np.float32) * mask).sum(axis=1) / masked_count
            bins = np.where(mask, np.minimum(pool_ages // 30, 31), -1)
            bins.sort(axis=1)
            active_bins = ((bins[:, 1:] != bins[:, :-1]) & (bins[:, 1:] >= 0)).sum(axis=1) + (bins[:, 0] >= 0)
            possible_bins = max(1, min(32, math.ceil(window / 30)))
            persistence = active_bins.astype(np.float32) / possible_bins
            age_span = np.where(mask, pool_ages, -1).max(axis=1).astype(np.float32)
            summary_numeric[:, window_index] = np.column_stack(
                (
                    np.log1p(counts) / 8.0,
                    np.log1p(distinct.astype(np.float32)) / 5.0,
                    np.log1p(mean_gap) / 7.0,
                    mean_rating / 5.0,
                    verified_share,
                    persistence,
                    np.log1p(np.maximum(age_span, 0.0)) / 8.0,
                )
            )
        return {
            "review": torch.from_numpy(review).cuda(non_blocking=True),
            "product": torch.from_numpy(product_embedding).cuda(non_blocking=True),
            "numeric": torch.from_numpy(numeric).cuda(non_blocking=True),
            "category": torch.from_numpy(self.category[products].astype(np.int64)).cuda(non_blocking=True),
            "brand": torch.from_numpy(self.brand[products].astype(np.int64)).cuda(non_blocking=True),
            "attributes": torch.from_numpy(attributes).cuda(non_blocking=True),
            "valid": torch.from_numpy(valid).cuda(non_blocking=True),
            "summary_review": torch.from_numpy(summary_review).cuda(non_blocking=True),
            "summary_numeric": torch.from_numpy(summary_numeric).cuda(non_blocking=True),
            "context": torch.from_numpy(seeds.context[rows].astype(np.float32)).cuda(non_blocking=True),
            "auxiliary": torch.from_numpy(seeds.auxiliary[rows].astype(np.int64)).cuda(non_blocking=True),
            "target": torch.from_numpy(seeds.target[rows].astype(np.float32)).cuda(non_blocking=True),
        }


class SequenceInput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.category = torch.nn.Embedding(1024, 8)
        self.brand = torch.nn.Embedding(65536, 12)
        self.event_projection = torch.nn.Sequential(
            torch.nn.Linear(64 + 32 + 9 + 8 + 12 + 5, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
        )
        self.summary_projection = torch.nn.Sequential(
            torch.nn.Linear(64 + 7, 128),
            torch.nn.LayerNorm(128),
            torch.nn.GELU(),
        )
        self.summary_tokens = torch.nn.Parameter(torch.randn(5, 128) * 0.02)
        self.cutoff_token = torch.nn.Parameter(torch.randn(1, 1, 128) * 0.02)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        event = torch.cat(
            (
                batch["review"],
                batch["product"],
                batch["numeric"],
                self.category(batch["category"]),
                self.brand(batch["brand"]),
                batch["attributes"],
            ),
            dim=-1,
        )
        event = self.event_projection(event)
        event = event * batch["valid"].unsqueeze(-1)
        summary = self.summary_projection(torch.cat((batch["summary_review"], batch["summary_numeric"]), dim=-1))
        summary = summary + self.summary_tokens.unsqueeze(0)
        cutoff = self.cutoff_token.expand(len(event), -1, -1)
        return torch.cat((summary, event, cutoff), dim=1)


class GRUSurvival(torch.nn.Module):
    def __init__(self, context_dimensions: int):
        super().__init__()
        self.input = SequenceInput()
        self.encoder = torch.nn.GRU(
            128,
            128,
            num_layers=2,
            dropout=0.20,
            bidirectional=True,
            batch_first=True,
        )
        self.head = torch.nn.Sequential(
            torch.nn.Linear(256 + context_dimensions, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
        )
        self.interval = torch.nn.Linear(128, 5)
        self.binary = torch.nn.Linear(128, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.input(batch)
        _, state = self.encoder(sequence)
        pooled = torch.cat((state[-2], state[-1]), dim=-1)
        hidden = self.head(torch.cat((pooled, batch["context"]), dim=-1))
        return self.interval(hidden), self.binary(hidden).squeeze(-1)


class CausalBlock(torch.nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        self.dilation = dilation
        self.conv = torch.nn.Conv1d(128, 128, kernel_size=3, padding=2 * dilation, dilation=dilation)
        self.norm = torch.nn.LayerNorm(128)
        self.dropout = torch.nn.Dropout(0.20)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        length = value.shape[1]
        transformed = self.conv(value.transpose(1, 2))[:, :, :length].transpose(1, 2)
        transformed = self.dropout(torch.nn.functional.gelu(self.norm(transformed)))
        return value + transformed


class TCNSurvival(torch.nn.Module):
    def __init__(self, context_dimensions: int):
        super().__init__()
        self.input = SequenceInput()
        self.blocks = torch.nn.ModuleList([CausalBlock(2**index) for index in range(6)])
        self.head = torch.nn.Sequential(
            torch.nn.Linear(128 + context_dimensions, 128),
            torch.nn.GELU(),
            torch.nn.Dropout(0.20),
        )
        self.interval = torch.nn.Linear(128, 5)
        self.binary = torch.nn.Linear(128, 1)

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        sequence = self.input(batch)
        for block in self.blocks:
            sequence = block(sequence)
        hidden = self.head(torch.cat((sequence[:, -1], batch["context"]), dim=-1))
        return self.interval(hidden), self.binary(hidden).squeeze(-1)


def create_model(architecture: str, context_dimensions: int) -> torch.nn.Module:
    torch.manual_seed(1337)
    if architecture == "gru":
        return GRUSurvival(context_dimensions).cuda()
    if architecture == "tcn":
        return TCNSurvival(context_dimensions).cuda()
    raise ValueError(architecture)


def train_epochs(
    model: torch.nn.Module,
    store: FeatureStore,
    seeds: SeedSet,
    sequence_length: int,
    epochs: int,
    label: str,
    batch_size: int = 4096,
) -> list[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    history = []
    for epoch in range(epochs):
        model.train()
        generator = np.random.default_rng(1337 + epoch)
        order = generator.permutation(len(seeds.customer))
        total_loss = 0.0
        total_rows = 0
        started = time.time()
        for start in range(0, len(order), batch_size):
            rows = order[start : start + batch_size]
            batch = store.batch(seeds, rows, sequence_length)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                intervals, binary = model(batch)
                interval_loss = torch.nn.functional.cross_entropy(intervals, batch["auxiliary"])
                binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(binary, batch["target"])
                loss = interval_loss + 0.30 * binary_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(rows)
            total_rows += len(rows)
        elapsed = time.time() - started
        record = {"epoch": epoch + 1, "loss": total_loss / total_rows, "seconds": elapsed, "rows_per_second": total_rows / elapsed}
        history.append(record)
        print(
            f"[train] {label} epoch={epoch + 1} loss={record['loss']:.6f} rows_per_second={record['rows_per_second']:.1f} elapsed={elapsed:.1f}s",
            flush=True,
        )
    return history


def predict_network(
    model: torch.nn.Module,
    store: FeatureStore,
    seeds: SeedSet,
    sequence_length: int,
    batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(seeds.customer), dtype=np.float64)
    with torch.inference_mode():
        for start in range(0, len(predictions), batch_size):
            rows = np.arange(start, min(start + batch_size, len(predictions)))
            batch = store.batch(seeds, rows, sequence_length)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                intervals, _ = model(batch)
            predictions[rows] = torch.softmax(intervals.float(), dim=1)[:, 4].cpu().numpy()
    return predictions


@dataclass
class HazardModel:
    mean: np.ndarray
    scale: np.ndarray
    model: SGDClassifier

    def predict(self, context: np.ndarray) -> np.ndarray:
        return self.model.predict_proba((context - self.mean) / self.scale)[:, 1]


def fit_hazard(seeds: SeedSet) -> HazardModel:
    mean = seeds.context.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = seeds.context.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[scale < 1e-5] = 1.0
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=2e-5,
        max_iter=30,
        tol=1e-4,
        random_state=1337,
        average=True,
        n_jobs=int(os.environ.get("OMP_NUM_THREADS", "1")),
    )
    model.fit((seeds.context - mean) / scale, seeds.target)
    return HazardModel(mean, scale, model)


def fold_sets(train: SeedSet, day: int) -> tuple[SeedSet, SeedSet]:
    earlier = np.flatnonzero(train.day < day)
    fold = np.flatnonzero(train.day == day)
    return train.subset(earlier, f"before_{day}"), train.subset(fold, f"fold_{day}")


def forward_selection(store: FeatureStore, train: SeedSet) -> dict:
    path = cache_root() / MODEL_VERSION / "selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        selection = json.loads(path.read_text())
        print(f"[selection] reused {selection}", flush=True)
        return selection
    fold_days = [int(day) for day in np.unique(train.day)[-2:]]
    candidates = [("gru", 16), ("gru", 32), ("tcn", 32)]
    scores: dict[str, list[float]] = {f"{architecture}_{length}": [] for architecture, length in candidates}
    states: dict[tuple[str, int, int], dict[str, torch.Tensor]] = {}
    sampled_folds = []
    for fold_index, fold_day in enumerate(fold_days):
        preceding, fold = fold_sets(train, fold_day)
        preceding = fixed_sample(preceding, 300_000, 1337 + fold_index)
        fold = fixed_sample(fold, 80_000, 2337 + fold_index)
        sampled_folds.append((preceding, fold))
        for architecture, length in candidates:
            model = create_model(architecture, train.context.shape[1])
            train_epochs(model, store, preceding, length, 1, f"select_{architecture}{length}_fold{fold_index + 1}")
            prediction = predict_network(model, store, fold, length)
            auc = roc_auc_score(fold.target, prediction)
            key = f"{architecture}_{length}"
            scores[key].append(float(auc))
            states[(architecture, length, fold_index)] = {name: value.detach().cpu() for name, value in model.state_dict().items()}
            print(f"[selection] candidate={key} fold={fold_index + 1} auc={auc:.6f}", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    chosen_key = max(means, key=means.get)
    architecture, length_text = chosen_key.split("_")
    sequence_length = int(length_text)
    epoch_scores = {1: scores[chosen_key]}
    chosen_predictions: dict[int, np.ndarray] = {}
    for fold_index, (preceding, fold) in enumerate(sampled_folds):
        model = create_model(architecture, train.context.shape[1])
        model.load_state_dict(states[(architecture, sequence_length, fold_index)])
        chosen_predictions[fold_index] = predict_network(model, store, fold, sequence_length)
        for epoch in [2, 3]:
            train_epochs(model, store, preceding, sequence_length, 1, f"epoch_gate_fold{fold_index + 1}_e{epoch}")
            prediction = predict_network(model, store, fold, sequence_length)
            epoch_scores.setdefault(epoch, []).append(float(roc_auc_score(fold.target, prediction)))
            if epoch == 3:
                chosen_predictions[fold_index] = prediction
        del model
        gc.collect()
        torch.cuda.empty_cache()
    epoch_means = {epoch: float(np.mean(value)) for epoch, value in epoch_scores.items()}
    chosen_epoch = max(epoch_means, key=epoch_means.get)
    blend_improvements = []
    for fold_index, (preceding, fold) in enumerate(sampled_folds):
        model = create_model(architecture, train.context.shape[1])
        model.load_state_dict(states[(architecture, sequence_length, fold_index)])
        if chosen_epoch > 1:
            train_epochs(model, store, preceding, sequence_length, chosen_epoch - 1, f"blend_gate_fold{fold_index + 1}")
        sequence_prediction = predict_network(model, store, fold, sequence_length)
        hazard = fit_hazard(preceding).predict(fold.context)
        blend = 0.75 * rank_fraction(sequence_prediction) + 0.25 * rank_fraction(hazard)
        base_auc = roc_auc_score(fold.target, sequence_prediction)
        blend_auc = roc_auc_score(fold.target, blend)
        blend_improvements.append(float(blend_auc - base_auc))
        print(
            f"[selection] blend fold={fold_index + 1} base_auc={base_auc:.6f} blend_auc={blend_auc:.6f} delta={blend_auc - base_auc:+.6f}",
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    selection = {
        "architecture": architecture,
        "sequence_length": sequence_length,
        "epochs": int(chosen_epoch),
        "candidate_scores": scores,
        "candidate_means": means,
        "epoch_scores": {str(key): value for key, value in epoch_scores.items()},
        "epoch_means": {str(key): value for key, value in epoch_means.items()},
        "blend_improvements": blend_improvements,
        "blend": bool(all(value > 0 for value in blend_improvements)),
        "blend_weight": 0.25,
    }
    atomic_json(path, selection)
    print(f"[selection] chosen={selection}", flush=True)
    return selection


def train_final_model(store: FeatureStore, seeds: SeedSet, selection: dict, label: str) -> torch.nn.Module:
    model = create_model(selection["architecture"], seeds.context.shape[1])
    train_epochs(model, store, seeds, int(selection["sequence_length"]), int(selection["epochs"]), label)
    return model
