# Imports

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from kapso_datasets.common import load_task, shared_cache_dir
from trial_pipeline import _candidate_truth, _pair_rows_for_year, build_bundle, fit_heads


# Constants

MODEL_ID = "answerdotai/ModernBERT-large"
VERSION = "exact_success_modernbert_large_v1"
MAX_LENGTH = 4096
BATCH_SIZE = 8
ACCUMULATION = 4
EPOCHS = 2
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01


# Data

class WindowDataset(Dataset):
    def __init__(self, windows: list[dict[str, Any]]) -> None:
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.windows[index]


class SortishBatchSampler(Sampler[list[int]]):
    def __init__(self, lengths: list[int], batch_size: int, seed: int, shuffle: bool) -> None:
        order = np.argsort(np.asarray(lengths), kind="stable")
        self.batches = [order[start:start + batch_size].tolist() for start in range(0, len(order), batch_size)]
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(batches)
        self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def training_rows(bundle: Any, cutoff: pd.Timestamp) -> pd.DataFrame:
    frame = bundle.replay[
        bundle.replay["report_label"].eq(1)
        & (bundle.replay["timestamp"] + pd.Timedelta(days=365) <= cutoff)
    ].copy()
    age = (cutoff - frame["timestamp"]).dt.days.to_numpy(dtype=np.float64) / 365.25
    weight = np.power(0.5, age / 4.0)
    origin_total = pd.Series(weight).groupby(frame["timestamp"].reset_index(drop=True)).transform("sum").to_numpy()
    frame["model_weight"] = weight / np.maximum(origin_total, 1e-12)
    frame["model_weight"] /= frame["model_weight"].mean()
    return frame.reset_index(drop=True)


def serialized_text(bundle: Any, pair_row: int) -> str:
    return f"Structured trial context:\n{bundle.contexts[pair_row]}\nComplete trial document:\n{bundle.documents[pair_row]}"


def tokenize_rows(tokenizer: Any, texts: list[str], labels: np.ndarray | None, weights: np.ndarray | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    windows: list[dict[str, Any]] = []
    long_documents = 0
    for document_index, text in enumerate(texts):
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LENGTH,
            stride=128,
            return_overflowing_tokens=True,
        )
        count = len(encoded["input_ids"])
        long_documents += int(count > 1)
        for window_index in range(count):
            record: dict[str, Any] = {
                "input_ids": encoded["input_ids"][window_index],
                "attention_mask": encoded["attention_mask"][window_index],
                "document_index": document_index,
            }
            if labels is not None and weights is not None:
                record["label"] = float(labels[document_index])
                record["weight"] = float(weights[document_index]) / count
            windows.append(record)
    return windows, {"documents": len(texts), "windows": len(windows), "long_documents": long_documents}


def collator(tokenizer: Any):
    def collate(records: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer.pad(
            [{"input_ids": record["input_ids"], "attention_mask": record["attention_mask"]} for record in records],
            padding=True,
            pad_to_multiple_of=16,
            return_tensors="pt",
        )
        encoded["document_index"] = torch.tensor([record["document_index"] for record in records], dtype=torch.long)
        if "label" in records[0]:
            encoded["labels"] = torch.tensor([record["label"] for record in records], dtype=torch.float32)
            encoded["weights"] = torch.tensor([record["weight"] for record in records], dtype=torch.float32)
        return encoded
    return collate


def loader(windows: list[dict[str, Any]], tokenizer: Any, shuffle: bool, batch_size: int = BATCH_SIZE) -> DataLoader:
    sampler = SortishBatchSampler([len(record["input_ids"]) for record in windows], batch_size, 9127, shuffle)
    return DataLoader(
        WindowDataset(windows),
        batch_sampler=sampler,
        collate_fn=collator(tokenizer),
        num_workers=0,
        pin_memory=True,
    )


# Model

def cache_path(cache: Path, cutoff: pd.Timestamp) -> Path:
    key = hashlib.sha256(f"{VERSION}|{cutoff.date()}|{MAX_LENGTH}|{EPOCHS}".encode()).hexdigest()[:16]
    return cache / "modernbert_lane0" / key


def train_model(bundle: Any, cutoff: pd.Timestamp, cache: Path) -> tuple[Path, dict[str, Any]]:
    target = cache_path(cache, cutoff)
    manifest_path = target / "training_manifest.json"
    if manifest_path.exists() and (target / "model.safetensors").exists():
        return target, json.loads(manifest_path.read_text())
    target.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=cache / "huggingface")
    rows = training_rows(bundle, cutoff)
    texts = [serialized_text(bundle, int(index)) for index in rows["pair_row"]]
    labels = rows["success_all"].to_numpy(dtype=np.float32)
    weights = rows["model_weight"].to_numpy(dtype=np.float32)
    token_start = time.time()
    windows, token_stats = tokenize_rows(tokenizer, texts, labels, weights)
    train_loader = loader(windows, tokenizer, True)
    torch.manual_seed(9127)
    np.random.seed(9127)
    core_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=1,
        problem_type="regression",
        torch_dtype=torch.bfloat16,
        cache_dir=cache / "huggingface",
    )
    core_model.config.use_cache = False
    core_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    core_model.to("cuda:0")
    model = torch.nn.DataParallel(core_model, device_ids=list(range(min(2, torch.cuda.device_count()))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, foreach=False)
    updates_per_epoch = math.ceil(len(train_loader) / ACCUMULATION)
    total_updates = updates_per_epoch * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(total_updates * 0.06)), total_updates)
    optimizer.zero_grad(set_to_none=True)
    train_start = time.time()
    updates = 0
    losses: list[float] = []
    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
            labels_batch = batch["labels"].to("cuda:0", non_blocking=True)
            weights_batch = batch["weights"].to("cuda:0", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float().reshape(-1)
                item_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels_batch, reduction="none")
                loss = (item_loss * weights_batch).sum() / weights_batch.sum().clamp_min(1e-8)
                scaled_loss = loss / ACCUMULATION
            scaled_loss.backward()
            losses.append(float(loss.detach().cpu()))
            boundary = (step + 1) % ACCUMULATION == 0 or step + 1 == len(train_loader)
            if boundary:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
                if updates % 100 == 0 or updates == total_updates:
                    print(
                        f"[modernbert] cutoff={cutoff.date()} epoch={epoch + 1} update={updates}/{total_updates} loss={np.mean(losses[-100:]):.6f} elapsed={time.time() - train_start:.1f}s",
                        flush=True,
                    )
    core_model.save_pretrained(target, safe_serialization=True)
    tokenizer.save_pretrained(target)
    manifest = {
        "version": VERSION,
        "model": MODEL_ID,
        "cutoff": str(cutoff),
        "training_rows": len(rows),
        "label_mean": float(labels.mean()),
        "tokenization_seconds": time.time() - token_start,
        "training_seconds": time.time() - train_start,
        "updates": updates,
        "effective_batch": BATCH_SIZE * ACCUMULATION,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        **token_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return target, manifest


def predict_model(bundle: Any, pair_rows: np.ndarray, model_path: Path, cache: Path) -> tuple[np.ndarray, dict[str, Any]]:
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    texts = [serialized_text(bundle, int(index)) for index in pair_rows]
    windows, stats = tokenize_rows(tokenizer, texts, None, None)
    predict_loader = loader(windows, tokenizer, False, batch_size=24)
    core_model = AutoModelForSequenceClassification.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    core_model.to("cuda:0").eval()
    model = torch.nn.DataParallel(core_model, device_ids=list(range(min(2, torch.cuda.device_count()))))
    document_logits: list[list[float]] = [[] for _ in pair_rows]
    start = time.time()
    with torch.inference_mode():
        for batch in predict_loader:
            input_ids = batch["input_ids"].to("cuda:0", non_blocking=True)
            attention_mask = batch["attention_mask"].to("cuda:0", non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float().reshape(-1)
            for index, value in zip(batch["document_index"].numpy(), logits.cpu().numpy()):
                document_logits[int(index)].append(float(value))
    pooled = []
    for values in document_logits:
        current = np.asarray(values, dtype=np.float64)
        attention = np.exp(np.abs(current) - np.max(np.abs(current)))
        pooled.append(float(np.sum(attention * current) / np.sum(attention)))
    probabilities = 1.0 / (1.0 + np.exp(-np.asarray(pooled)))
    stats["inference_seconds"] = time.time() - start
    stats["rows_per_second"] = len(pair_rows) / max(stats["inference_seconds"], 1e-9)
    return probabilities, stats


# Gate

def forward_gate(cutoff_year: int) -> dict[str, Any]:
    cache = shared_cache_dir()
    context = load_task()
    bundle, _ = build_bundle(context, cache, False)
    cutoff = pd.Timestamp(f"{cutoff_year}-01-01")
    model_path, training = train_model(bundle, cutoff, cache)
    year = cutoff_year
    candidates = _pair_rows_for_year(bundle, year)
    reporting, success = _candidate_truth(bundle, candidates, cutoff)
    reporting_rows = candidates[reporting.astype(bool)]
    labels = success[reporting.astype(bool)].astype(np.int8)
    predictions, inference = predict_model(bundle, reporting_rows, model_path, cache)
    head = fit_heads(bundle, cutoff, "equal", False)
    tabular = head.success.predict(bundle.features.iloc[reporting_rows])
    text = head.text_model.predict_proba(head.text_vectorizer.transform([bundle.documents[index] for index in reporting_rows]))[:, 1]
    blend_grid = []
    components = {"tabular": tabular, "tfidf": text, "modernbert": predictions}
    for modernbert_weight in np.linspace(0.0, 0.5, 6):
        for tfidf_weight in np.linspace(0.0, 1.0 - modernbert_weight, 6):
            tabular_weight = 1.0 - modernbert_weight - tfidf_weight
            blended = tabular_weight * tabular + tfidf_weight * text + modernbert_weight * predictions
            blend_grid.append({
                "tabular_weight": float(tabular_weight),
                "tfidf_weight": float(tfidf_weight),
                "modernbert_weight": float(modernbert_weight),
                "auc": float(roc_auc_score(labels, blended)),
                "mae": float(mean_absolute_error(labels, blended)),
            })
    best_blend = max(blend_grid, key=lambda item: (item["auc"], -item["mae"]))
    result = {
        "year": year,
        "training": training,
        "inference": inference,
        "rows": len(labels),
        "label_mean": float(labels.mean()),
        "modernbert_auc": float(roc_auc_score(labels, predictions)),
        "modernbert_mae": float(mean_absolute_error(labels, predictions)),
        "tabular_auc": float(roc_auc_score(labels, tabular)),
        "tfidf_auc": float(roc_auc_score(labels, text)),
        "best_nonnegative_blend": best_blend,
        "blend_grid": blend_grid,
        "prediction_path": str(model_path / f"forward_{year}_reporting_predictions.npy"),
    }
    np.save(result["prediction_path"], predictions)
    np.savez(model_path / f"forward_{year}_components.npz", labels=labels, **components)
    (model_path / f"forward_{year}_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("MODERNBERT_GATE " + json.dumps(result, sort_keys=True), flush=True)
    return result


# Entrypoint

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forward-year", type=int, default=2019)
    args = parser.parse_args()
    forward_gate(args.forward_year)


if __name__ == "__main__":
    main()
