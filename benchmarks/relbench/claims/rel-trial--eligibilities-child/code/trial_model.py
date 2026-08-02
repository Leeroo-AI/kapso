# Imports

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging as transformers_logging

from trial_features import STRUCT_DIM

transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


# Configuration

MODEL_ID = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MODEL_REVISION = "e1354b7a3a09615f6aba48dfad4b7a613eef7062"
MODEL_SHA256 = "ad7bbb66376cfd6b2db3447192b034efe016337cbef135c35c411fd61b13c193"
TOKEN_VERSION = "lane1_tokens_v3"
SLOT_LENGTHS = {"study": 256, "concept_org": 192, "design_aux": 192}
TRAIN_BATCH_SIZE = 32
GRADIENT_ACCUMULATION = 2
INFERENCE_BATCH_SIZE = 64
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.06
POSITIVE_WEIGHT = 2.0


# Provenance

def resolve_model_path() -> Path:
    path = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            allow_patterns=["config.json", "pytorch_model.bin", "tokenizer_config.json", "vocab.txt"],
            local_files_only=False,
        )
    )
    weight_path = path / "pytorch_model.bin"
    digest = hashlib.sha256()
    with weight_path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != MODEL_SHA256:
        raise RuntimeError(f"Pinned model checksum mismatch: {digest.hexdigest()}")
    return path


# Token cache

class TokenStore:
    def __init__(self, path: Path, structured: np.ndarray):
        self.path = path
        self.structured = structured
        self.ids = {
            slot: np.load(path / f"{slot}_ids.npy", mmap_mode="r") for slot in SLOT_LENGTHS
        }
        self.lengths = {
            slot: np.load(path / f"{slot}_lengths.npy", mmap_mode="r") for slot in SLOT_LENGTHS
        }

    def batch(self, ids: np.ndarray, device: torch.device) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        ids = np.asarray(ids, dtype=np.int64)
        slots = {}
        for slot in SLOT_LENGTHS:
            token_ids = torch.from_numpy(np.asarray(self.ids[slot][ids], dtype=np.int64)).to(device)
            lengths = torch.from_numpy(np.asarray(self.lengths[slot][ids], dtype=np.int64)).to(device)
            slots[slot] = (token_ids, lengths)
        structured = torch.from_numpy(np.asarray(self.structured[ids], dtype=np.float32)).to(device)
        return slots, structured


def ensure_token_cache(features, structured: np.ndarray, feature_cache: Path, model_path: Path) -> tuple[TokenStore, dict]:
    target = feature_cache / TOKEN_VERSION
    metadata_path = target / "metadata.json"
    expected = {
        "version": TOKEN_VERSION,
        "revision": MODEL_REVISION,
        "rows": int(len(features)),
        "slot_lengths": SLOT_LENGTHS,
    }
    ready = metadata_path.exists()
    if ready:
        metadata = json.loads(metadata_path.read_text())
        ready = all(metadata.get(key) == value for key, value in expected.items())
        ready = ready and all(
            (target / f"{slot}_ids.npy").exists() and (target / f"{slot}_lengths.npy").exists()
            for slot in SLOT_LENGTHS
        )
    if ready:
        metadata["cached"] = True
        return TokenStore(target, structured), metadata

    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    temporary_files = []
    for slot, length in SLOT_LENGTHS.items():
        ids_tmp = target / f"{slot}_ids.{os.getpid()}.tmp.npy"
        lengths_tmp = target / f"{slot}_lengths.{os.getpid()}.tmp.npy"
        id_matrix = np.lib.format.open_memmap(ids_tmp, mode="w+", dtype=np.uint16, shape=(len(features), length))
        length_vector = np.lib.format.open_memmap(lengths_tmp, mode="w+", dtype=np.uint16, shape=(len(features),))
        values = features[slot].tolist()
        for start in range(0, len(values), 768):
            stop = min(start + 768, len(values))
            encoded = tokenizer(
                values[start:stop],
                padding="max_length",
                truncation=True,
                max_length=length,
                return_attention_mask=True,
                return_tensors="np",
                verbose=False,
            )
            id_matrix[start:stop] = encoded["input_ids"].astype(np.uint16)
            length_vector[start:stop] = encoded["attention_mask"].sum(axis=1).astype(np.uint16)
        id_matrix.flush()
        length_vector.flush()
        del id_matrix, length_vector, values
        temporary_files.append((ids_tmp, target / f"{slot}_ids.npy"))
        temporary_files.append((lengths_tmp, target / f"{slot}_lengths.npy"))
    for temporary, final in temporary_files:
        os.replace(temporary, final)
    metadata = {
        **expected,
        "elapsed_seconds": round(time.time() - started, 3),
        "cached": False,
    }
    temporary_metadata = target / f"metadata.{os.getpid()}.tmp.json"
    temporary_metadata.write_text(json.dumps(metadata, indent=2))
    os.replace(temporary_metadata, metadata_path)
    return TokenStore(target, structured), metadata


# LoRA

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: int = 16, dropout: float = 0.05):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scale


def inject_lora(module: nn.Module) -> int:
    targets = []
    for name, child in module.named_modules():
        if name.endswith("attention.self.query") or name.endswith("attention.self.value"):
            targets.append((name, child))
    for name, child in targets:
        parent = module
        pieces = name.split(".")
        for piece in pieces[:-1]:
            parent = getattr(parent, piece)
        setattr(parent, pieces[-1], LoRALinear(child, rank=8, alpha=16, dropout=0.05))
    if len(targets) != 24:
        raise RuntimeError(f"Expected 24 query/value LoRA targets, found {len(targets)}")
    return len(targets)


# Classifier

class HierarchicalClassifier(nn.Module):
    def __init__(self, model_path: Path, train_lora: bool = True, checkpointing: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        inject_lora(self.encoder)
        if not train_lora:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        if train_lora and checkpointing:
            self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        hidden = int(self.encoder.config.hidden_size)
        self.gate = nn.Sequential(nn.Linear(hidden, 256), nn.Tanh(), nn.Dropout(0.15), nn.Linear(256, 1))
        self.structured_mlp = nn.Sequential(
            nn.Linear(STRUCT_DIM, 256), nn.GELU(), nn.Dropout(0.15), nn.Linear(256, 256), nn.GELU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3 + 256, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 1),
        )

    def encode_slots(self, slots: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        max_length = SLOT_LENGTHS["study"]
        ids = []
        masks = []
        for slot in SLOT_LENGTHS:
            token_ids, lengths = slots[slot]
            pad = max_length - token_ids.shape[1]
            if pad:
                token_ids = functional.pad(token_ids, (0, pad), value=0)
            mask = torch.arange(max_length, device=token_ids.device).unsqueeze(0) < lengths.unsqueeze(1)
            ids.append(token_ids)
            masks.append(mask)
        input_ids = torch.cat(ids, dim=0)
        attention_mask = torch.cat(masks, dim=0)
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        batch = slots["study"][0].shape[0]
        return encoded.reshape(3, batch, -1).transpose(0, 1)

    def forward(self, slots: dict[str, tuple[torch.Tensor, torch.Tensor]], structured: torch.Tensor) -> torch.Tensor:
        vectors = self.encode_slots(slots)
        weights = torch.softmax(self.gate(vectors).squeeze(-1), dim=1)
        gated = torch.sum(vectors * weights.unsqueeze(-1), dim=1)
        mean = vectors.mean(dim=1)
        maximum = vectors.max(dim=1).values
        side = self.structured_mlp(structured.to(vectors.dtype))
        return self.classifier(torch.cat([gated, mean, maximum, side], dim=1)).squeeze(-1)


# Training and inference

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def predict_logits(model: nn.Module, store: TokenStore, ids: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    predictions = np.empty(len(ids), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(ids), INFERENCE_BATCH_SIZE):
            stop = min(start + INFERENCE_BATCH_SIZE, len(ids))
            slots, structured = store.batch(ids[start:stop], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(slots, structured)
            predictions[start:stop] = logits.float().cpu().numpy()
    model.train()
    return predictions.astype(np.float64)


def train_and_predict(
    store: TokenStore,
    train_ids: np.ndarray,
    labels: np.ndarray,
    predict_ids: np.ndarray,
    schedules: tuple[float, ...],
    seed: int,
    model_path: Path,
    train_lora: bool = True,
    deadline: float | None = None,
) -> tuple[dict[float, np.ndarray], dict]:
    seed_everything(seed)
    device = torch.device("cuda")
    model = HierarchicalClassifier(model_path, train_lora=train_lora, checkpointing=train_lora).to(device)
    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    batches_per_epoch = math.ceil(len(train_ids) / TRAIN_BATCH_SIZE)
    target_batches = {schedule: math.ceil(schedule * batches_per_epoch) for schedule in schedules}
    maximum_batches = max(target_batches.values())
    maximum_steps = math.ceil(maximum_batches / GRADIENT_ACCUMULATION)
    warmup_steps = max(1, math.ceil(maximum_steps * WARMUP_FRACTION))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup_steps)
    )
    positive_weight = torch.tensor(POSITIVE_WEIGHT, device=device)
    label_lookup = np.asarray(labels, dtype=np.float32)
    train_ids = np.asarray(train_ids, dtype=np.int64)
    started = time.time()
    batch_number = 0
    optimizer_steps = 0
    epoch_index = 0
    outputs: dict[float, np.ndarray] = {}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    while batch_number < maximum_batches:
        rng = np.random.default_rng(seed + epoch_index * 7919)
        order = rng.permutation(len(train_ids))
        for start in range(0, len(order), TRAIN_BATCH_SIZE):
            if batch_number >= maximum_batches:
                break
            positions = order[start : start + TRAIN_BATCH_SIZE]
            batch_ids = train_ids[positions]
            targets = torch.from_numpy(label_lookup[positions]).to(device)
            slots, structured = store.batch(batch_ids, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(slots, structured)
                loss = functional.binary_cross_entropy_with_logits(
                    logits.float(), targets, pos_weight=positive_weight
                ) / GRADIENT_ACCUMULATION
            loss.backward()
            accumulation += 1
            batch_number += 1
            due = [schedule for schedule, target in target_batches.items() if target == batch_number]
            if accumulation == GRADIENT_ACCUMULATION or due or batch_number == maximum_batches:
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                accumulation = 0
            if batch_number % 250 == 0:
                elapsed = time.time() - started
                rate = batch_number / max(elapsed, 1e-6)
                print(
                    f"[transformer] batch={batch_number}/{maximum_batches} "
                    f"rate={rate:.3f}/s elapsed={elapsed:.1f}s loss={float(loss) * GRADIENT_ACCUMULATION:.5f}",
                    flush=True,
                )
            for schedule in due:
                outputs[schedule] = predict_logits(model, store, predict_ids, device)
                print(
                    f"[transformer] checkpoint={schedule:.2f} predict_rows={len(predict_ids)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            if deadline is not None and time.time() >= deadline and len(outputs) > 0:
                batch_number = maximum_batches
                break
        epoch_index += 1
    elapsed = time.time() - started
    metadata = {
        "train_rows": int(len(train_ids)),
        "predict_rows": int(len(predict_ids)),
        "batches": int(batch_number),
        "optimizer_steps": int(optimizer_steps),
        "batch_rate": round(batch_number / max(elapsed, 1e-6), 4),
        "elapsed_seconds": round(elapsed, 3),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "schedules_completed": sorted(outputs),
        "train_lora": train_lora,
    }
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    return outputs, metadata
