from __future__ import annotations

import gc
import hashlib
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as transformers_logging

from data_pipeline import protocol_digest, register_artifact


MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
TOKEN_VERSION = "biomedbert_sections_128_256_256_256_v1"


class SectionEncoder(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        config = AutoConfig.from_pretrained(MODEL_NAME)
        config.hidden_dropout_prob = dropout
        self.encoder = AutoModel.from_pretrained(
            MODEL_NAME, config=config, attn_implementation="sdpa"
        )
        hidden = int(config.hidden_size)
        self.section_attention = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch, sections, length = input_ids.shape
        flat_ids = input_ids.reshape(batch * sections, length)
        flat_mask = attention_mask.reshape(batch * sections, length)
        encoded = self.encoder(input_ids=flat_ids, attention_mask=flat_mask)
        cls = encoded.last_hidden_state[:, 0].reshape(batch, sections, -1)
        section_mask = attention_mask.sum(dim=-1).gt(2)
        scores = self.section_attention(cls).squeeze(-1)
        scores = scores.masked_fill(~section_mask, -1e4)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(cls * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(-1)


def _pad_sequence(values: list[int], length: int, pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    values = values[:length]
    ids = np.full(length, pad_id, dtype=np.uint16)
    mask = np.zeros(length, dtype=np.uint8)
    ids[: len(values)] = np.asarray(values, dtype=np.uint16)
    mask[: len(values)] = 1
    return ids, mask


def tokenize_protocols(frame, cache_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    digest = protocol_digest(frame, TOKEN_VERSION)
    artifact_dir = cache_dir / "lane1_biomed_protocol_encoder"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"tokens_{digest[:20]}.npz"
    if path.exists():
        try:
            data = np.load(path, allow_pickle=False)
            keys = data["keys"].astype(str)
            if keys.tolist() == frame["_key"].astype(str).tolist():
                print(f"[encoder] token cache hit rows={len(frame)} key={digest[:12]}")
                return data["input_ids"], data["attention_mask"], data["eligibility_lengths"], digest
        except (ValueError, KeyError):
            pass
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    count = len(frame)
    input_ids = np.zeros((count, 4, 256), dtype=np.uint16)
    attention_mask = np.zeros((count, 4, 256), dtype=np.uint8)
    titles = tokenizer(
        frame["title_design"].tolist(),
        add_special_tokens=True,
        truncation=True,
        max_length=128,
        padding="max_length",
        return_attention_mask=True,
    )
    summaries = tokenizer(
        frame["summary"].tolist(),
        add_special_tokens=True,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_attention_mask=True,
    )
    input_ids[:, 0, :128] = np.asarray(titles["input_ids"], dtype=np.uint16)
    attention_mask[:, 0, :128] = np.asarray(titles["attention_mask"], dtype=np.uint8)
    input_ids[:, 1, :] = np.asarray(summaries["input_ids"], dtype=np.uint16)
    attention_mask[:, 1, :] = np.asarray(summaries["attention_mask"], dtype=np.uint8)
    eligibility_lengths = np.zeros(count, dtype=np.int32)
    pad_id = int(tokenizer.pad_token_id or 0)
    cls_id = int(tokenizer.cls_token_id)
    sep_id = int(tokenizer.sep_token_id)
    for index, text in enumerate(frame["eligibility"].tolist()):
        tokens = tokenizer.encode(text, add_special_tokens=False)
        eligibility_lengths[index] = len(tokens)
        beginning = [cls_id, *tokens[:254], sep_id]
        ending = [cls_id, *tokens[-254:], sep_id]
        ids, mask = _pad_sequence(beginning, 256, pad_id)
        input_ids[index, 2] = ids
        attention_mask[index, 2] = mask
        ids, mask = _pad_sequence(ending, 256, pad_id)
        input_ids[index, 3] = ids
        attention_mask[index, 3] = mask
    temporary = artifact_dir / f"tokens_{digest[:20]}.tmp.npz"
    np.savez(
        temporary,
        keys=np.asarray(frame["_key"].astype(str).tolist(), dtype=np.str_),
        input_ids=input_ids,
        attention_mask=attention_mask,
        eligibility_lengths=eligibility_lengths,
    )
    os.replace(temporary, path)
    register_artifact(
        cache_dir,
        {
            "name": "lane1 BiomedBERT section tokens",
            "path": str(path.relative_to(cache_dir)),
            "description": "Identity-redacted title/design, summary, and eligibility head/tail token arrays",
            "content_key": f"{TOKEN_VERSION}:{digest}",
            "rebuild_hint": "Run main.py to tokenize visible protocol sections with the pinned BiomedBERT tokenizer",
        },
    )
    print(f"[encoder] tokenized rows={count} key={digest[:12]}")
    return input_ids, attention_mask, eligibility_lengths, digest


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray | None,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    ids = torch.from_numpy(input_ids[indices].astype(np.int32, copy=False))
    masks = torch.from_numpy(attention_mask[indices].astype(np.uint8, copy=False))
    tensors = [ids, masks]
    if labels is not None:
        tensors.append(torch.from_numpy(labels[indices].astype(np.float32, copy=False)))
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )


def _predict(
    model: SectionEncoder,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    model.eval()
    loader = _loader(input_ids, attention_mask, None, indices, batch_size, False, 0)
    predictions = []
    with torch.inference_mode():
        for ids, masks in loader:
            ids = ids.cuda(non_blocking=True).long()
            masks = masks.cuda(non_blocking=True).long()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, masks)
            predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(predictions).astype(np.float64)


def train_encoder(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    seed: int,
    epochs: int,
    capture_epochs: set[int] | None = None,
    max_optimizer_steps: int | None = None,
) -> dict[int, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("BiomedBERT training requires the assigned CUDA device")
    transformers_logging.set_verbosity_error()
    transformers_logging.disable_progress_bar()
    _seed_everything(seed)
    torch.set_float32_matmul_precision("high")
    model = SectionEncoder(dropout=0.2).cuda()
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.section_attention.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 2e-5, "weight_decay": 0.01},
            {"params": head_parameters, "lr": 1e-4, "weight_decay": 0.01},
        ]
    )
    batch_size = 8
    accumulation = 3
    loader = _loader(
        input_ids,
        attention_mask,
        labels,
        train_indices,
        batch_size,
        True,
        seed,
    )
    updates_per_epoch = math.ceil(len(loader) / accumulation)
    total_updates = updates_per_epoch * epochs
    if max_optimizer_steps is not None:
        total_updates = min(total_updates, max_optimizer_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_updates * 0.1)),
        num_training_steps=max(1, total_updates),
    )
    captured = {}
    capture_epochs = capture_epochs or {epochs}
    optimizer.zero_grad(set_to_none=True)
    update_count = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0.0
        batch_count = 0
        for batch_index, (ids, masks, targets) in enumerate(loader, start=1):
            ids = ids.cuda(non_blocking=True).long()
            masks = masks.cuda(non_blocking=True).long()
            targets = targets.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids, masks)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            loss_total += float(loss.detach().cpu())
            batch_count += 1
            should_update = batch_index % accumulation == 0 or batch_index == len(loader)
            if should_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            if max_optimizer_steps is not None and update_count >= max_optimizer_steps:
                break
        print(
            f"[encoder] seed={seed} epoch={epoch} loss={loss_total/max(1,batch_count):.6f} "
            f"updates={update_count} elapsed={time.time()-started:.1f}s"
        )
        if epoch in capture_epochs or (
            max_optimizer_steps is not None and update_count >= max_optimizer_steps
        ):
            captured[epoch] = _predict(
                model, input_ids, attention_mask, predict_indices, batch_size=32
            )
        if max_optimizer_steps is not None and update_count >= max_optimizer_steps:
            break
    if not captured:
        captured[epoch] = _predict(
            model, input_ids, attention_mask, predict_indices, batch_size=32
        )
    del model, optimizer, scheduler, loader
    gc.collect()
    torch.cuda.empty_cache()
    return captured


def model_content_key(
    token_digest: str,
    labels: np.ndarray,
    train_keys: list[str],
    epochs: int,
    seed: int,
    role: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"section_encoder_train_v2")
    digest.update(token_digest.encode())
    digest.update(np.asarray(labels, dtype=np.float32).tobytes())
    digest.update("\x1f".join(train_keys).encode())
    digest.update(f"{epochs}:{seed}:{role}".encode())
    return digest.hexdigest()
