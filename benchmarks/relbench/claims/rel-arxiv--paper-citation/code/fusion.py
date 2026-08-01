from __future__ import annotations

import json
import logging
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)


def save_array(path: Path, value: np.ndarray):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def tokenize_papers(papers, cache_dir: Path, maximum_length: int = 192):
    from transformers import AutoTokenizer
    from transformers.utils import logging as transformer_logging

    transformer_logging.set_verbosity_error()
    version = f"allenai_specter2_base_len{maximum_length}_v2"
    ids_path = cache_dir / "token_ids.npy"
    mask_path = cache_dir / "token_mask.npy"
    count_path = cache_dir / "token_counts.npy"
    manifest_path = cache_dir / "token_manifest.json"
    if manifest_path.exists() and ids_path.exists() and mask_path.exists() and count_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") == version and manifest.get("rows") == len(papers):
            print("[text] loaded cached SPECTER2 tokens")
            return (
                np.load(ids_path, mmap_mode="r"),
                np.load(mask_path, mmap_mode="r"),
                np.load(count_path, mmap_mode="r"),
            )

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base", local_files_only=True)
    titles = papers.sort_values("Paper_ID")["Title"].astype(str).tolist()
    abstracts = papers.sort_values("Paper_ID")["Abstract"].astype(str).tolist()
    token_ids = np.empty((len(papers), maximum_length), dtype=np.int32)
    token_mask = np.empty((len(papers), maximum_length), dtype=np.uint8)
    token_counts = np.empty(len(papers), dtype=np.int32)
    batch_size = 2048
    for start in range(0, len(papers), batch_size):
        stop = min(start + batch_size, len(papers))
        texts = [
            titles[index] + " " + tokenizer.sep_token + " " + abstracts[index]
            for index in range(start, stop)
        ]
        lengths = tokenizer(texts, add_special_tokens=True, truncation=False, return_length=True)
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=maximum_length,
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        token_counts[start:stop] = np.asarray(lengths["length"], dtype=np.int32)
        token_ids[start:stop] = np.asarray(encoded["input_ids"], dtype=np.int32)
        token_mask[start:stop] = np.asarray(encoded["attention_mask"], dtype=np.uint8)
    save_array(ids_path, token_ids)
    save_array(mask_path, token_mask)
    save_array(count_path, token_counts)
    manifest_path.write_text(json.dumps({"version": version, "rows": len(papers)}, indent=2))
    print(f"[text] tokenized rows={len(papers)} elapsed={time.time() - started:.1f}s")
    return (
        np.load(ids_path, mmap_mode="r"),
        np.load(mask_path, mmap_mode="r"),
        np.load(count_path, mmap_mode="r"),
    )


def fit_normalizer(features: np.ndarray):
    mean = np.asarray(features.mean(axis=0, dtype=np.float64), dtype=np.float32)
    variance = np.asarray(((features.astype(np.float64) - mean) ** 2).mean(axis=0), dtype=np.float32)
    scale = np.sqrt(np.maximum(variance, 1e-8)).astype(np.float32)
    scale[scale < 1e-4] = 1.0
    return mean, scale


def temporal_weights(days: np.ndarray, half_life_days: float | None):
    if half_life_days is None:
        return np.ones(len(days), dtype=np.float32)
    days = np.asarray(days, dtype=np.float64)
    distance = days.max() - days
    weights = np.exp2(-distance / half_life_days)
    return np.asarray(weights / weights.mean(), dtype=np.float32)


def train_anchor(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    rounds: int,
    valid_features: np.ndarray | None = None,
    valid_labels: np.ndarray | None = None,
):
    import lightgbm as lgb

    started = time.time()
    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "min_data_in_leaf": 300,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "max_bin": 255,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        "seed": 1337,
        "feature_fraction_seed": 1337,
        "bagging_seed": 1337,
    }
    train_set = lgb.Dataset(
        np.asarray(features),
        label=np.asarray(labels),
        weight=np.asarray(weights),
        categorical_feature=[8, 10],
        free_raw_data=False,
    )
    valid_sets = None
    callbacks = [lgb.log_evaluation(0)]
    if valid_features is not None:
        valid_set = lgb.Dataset(
            np.asarray(valid_features),
            label=np.asarray(valid_labels),
            reference=train_set,
            categorical_feature=[8, 10],
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        callbacks.append(lgb.early_stopping(80, verbose=False))
    model = lgb.train(
        params,
        train_set,
        num_boost_round=rounds,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )
    print(
        f"[anchor] rows={len(labels)} rounds={model.best_iteration or rounds} "
        f"elapsed={time.time() - started:.1f}s"
    )
    return model


def anchor_predict(model, features: np.ndarray):
    rounds = model.best_iteration if model.best_iteration else model.current_iteration()
    return np.asarray(model.predict(np.asarray(features), num_iteration=rounds), dtype=np.float64)


def percentile_rank(values: np.ndarray):
    from scipy.stats import rankdata

    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 0.5) / len(values)


def calibrated_rank_blend(anchor, neural, anchor_weight: float, calibrator=None):
    blended = anchor_weight * percentile_rank(anchor) + (1.0 - anchor_weight) * percentile_rank(neural)
    if calibrator is None:
        return blended
    return np.clip(calibrator.predict(blended), 1e-5, 1.0 - 1e-5)


def fit_calibrator(values: np.ndarray, labels: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    calibrator = IsotonicRegression(y_min=1e-4, y_max=1.0 - 1e-4, out_of_bounds="clip")
    calibrator.fit(np.asarray(values), np.asarray(labels))
    return calibrator


def make_fusion_model():
    import torch
    from adapters import AutoAdapterModel
    from torch import nn
    from transformers.utils import logging as transformer_logging

    transformer_logging.set_verbosity_error()

    class FusionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoAdapterModel.from_pretrained(
                "allenai/specter2_base",
                local_files_only=True,
            )
            self.encoder.load_adapter(
                "allenai/specter2",
                source="hf",
                load_as="proximity",
                local_files_only=True,
            )
            self.encoder.set_active_adapters("proximity")
            self.tabular = nn.Sequential(
                nn.Linear(80, 128),
                nn.GELU(),
                nn.Dropout(0.10),
            )
            self.head = nn.Sequential(
                nn.Linear(896, 256),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(256, 1),
            )

        def forward(self, input_ids, attention_mask, tabular):
            encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            cls = encoded.last_hidden_state[:, 0]
            tabular_hidden = self.tabular(tabular)
            return self.head(torch.cat((cls, tabular_hidden), dim=1)).squeeze(1)

    return FusionModel()


def fusion_predict(
    model,
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    paper_ids: np.ndarray,
    features: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    batch_size: int = 512,
):
    import torch

    started = time.time()
    device = torch.device("cuda")
    result = np.empty(len(paper_ids), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(paper_ids), batch_size):
            stop = min(start + batch_size, len(paper_ids))
            ids = np.asarray(paper_ids[start:stop], dtype=np.int64)
            input_ids = torch.from_numpy(np.asarray(token_ids[ids], dtype=np.int64)).to(device)
            attention_mask = torch.from_numpy(np.asarray(token_mask[ids], dtype=np.int64)).to(device)
            tabular_values = (np.asarray(features[start:stop], dtype=np.float32) - mean) / scale
            tabular = torch.from_numpy(tabular_values).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask, tabular)
            result[start:stop] = torch.sigmoid(logits).float().cpu().numpy()
    print(f"[fusion] inference rows={len(paper_ids)} elapsed={time.time() - started:.1f}s")
    return result


def train_fusion(
    token_ids: np.ndarray,
    token_mask: np.ndarray,
    paper_ids: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    sample_weights: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    epochs: float,
    batch_size: int = 128,
    maximum_steps: int | None = None,
    checkpoints: list[float] | None = None,
    evaluation_data=None,
):
    import torch
    from torch import nn

    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = make_fusion_model().to(device)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = list(model.tabular.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 2e-5},
            {"params": head_parameters, "lr": 8e-4},
        ],
        weight_decay=1e-2,
    )
    natural_steps = int(math.ceil(len(labels) * epochs / batch_size))
    total_steps = min(natural_steps, maximum_steps) if maximum_steps is not None else natural_steps
    warmup_steps = max(1, int(total_steps * 0.05))

    def learning_rate_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.05, (total_steps - step) / max(1, total_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    rng = np.random.default_rng(1337)
    order = rng.permutation(len(labels))
    position = 0
    checkpoints = sorted(checkpoints or [])
    checkpoint_steps = {max(1, int(round(total_steps * fraction))): fraction for fraction in checkpoints}
    checkpoint_predictions = {}
    started = time.time()
    model.train()
    for step in range(total_steps):
        if position + batch_size > len(order):
            order = rng.permutation(len(labels))
            position = 0
        batch_indices = order[position : position + batch_size]
        position += batch_size
        ids = np.asarray(paper_ids[batch_indices], dtype=np.int64)
        input_ids = torch.from_numpy(np.asarray(token_ids[ids], dtype=np.int64)).to(device)
        attention_mask = torch.from_numpy(np.asarray(token_mask[ids], dtype=np.int64)).to(device)
        tabular_values = (np.asarray(features[batch_indices], dtype=np.float32) - mean) / scale
        tabular = torch.from_numpy(tabular_values).to(device)
        target = torch.from_numpy(np.asarray(labels[batch_indices], dtype=np.float32)).to(device)
        weights = torch.from_numpy(np.asarray(sample_weights[batch_indices], dtype=np.float32)).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask, tabular)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = (losses * weights).sum() / weights.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed = step + 1
        if completed % 500 == 0 or completed == total_steps:
            print(
                f"[fusion] step={completed}/{total_steps} loss={loss.detach().float().item():.5f} "
                f"elapsed={time.time() - started:.1f}s"
            )
        if completed in checkpoint_steps:
            if evaluation_data is None:
                raise RuntimeError("Checkpoint evaluation data is required")
            fraction = checkpoint_steps[completed]
            checkpoint_predictions[fraction] = fusion_predict(
                model,
                token_ids,
                token_mask,
                evaluation_data["paper_ids"],
                evaluation_data["features"],
                mean,
                scale,
            )
            model.train()
    return model, checkpoint_predictions
