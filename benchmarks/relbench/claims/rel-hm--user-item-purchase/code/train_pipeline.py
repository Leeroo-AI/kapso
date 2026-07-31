from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from basket_transformer import BasketTransformer, move_batch
from hm_data import EpisodeDataset, InferenceDataset, choose_recent_indices
from kapso_datasets.common import eval_map_at_k
from relational import (
    build_bank,
    build_candidate_pool,
    relational_features,
)


def log_phase(name: str, start: float, extra: str = ""):
    elapsed = time.time() - start
    print(f"[basket] phase={name} elapsed_seconds={elapsed:.1f} {extra}".rstrip())


def make_model(state):
    return BasketTransformer(
        state.n_items,
        state.item_features,
        state.item_cardinalities,
        state.customer_cardinalities,
        state.popularity_buckets,
        state.price_buckets,
        dimension=128,
        layers=3,
        heads=4,
        feedforward=512,
        dropout=0.15,
    )


def loader_for(dataset, batch_size: int, shuffle: bool, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        collate_fn=dataset.collate,
        generator=generator,
        drop_last=False,
    )


def train_epochs(
    model,
    dataset,
    bank,
    epochs: int,
    repeat_weight: float,
    device,
    seed: int,
    optimizer=None,
    batch_size: int = 2048,
):
    model.train()
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.002, weight_decay=1e-5
        )
    family_hard = torch.from_numpy(bank.family_top[:, :8]).to(device)
    transition_hard = torch.from_numpy(bank.transition_top[:, :8]).to(device)
    popularity_weights = torch.from_numpy(bank.sampling_weights).to(device)
    loader = loader_for(dataset, batch_size, True, seed)
    total_start = time.time()
    step = 0
    for epoch in range(epochs):
        epoch_start = time.time()
        totals = {
            "loss": 0.0,
            "sigmoid": 0.0,
            "contrastive": 0.0,
            "gate": 0.0,
            "gate_mean": 0.0,
        }
        count = 0
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, diagnostics = model.training_loss(
                    batch,
                    popularity_weights,
                    family_hard,
                    transition_hard,
                    repeat_weight,
                    sampled_negatives=256,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key, value in diagnostics.items():
                totals[key] += float(value)
            count += 1
            step += 1
            if step % 100 == 0:
                print(
                    f"[basket] train_step={step} loss={totals['loss']/count:.5f} "
                    f"gate={totals['gate_mean']/count:.5f}"
                )
        summary = " ".join(
            f"{key}={value/max(1,count):.5f}" for key, value in totals.items()
        )
        log_phase(
            f"train_epoch_{epoch + 1}",
            epoch_start,
            f"queries={len(dataset)} {summary}",
        )
    log_phase("training_complete", total_start, f"epochs={epochs}")
    return optimizer


@torch.no_grad()
def catalog_vectors(model, origin: int, device):
    output = []
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, model.n_items, 8192):
            stop = min(model.n_items, start + 8192)
            ids = torch.arange(start, stop, device=device)
            output.append(model.article_encoder(ids, origin))
    return torch.cat(output, dim=0)


@torch.no_grad()
def retrieve_dataset(model, dataset, top_k: int, device, batch_size: int = 512):
    model.eval()
    origin = int(dataset.origins[0])
    catalog = catalog_vectors(model, origin, device)
    popularity = model.article_encoder.popularity_buckets[origin].to(device)
    output = np.empty((len(dataset), top_k), dtype=np.int32)
    output_score = np.empty((len(dataset), top_k), dtype=np.float32)
    loader = loader_for(dataset, batch_size, False, 1337)
    offset = 0
    for batch in loader:
        size = len(batch["customer"])
        batch = move_batch(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sequence, _, _ = model.encode(batch)
            score = sequence @ catalog.T / np.sqrt(model.dimension)
            score = score + popularity.to(score.dtype).unsqueeze(0) * 0.05
            values, items = score.topk(top_k, dim=1)
        output[offset : offset + size] = items.cpu().numpy()
        output_score[offset : offset + size] = values.float().cpu().numpy()
        offset += size
    del catalog
    torch.cuda.empty_cache()
    return output, output_score


@torch.no_grad()
def neural_pool_features(model, dataset, pool, device, batch_size: int = 384):
    model.eval()
    output = np.empty((len(dataset), pool.shape[1], 9), dtype=np.float32)
    loader = loader_for(dataset, batch_size, False, 1337)
    offset = 0
    for batch in loader:
        size = len(batch["customer"])
        batch = move_batch(batch, device)
        candidates = torch.from_numpy(pool[offset : offset + size]).to(
            device, non_blocking=True
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sequence, baskets, gate = model.encode(batch)
            mixed, explore, repeat, repeat_features = model.score_candidates(
                batch, sequence, baskets, gate, candidates
            )
            features = torch.cat(
                [
                    mixed.unsqueeze(-1),
                    explore.unsqueeze(-1),
                    repeat.unsqueeze(-1),
                    repeat_features,
                    gate.unsqueeze(1).expand(-1, candidates.shape[1], -1),
                ],
                dim=2,
            )
        output[offset : offset + size] = features.float().cpu().numpy()
        offset += size
    return output


def prepare_features(model, state, bank, dataset, device):
    start = time.time()
    explore, _ = retrieve_dataset(model, dataset, 200, device)
    log_phase("neural_retrieval", start, f"queries={len(dataset)}")
    start = time.time()
    pool, ranks, fallback = build_candidate_pool(state, bank, dataset, explore)
    log_phase("candidate_union", start, f"queries={len(dataset)} width=250")
    start = time.time()
    relation = relational_features(state, bank, dataset, pool, ranks)
    neural = neural_pool_features(model, dataset, pool, device)
    features = np.concatenate([relation, neural], axis=2)
    log_phase(
        "reranker_features",
        start,
        f"queries={len(dataset)} features={features.shape[2]}",
    )
    return pool, features, fallback


def pool_labels(dataset, pool):
    labels = np.zeros(pool.shape, dtype=np.int8)
    hit = 0
    total = 0
    groups = 0
    for row_index in range(len(dataset)):
        positives = np.asarray(dataset.positives[row_index], dtype=np.int32)
        labels[row_index] = np.isin(pool[row_index], positives)
        found = int(labels[row_index].sum())
        hit += found
        total += len(positives)
        groups += int(found > 0)
    print(
        f"[basket] candidate_recall={hit/max(1,total):.6f} "
        f"positive_groups={groups}/{len(dataset)}"
    )
    return labels


def fit_reranker(features, labels, trees: int):
    import lightgbm as lgb

    rows, width, feature_count = features.shape
    train_set = lgb.Dataset(
        features.reshape(rows * width, feature_count),
        label=labels.reshape(-1),
        group=np.full(rows, width, dtype=np.int32),
        free_raw_data=False,
    )
    params = {
        "objective": "lambdarank",
        "metric": "map",
        "eval_at": [12],
        "label_gain": [0, 1],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l2": 0.1,
        "lambdarank_truncation_level": 12,
        "verbosity": -1,
        "num_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
        "force_col_wise": True,
        "seed": 1337,
    }
    return lgb.train(params, train_set, num_boost_round=trees)


def rerank(ranker, pool, features):
    rows, width, feature_count = features.shape
    score = ranker.predict(features.reshape(rows * width, feature_count)).reshape(
        rows, width
    )
    choice = np.argpartition(score, -12, axis=1)[:, -12:]
    choice_score = np.take_along_axis(score, choice, axis=1)
    order = np.argsort(choice_score, axis=1)[:, ::-1]
    choice = np.take_along_axis(choice, order, axis=1)
    return np.take_along_axis(pool, choice, axis=1).astype(np.int64)


def forward_study(state, device, cache_file=None):
    if cache_file is not None and Path(cache_file).exists():
        cached = json.loads(Path(cache_file).read_text())
        print(
            f"[basket] forward_selection_cached repeat_weight={cached['repeat_weight']:.2f} "
            f"epochs={cached['epochs']} mean_map={cached['mean_map']:.6f}"
        )
        return float(cached["repeat_weight"]), int(cached["epochs"])
    cutoff = pd.Timestamp("2020-08-10")
    source = state.train[state.train.timestamp <= cutoff]
    source_indices = choose_recent_indices(source, 60000, 1337)
    training = EpisodeDataset(state, [source], [source_indices])
    minimum_day = int(training.days.min())
    from relational import build_bank

    bank = build_bank(state, minimum_day)
    folds = []
    for timestamp in (pd.Timestamp("2020-08-17"), pd.Timestamp("2020-08-24")):
        frame = state.train[state.train.timestamp == timestamp].iloc[:512]
        folds.append((frame, EpisodeDataset(state, [frame])))
    results = []
    for repeat_weight in (0.20, 0.35):
        model = make_model(state).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.002, weight_decay=1e-5
        )
        for epoch in range(1, 4):
            optimizer = train_epochs(
                model,
                training,
                bank,
                1,
                repeat_weight,
                device,
                1337 + epoch,
                optimizer,
            )
            scores = []
            for frame, fold in folds:
                prediction, _ = retrieve_dataset(model, fold, 12, device)
                scores.append(eval_map_at_k(prediction, frame.article_id, 12))
            mean = float(np.mean(scores))
            stability = float(np.std(scores))
            results.append((mean, -stability, -epoch, -abs(repeat_weight - 0.20), repeat_weight, epoch, scores))
            print(
                f"[basket] forward_study repeat_weight={repeat_weight:.2f} epoch={epoch} "
                f"fold_map={json.dumps(scores)} mean={mean:.6f}"
            )
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
    selected = max(results)
    print(
        f"[basket] forward_selection repeat_weight={selected[4]:.2f} "
        f"epochs={selected[5]} mean_map={selected[0]:.6f}"
    )
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_file).write_text(
            json.dumps(
                {
                    "repeat_weight": float(selected[4]),
                    "epochs": int(selected[5]),
                    "mean_map": float(selected[0]),
                    "results": [
                        {
                            "mean_map": float(result[0]),
                            "stability": float(-result[1]),
                            "repeat_weight": float(result[4]),
                            "epoch": int(result[5]),
                            "fold_map": [float(value) for value in result[6]],
                        }
                        for result in results
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    return float(selected[4]), int(selected[5])


def sample_reranker_dataset(state, count: int, include_validation: bool, seed: int):
    generator = np.random.default_rng(seed)
    if include_validation:
        validation_choice = generator.choice(
            len(state.val), min(count, len(state.val)), replace=False
        )
        return EpisodeDataset(state, [state.val], [np.sort(validation_choice)])
    latest_origin = sorted(state.train.timestamp.unique())[-1]
    train_rows = state.train[state.train.timestamp == latest_origin]
    train_choice = generator.choice(
        len(train_rows), min(count, len(train_rows)), replace=False
    )
    return EpisodeDataset(state, [train_rows], [np.sort(train_choice)])


def fit_model_reranker(model, state, bank, dataset, device, trees: int):
    pool, features, _ = prepare_features(model, state, bank, dataset, device)
    labels = pool_labels(dataset, pool)
    start = time.time()
    reranker = fit_reranker(features, labels, trees)
    log_phase("reranker_fit", start, f"queries={len(dataset)} trees={trees}")
    del features, labels
    gc.collect()
    return reranker


def predict_with_reranker(model, reranker, state, bank, dataset, device):
    pool, features, fallback = prepare_features(
        model, state, bank, dataset, device
    )
    start = time.time()
    prediction = rerank(reranker, pool, features)
    log_phase("reranker_predict", start, f"queries={len(dataset)}")
    del pool, features
    gc.collect()
    return prediction, fallback


def slice_report(state, prediction):
    dataset = EpisodeDataset(state, [state.val])
    row_scores = np.zeros(len(dataset), dtype=np.float64)
    history_count = np.zeros(len(dataset), dtype=np.int32)
    recency = np.full(len(dataset), 999, dtype=np.int32)
    target_size = np.zeros(len(dataset), dtype=np.int32)
    repeat_row = np.zeros(len(dataset), dtype=np.bool_)
    for index in range(len(dataset)):
        row = dataset[index]
        _, query_day, _, items, item_days, _, _, positives = row
        true = set(positives.tolist())
        hits = np.asarray([int(item in true) for item in prediction[index]], dtype=np.float64)
        precision = np.cumsum(hits) / np.arange(1, 13)
        row_scores[index] = float((precision * hits).sum() / min(12, len(true)))
        history_count[index] = len(items)
        if len(item_days):
            recency[index] = query_day - int(item_days[-1])
        target_size[index] = len(positives)
        repeat_row[index] = bool(np.isin(positives, items).any())
    reports = {}
    definitions = {
        "history": (
            history_count,
            [(-1, 0, "0"), (0, 5, "1-5"), (5, 20, "6-20"), (20, 50, "21-50"), (50, 10**9, "51+")],
        ),
        "recency": (
            recency,
            [(-1, 7, "0-7"), (7, 28, "8-28"), (28, 90, "29-90"), (90, 998, "91+"), (998, 10**9, "none")],
        ),
        "target_size": (
            target_size,
            [(0, 1, "1"), (1, 2, "2"), (2, 4, "3-4"), (4, 10**9, "5+")],
        ),
    }
    for axis, (values, bins) in definitions.items():
        reports[axis] = {}
        for low, high, label in bins:
            mask = (values > low) & (values <= high)
            reports[axis][label] = {
                "count": int(mask.sum()),
                "map": float(row_scores[mask].mean()) if mask.any() else 0.0,
            }
    for value, label in ((False, "novel_only"), (True, "any_repeat")):
        mask = repeat_row == value
        reports[label] = {
            "count": int(mask.sum()),
            "map": float(row_scores[mask].mean()),
        }
    print(f"[basket] slice_metrics={json.dumps(reports, sort_keys=True)}")
    return reports


def validate_predictions(prediction, rows: int, n_items: int):
    if prediction.shape != (rows, 12):
        raise RuntimeError(f"prediction shape {prediction.shape} != {(rows, 12)}")
    if not np.issubdtype(prediction.dtype, np.integer):
        raise RuntimeError(f"prediction dtype {prediction.dtype} is not integer")
    if prediction.min() < 0 or prediction.max() >= n_items:
        raise RuntimeError("prediction id outside catalog")
    if np.any(np.sort(prediction, axis=1)[:, 1:] == np.sort(prediction, axis=1)[:, :-1]):
        raise RuntimeError("duplicate prediction id within a row")


def save_metrics(out: Path, metrics):
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
