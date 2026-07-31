from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from kapso_datasets.common import load_task, run_data_dir
from temporal_recommender import (
    HorizonGraphRecommender,
    HorizonTemporalMemory,
    RecommenderConfig,
    TemporalData,
    TemporalFeatureIndex,
    average_precision_rows,
    bootstrap_standard_error,
    checkpoint_payload,
    infer_predictions,
    quiet_runtime,
    register_artifact,
    set_seed,
    timestamp_day,
    train_query_group,
    validate_predictions,
)


# Orchestration

def phase(start: float, name: str) -> None:
    print(f"[horizon_tgn] {name}: elapsed={time.perf_counter() - start:.2f}s", flush=True)


def train_model_a(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    index: TemporalFeatureIndex,
    train: pd.DataFrame,
    config: RecommenderConfig,
    device: torch.device,
    start: float,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    timestamps = sorted(pd.to_datetime(train.timestamp.unique()))
    diagnostics = []
    for epoch in range(config.model_a_epochs):
        memory.reset()
        index.reset()
        processed = 0
        for timestamp in timestamps:
            day = int(np.datetime64(timestamp, "D").astype(np.int64))
            memory.advance(day)
            index.advance(day)
            group = train[pd.to_datetime(train.timestamp) == timestamp]
            if config.debug:
                remaining = config.debug_queries - processed
                if remaining <= 0:
                    break
                group = group.iloc[:remaining]
            year = int(timestamp.year)
            if not config.debug and epoch == 0 and year in (2018, 2019):
                fold_prediction = infer_predictions(
                    model,
                    memory,
                    index,
                    group.facility_id.to_numpy(np.int64),
                    config,
                    device,
                )
                fold_ap = average_precision_rows(fold_prediction, group.sponsor_id.values)
                clusters = index.data.city_group[group.facility_id.to_numpy(np.int64)]
                standard_error = bootstrap_standard_error(
                    fold_ap, clusters, config.seed + year
                )
                warm = index.fac_total[group.facility_id.to_numpy(np.int64)] > 0
                label_size = np.asarray([len(value) for value in group.sponsor_id.values])
                diagnostic = {
                    "year": year,
                    "rows": int(len(group)),
                    "map": float(fold_ap.mean()),
                    "cluster_bootstrap_se": standard_error,
                    "warm_rows": int(warm.sum()),
                    "warm_map": float(fold_ap[warm].mean()) if warm.any() else 0.0,
                    "cold_rows": int((~warm).sum()),
                    "cold_map": float(fold_ap[~warm].mean()) if (~warm).any() else 0.0,
                    "singleton_rows": int((label_size == 1).sum()),
                    "singleton_map": float(fold_ap[label_size == 1].mean()),
                    "multi_rows": int((label_size > 1).sum()),
                    "multi_map": float(fold_ap[label_size > 1].mean()),
                }
                diagnostics.append(diagnostic)
                print(
                    f"[horizon_tgn] forward_fold={json.dumps(diagnostic, sort_keys=True)}",
                    flush=True,
                )
            rng = np.random.default_rng(config.seed + epoch * 101 + year)
            loss = train_query_group(
                model,
                memory,
                index,
                group,
                optimizer,
                config,
                device,
                rng,
            )
            processed += len(group)
            print(
                f"[horizon_tgn] model=A epoch={epoch + 1} year={year} "
                f"queries={len(group)} loss={loss:.6f}",
                flush=True,
            )
        phase(start, f"Model A epoch {epoch + 1}/{config.model_a_epochs}")
    return diagnostics


def continue_model_b(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    index: TemporalFeatureIndex,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: RecommenderConfig,
    device: torch.device,
    start: float,
) -> None:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    combined = pd.concat([train, validation], ignore_index=True)
    timestamps = sorted(pd.to_datetime(combined.timestamp.unique()))
    for epoch in range(config.model_b_epochs):
        memory.reset()
        index.reset()
        for timestamp in timestamps:
            day = int(np.datetime64(timestamp, "D").astype(np.int64))
            memory.advance(day)
            index.advance(day)
            group = combined[pd.to_datetime(combined.timestamp) == timestamp]
            rng = np.random.default_rng(config.seed + 10000 + int(timestamp.year))
            loss = train_query_group(
                model,
                memory,
                index,
                group,
                optimizer,
                config,
                device,
                rng,
            )
            print(
                f"[horizon_tgn] model=B epoch={epoch + 1} year={timestamp.year} "
                f"queries={len(group)} loss={loss:.6f}",
                flush=True,
            )
        phase(start, f"Model B continuation {epoch + 1}/{config.model_b_epochs}")


def rolling_fallback(context) -> bool:
    if len(context.val) != 0:
        return False
    count = len(context.test)
    output = np.tile(np.arange(10, dtype=np.int64), (count, 1))
    np.save(run_data_dir() / "test_predictions.npy", output)
    print(f"[horizon_tgn] rolling fallback wrote test{output.shape}", flush=True)
    return True


def restore_checkpoint(
    model: HorizonGraphRecommender,
    memory: HorizonTemporalMemory,
    path: Path,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    memory.memory.copy_(
        payload["memory"].to(device=memory.device, dtype=torch.float32)
    )
    cutoff = int(payload["cutoff"])
    memory.cutoff = cutoff
    memory.pointer = min(
        int(np.searchsorted(memory.data.date_day, cutoff, side="right")),
        memory.data.event_limit,
    )
    return cutoff


def main() -> None:
    quiet_runtime()
    start = time.perf_counter()
    debug = "--debug" in sys.argv
    config = RecommenderConfig(debug=debug)
    config.apply_debug()
    set_seed(config.seed)
    context = load_task()
    if rolling_fallback(context):
        return
    train = context.train.df.copy()
    validation = context.val.df.copy()
    test = context.test.df.copy()
    phase(start, "task loaded")
    cache_root = Path(os.environ["KAPSO_SHARED_CACHE_DIR"])
    database_root = Path(os.environ["RELBENCH_CACHE_DIR"]) / context.dataset_name / "db"
    data = TemporalData(config, cache_root, database_root)
    register_artifact(
        cache_root,
        "lane3 horizon TGN chronological arrays",
        data.cache_dir,
        "Signed-hashed chronological facility-sponsor events, late evidence, static hashes, and sparse temporal indices.",
        f"{config.version}:arrays:v3",
        "Delete the versioned directory and run main.py to rebuild from the sanitized RelBench parquet tables.",
    )
    phase(start, f"data arrays ready events={data.event_limit}/{len(data.date_day)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the horizon temporal recommender")
    device = torch.device("cuda")
    validation_day = int(timestamp_day(validation.timestamp)[0])
    test_day = int(timestamp_day(test.timestamp)[0])
    checkpoint_a_path = data.cache_dir / (
        "model_a_debug.pt" if debug else "model_a_full.pt"
    )
    checkpoint_b_path = data.cache_dir / (
        "model_b_debug.pt" if debug else "model_b_full.pt"
    )
    model_a = HorizonGraphRecommender(data, config).to(device)
    memory_a = HorizonTemporalMemory(data, config, device)
    index_a = TemporalFeatureIndex(data, config)
    reuse_full = not debug and checkpoint_a_path.exists() and checkpoint_b_path.exists()
    if reuse_full:
        restored_a_day = restore_checkpoint(model_a, memory_a, checkpoint_a_path)
        if restored_a_day != validation_day:
            raise RuntimeError(
                f"Model A checkpoint cutoff {restored_a_day} != validation {validation_day}"
            )
        diagnostics = [{"checkpoint_reused": True}]
        phase(start, "Model A checkpoint restored")
    else:
        diagnostics = train_model_a(
            model_a,
            memory_a,
            index_a,
            train,
            config,
            device,
            start,
        )
        memory_a.advance(validation_day)
    index_a.advance(validation_day)
    validation_prediction = infer_predictions(
        model_a,
        memory_a,
        index_a,
        validation.facility_id.to_numpy(np.int64),
        config,
        device,
    )
    validate_predictions(
        validation_prediction,
        len(validation),
        data.n_sponsors,
        config.eval_k,
    )
    output_dir = run_data_dir()
    np.save(output_dir / "val_predictions.npy", validation_prediction)
    if not reuse_full:
        torch.save(
            checkpoint_payload(model_a, memory_a, validation_day, config),
            checkpoint_a_path,
        )
    register_artifact(
        cache_root,
        "lane3 horizon TGN Model A",
        checkpoint_a_path,
        "Train-label-only neural weights and memory at the validation cutoff.",
        f"{config.version}:model_a:{'debug' if debug else 'full'}",
        "Run main.py at the matching fidelity to retrain chronologically.",
    )
    phase(start, "Model A validation predictions frozen")
    if reuse_full:
        model_b = HorizonGraphRecommender(data, config).to(device)
        memory_b = HorizonTemporalMemory(data, config, device)
        index_b = TemporalFeatureIndex(data, config)
        restored_b_day = restore_checkpoint(model_b, memory_b, checkpoint_b_path)
        if restored_b_day != test_day:
            raise RuntimeError(
                f"Model B checkpoint cutoff {restored_b_day} != test {test_day}"
            )
        index_b.advance(test_day)
        phase(start, "Model B checkpoint restored")
    elif config.model_b_epochs > 0:
        model_b = copy.deepcopy(model_a)
        memory_b = HorizonTemporalMemory(data, config, device)
        index_b = TemporalFeatureIndex(data, config)
        continue_model_b(
            model_b,
            memory_b,
            index_b,
            train,
            validation,
            config,
            device,
            start,
        )
    else:
        model_b = model_a
        memory_b = memory_a
        index_b = index_a
    if not reuse_full:
        memory_b.advance(test_day)
        index_b.advance(test_day)
    test_prediction = infer_predictions(
        model_b,
        memory_b,
        index_b,
        test.facility_id.to_numpy(np.int64),
        config,
        device,
    )
    validate_predictions(
        test_prediction,
        len(test),
        data.n_sponsors,
        config.eval_k,
    )
    np.save(output_dir / "test_predictions.npy", test_prediction)
    np.save(output_dir / "val_predictions.npy", validation_prediction)
    if not reuse_full:
        torch.save(
            checkpoint_payload(model_b, memory_b, test_day, config),
            checkpoint_b_path,
        )
    register_artifact(
        cache_root,
        "lane3 horizon TGN Model B",
        checkpoint_b_path,
        "Train-plus-validation continuation weights and memory at the test cutoff.",
        f"{config.version}:model_b:{'debug' if debug else 'full'}",
        "Run main.py at the matching fidelity to retrain with the fixed continuation.",
    )
    metrics = {
        "debug": debug,
        "events_replayed": int(data.event_limit),
        "model_a_fit_labels": "train",
        "model_b_fit_labels": "train+validation" if config.model_b_epochs > 0 else "train",
        "validation_prediction_source": "Model A",
        "forward_folds": diagnostics,
        "elapsed_seconds": time.perf_counter() - start,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    phase(start, f"outputs complete val={validation_prediction.shape} test={test_prediction.shape}")


if __name__ == "__main__":
    main()
