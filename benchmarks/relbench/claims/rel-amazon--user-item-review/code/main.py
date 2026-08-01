from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from relational_ranker import (
    FEATURE_NAMES,
    N_PRODUCTS,
    Catalog,
    RankingData,
    SemanticIndex,
    Snapshot,
    build_ranking_data,
    combine_data,
    infer_split,
    ranking_map,
    sample_inference_data,
    train_ranker,
    validate_predictions,
)


SEED = 1337
FULL_ORIGINS = ("2014-10-02", "2015-01-01", "2015-04-02", "2015-07-02")
DEBUG_ORIGINS = ("2015-04-02", "2015-07-02")


def task_paths() -> tuple[Path, Path, Path]:
    root = Path(os.environ["RELBENCH_CACHE_DIR"]) / "rel-amazon/tasks/user-item-review"
    return root / "train.parquet", root / "val.parquet", root / "test.parquet"


def select_origin(frame: pd.DataFrame, timestamp: str, size: int) -> tuple[pd.DataFrame, list]:
    origin = frame[frame["timestamp"] == pd.Timestamp(timestamp)].reset_index(drop=True)
    if len(origin) > size:
        generator = np.random.default_rng(SEED + int(pd.Timestamp(timestamp).timestamp()) % 100000)
        indices = np.sort(generator.choice(len(origin), size=size, replace=False))
        origin = origin.iloc[indices].reset_index(drop=True)
    seeds = origin[["timestamp", "customer_id"]].copy()
    truths = origin["product_id"].tolist()
    return seeds, truths


def replay_chain(catalog: Catalog, semantic_index: SemanticIndex | None, train_frame: pd.DataFrame, origins: tuple[str, ...], rows: int, semantic_enabled: bool, maximum_rounds: int) -> tuple[list[RankingData], list[int], list[dict[float, float]], list[float], list[dict]]:
    sampled_parts = []
    best_rounds = []
    fold_maps = []
    semantic_gains = []
    fold_records = []
    for index, origin in enumerate(origins):
        seeds, truths = select_origin(train_frame, origin, rows)
        snapshot = Snapshot.build(catalog, pd.Timestamp(origin))
        replay = build_ranking_data(catalog, snapshot, semantic_index, seeds, truths, semantic_enabled, 96, False)
        semantic_gains.append(replay.recall800 - replay.recall800_no_semantic)
        if index:
            training = combine_data(sampled_parts)
            model = train_ranker(training, maximum_rounds, replay)
            blend_maps = {weight: ranking_map(model, replay, weight) for weight in (0.0, 0.25, 0.5, 0.75, 1.0)}
            value = blend_maps[0.5]
            rounds = model.best_iteration or model.current_iteration()
            best_rounds.append(int(rounds))
            fold_maps.append(blend_maps)
            fold_records.append({
                "origin": origin,
                "rows": len(seeds),
                "map": value,
                "blend_maps": {str(weight): score for weight, score in blend_maps.items()},
                "rounds": int(rounds),
                "recall200": replay.recall200,
                "recall800": replay.recall800,
                "recall800_no_semantic": replay.recall800_no_semantic,
                "strata": replay.strata,
            })
            print(f"[replay] origin={origin} blend_maps={json.dumps(blend_maps)} rounds={rounds} recall200={replay.recall200:.6f} recall800={replay.recall800:.6f} semantic_gain={semantic_gains[-1]:.6f}", flush=True)
            del training, model
        sampled_parts.append(sample_inference_data(replay, 96))
        del replay, snapshot
        gc.collect()
    return sampled_parts, best_rounds, fold_maps, semantic_gains, fold_records


def validation_training(catalog: Catalog, snapshot: Snapshot, semantic_index: SemanticIndex | None, seeds: pd.DataFrame, truths: list, semantic_enabled: bool, chunk_size: int) -> list[RankingData]:
    parts = []
    for start in range(0, len(seeds), chunk_size):
        stop = min(start + chunk_size, len(seeds))
        part = build_ranking_data(
            catalog,
            snapshot,
            semantic_index,
            seeds.iloc[start:stop].reset_index(drop=True),
            truths[start:stop],
            semantic_enabled,
            96,
            True,
        )
        parts.append(part)
        print(f"[refit-data] validation_rows={stop}/{len(seeds)}", flush=True)
    return parts


def popularity_predictions(rows: int, snapshot: Snapshot) -> np.ndarray:
    values = []
    seen = set()
    for product in np.concatenate([snapshot.popularity, snapshot.global_padding, np.arange(N_PRODUCTS, dtype=np.int32)]):
        item = int(product)
        if item not in seen:
            seen.add(item)
            values.append(item)
            if len(values) == 10:
                break
    return np.tile(np.asarray(values, dtype=np.int64), (rows, 1))


def main() -> None:
    started = time.time()
    debug = "--debug" in sys.argv
    train_path, val_path, test_path = task_paths()
    train_frame = pd.read_parquet(train_path)
    val_seeds = pd.read_parquet(val_path, columns=["timestamp", "customer_id"])
    test_seeds = pd.read_parquet(test_path, columns=["timestamp", "customer_id"])
    print(f"[phase] task_loaded train={len(train_frame)} val={len(val_seeds)} test={len(test_seeds)} elapsed={time.time() - started:.1f}s", flush=True)
    catalog = Catalog.load(use_text=True)
    semantic_index = SemanticIndex(catalog) if catalog.embeddings is not None else None
    requested_semantic = semantic_index is not None and semantic_index.index is not None
    origins = DEBUG_ORIGINS if debug else FULL_ORIGINS
    replay_rows = 2000 if debug else 5000
    maximum_rounds = 160 if debug else 1200
    sampled_parts, fold_rounds, fold_maps, semantic_gains, fold_records = replay_chain(catalog, semantic_index, train_frame, origins, replay_rows, requested_semantic, maximum_rounds)
    semantic_enabled = requested_semantic and all(value > 0 for value in semantic_gains[1:])
    if requested_semantic and not semantic_enabled:
        print(f"[semantic] replay_gate_failed gains={semantic_gains}; rebuilding selected design without text", flush=True)
        del sampled_parts
        gc.collect()
        sampled_parts, fold_rounds, fold_maps, semantic_gains, fold_records = replay_chain(catalog, semantic_index, train_frame, origins, replay_rows, False, maximum_rounds)
    blend_statistics = {}
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        values = [record[weight] for record in fold_maps]
        mean = float(np.mean(values)) if values else 0.0
        sd = float(np.std(values)) if values else 0.0
        blend_statistics[weight] = (mean - 0.5 * sd, mean, sd)
    blend_weight = max(blend_statistics, key=lambda weight: (blend_statistics[weight][0], -weight))
    _, fold_mean, fold_sd = blend_statistics[blend_weight]
    replay_objective = fold_mean - 0.5 * fold_sd
    selected_rounds = int(np.median(fold_rounds)) if fold_rounds else maximum_rounds
    selected_rounds = max(40, min(maximum_rounds, selected_rounds))
    print(f"[selection] semantic={semantic_enabled} blend_weight={blend_weight} blend_statistics={json.dumps(blend_statistics)} replay_mean={fold_mean:.6f} replay_sd={fold_sd:.6f} objective={replay_objective:.6f} rounds={selected_rounds}", flush=True)
    model_a_training = combine_data(sampled_parts)
    model_a = train_ranker(model_a_training, selected_rounds)
    output_root = Path("output_data_generic_exp_0")
    output_root.mkdir(parents=True, exist_ok=True)
    model_a.save_model(output_root / "model_a.txt")
    validation_snapshot = Snapshot.build(catalog, pd.Timestamp(val_seeds["timestamp"].iloc[0]))
    validation_limit = min(len(val_seeds), 5000) if debug else len(val_seeds)
    validation_predictions = popularity_predictions(len(val_seeds), validation_snapshot)
    real_validation, _ = infer_split(
        catalog,
        validation_snapshot,
        semantic_index,
        val_seeds.iloc[:validation_limit].reset_index(drop=True),
        model_a,
        semantic_enabled,
        None,
        False,
        1000,
        blend_weight,
    )
    validation_predictions[:validation_limit] = real_validation
    validate_predictions(validation_predictions, len(val_seeds))
    np.save(output_root / "val_model_a_frozen.npy", validation_predictions)
    print(f"[freeze] model_a_validation_saved rows={len(validation_predictions)} elapsed={time.time() - started:.1f}s", flush=True)
    val_with_labels = pd.read_parquet(val_path)
    val_truths = val_with_labels["product_id"].tolist()
    refit_limit = min(len(val_seeds), 2000) if debug else len(val_seeds)
    validation_parts = validation_training(
        catalog,
        validation_snapshot,
        semantic_index,
        val_seeds.iloc[:refit_limit].reset_index(drop=True),
        val_truths[:refit_limit],
        semantic_enabled,
        1000,
    )
    model_b_training = combine_data(sampled_parts + validation_parts)
    model_b = train_ranker(model_b_training, selected_rounds)
    model_b.save_model(output_root / "model_b.txt")
    test_snapshot = Snapshot.build(catalog, pd.Timestamp(test_seeds["timestamp"].iloc[0]))
    test_limit = min(len(test_seeds), 5000) if debug else len(test_seeds)
    test_predictions = popularity_predictions(len(test_seeds), test_snapshot)
    real_test, _ = infer_split(
        catalog,
        test_snapshot,
        semantic_index,
        test_seeds.iloc[:test_limit].reset_index(drop=True),
        model_b,
        semantic_enabled,
        None,
        False,
        1000,
        blend_weight,
    )
    test_predictions[:test_limit] = real_test
    validate_predictions(test_predictions, len(test_seeds))
    run_dir = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "kapso_output"))
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "val_predictions.npy", validation_predictions)
    np.save(run_dir / "test_predictions.npy", test_predictions)
    diagnostics = {
        "debug": debug,
        "features": len(FEATURE_NAMES),
        "semantic_enabled": semantic_enabled,
        "selected_rounds": selected_rounds,
        "blend_weight": blend_weight,
        "replay_mean_map": fold_mean,
        "replay_sd_map": fold_sd,
        "replay_objective": replay_objective,
        "folds": fold_records,
        "elapsed_seconds": time.time() - started,
        "model_a_fit_rows": int(len(model_a_training.features)),
        "model_b_fit_rows": int(len(model_b_training.features)),
    }
    (run_dir / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    (output_root / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"[complete] val={validation_predictions.shape} test={test_predictions.shape} elapsed={time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
