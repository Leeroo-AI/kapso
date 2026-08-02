from __future__ import annotations

import fcntl
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from graph_pipeline import (
    BuildConfig,
    CandidateGenerator,
    FeatureBuilder,
    SEED,
    Snapshot,
    SnapshotBuilder,
    StaticData,
    add_training_positives,
    candidate_recall,
    content_key,
    exact_episodes,
    fit_ranker,
    labels_for_rows,
    map_at_10,
    rank_predictions,
)
from kapso_datasets.common import run_data_dir, save_predictions, shared_cache_dir


@dataclass
class TrainingBlock:
    checkpoint: pd.Timestamp
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    candidates: list[np.ndarray]
    labels: list[np.ndarray]
    users: np.ndarray
    warm: np.ndarray
    base_event_recall: float
    base_row_recall: float
    eval_x: np.ndarray
    eval_y: np.ndarray
    eval_groups: np.ndarray
    eval_candidates: list[np.ndarray]


def sample_queries(frame: pd.DataFrame, per_origin: int, seed: int, recency_weighted: bool = False) -> pd.DataFrame:
    selected = []
    for origin, rows in frame.groupby("timestamp", sort=True):
        year = int(pd.Timestamp(origin).year)
        if recency_weighted:
            scale = {2013: 0.5, 2014: 0.65, 2015: 0.8, 2016: 1.0, 2017: 1.5, 2018: 2.0}.get(year, 1.0)
        else:
            scale = 1.0
        count = min(int(round(per_origin * scale)), len(rows))
        rng = np.random.default_rng(seed + int(pd.Timestamp(origin).value // 10**9) % 1000003)
        indices = np.sort(rng.choice(len(rows), count, replace=False))
        selected.append(rows.iloc[indices])
    return pd.concat(selected, ignore_index=True)


def checkpoint_for(origin: pd.Timestamp, checkpoints: list[pd.Timestamp]) -> pd.Timestamp:
    eligible = [x for x in checkpoints if x <= pd.Timestamp(origin)]
    return eligible[-1]


def prepare_block(
    data: StaticData,
    config: BuildConfig,
    snapshot: Snapshot,
    frame: pd.DataFrame,
    generator: CandidateGenerator,
    features: FeatureBuilder,
) -> TrainingBlock:
    users = frame.user_id.to_numpy(np.int32)
    labels = [np.asarray(x, dtype=object) for x in frame.place_id]
    base_candidates, components = generator.generate(snapshot, users)
    event_recall, row_recall = candidate_recall(base_candidates, labels)
    eval_x, eval_row_users, eval_row_places = features.transform(snapshot, users, base_candidates, components)
    eval_y = labels_for_rows(eval_row_users, eval_row_places, users, labels)
    eval_groups = np.asarray([len(x) for x in base_candidates], dtype=np.int32)
    candidates = base_candidates
    x = eval_x
    y = eval_y
    groups = eval_groups
    warm = snapshot.user_place_count[users] > 0
    print(
        f"[training {snapshot.key}] queries={len(users)} rows={len(y)} positives={int(y.sum())} "
        f"recall_event={event_recall:.4f} recall_row={row_recall:.4f}"
    )
    return TrainingBlock(
        snapshot.cutoff, x, y, groups, candidates, labels, users, warm, event_recall,
        row_recall, eval_x, eval_y, eval_groups, base_candidates,
    )


def combine(blocks: list[TrainingBlock]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return np.vstack([x.x for x in blocks]), np.concatenate([x.y for x in blocks]), np.concatenate([x.groups for x in blocks])


def slice_map(predictions: np.ndarray, labels: list[np.ndarray], mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return map_at_10(predictions[mask], [labels[i] for i in np.flatnonzero(mask)])


def select_rounds(blocks: list[TrainingBlock], maximum: int, debug: bool) -> tuple[int, float, list[dict[str, float]]]:
    if debug or len(blocks) < 4:
        return maximum, 0.0, []
    reports: list[dict[str, float]] = []
    best_iterations = []
    blend_values = [0.0, 0.15, 0.3, 0.5]
    blend_scores = {x: [] for x in blend_values}
    blend_warm = {x: [] for x in blend_values}
    blend_cold = {x: [] for x in blend_values}
    years = sorted(set(int(x.checkpoint.year) for x in blocks))
    for valid_year in years[-3:]:
        train_blocks = [x for x in blocks if x.checkpoint.year < valid_year]
        valid_blocks = [x for x in blocks if x.checkpoint.year == valid_year]
        tx, ty, tg = combine(train_blocks)
        valid_x = np.vstack([x.eval_x for x in valid_blocks])
        valid_y = np.concatenate([x.eval_y for x in valid_blocks])
        valid_groups = np.concatenate([x.eval_groups for x in valid_blocks])
        valid_candidates = sum([x.eval_candidates for x in valid_blocks], [])
        valid_labels = sum([x.labels for x in valid_blocks], [])
        valid_warm = np.concatenate([x.warm for x in valid_blocks])
        valid_event_recall, valid_row_recall = candidate_recall(valid_candidates, valid_labels)
        model, best = fit_ranker(tx, ty, tg, maximum, (valid_x, valid_y, valid_groups))
        fold_blends = {}
        for blend in blend_values:
            blend_predictions = rank_predictions(model, valid_x, valid_candidates, blend)
            fold_blends[str(blend)] = map_at_10(blend_predictions, valid_labels)
            blend_scores[blend].append(fold_blends[str(blend)])
            blend_warm[blend].append(slice_map(blend_predictions, valid_labels, valid_warm))
            blend_cold[blend].append(slice_map(blend_predictions, valid_labels, ~valid_warm))
        predictions = rank_predictions(model, valid_x, valid_candidates)
        report = {
            "checkpoint": float(valid_year),
            "best_round": float(best),
            "map": map_at_10(predictions, valid_labels),
            "warm_map": slice_map(predictions, valid_labels, valid_warm),
            "cold_map": slice_map(predictions, valid_labels, ~valid_warm),
            "candidate_event_recall": valid_event_recall,
            "candidate_row_recall": valid_row_recall,
            "blend_map": fold_blends,
        }
        reports.append(report)
        best_iterations.append(best)
        print(f"[forward-fold] {json.dumps(report, sort_keys=True)}")
        del tx, ty, tg, model, predictions, valid_x, valid_y, valid_groups
    rounds = int(np.clip(np.median(best_iterations), 30, maximum))
    base_warm = float(np.mean(blend_warm[0.0]))
    base_cold = float(np.mean(blend_cold[0.0]))
    eligible = [
        x for x in blend_values
        if np.mean(blend_warm[x]) >= base_warm and np.mean(blend_cold[x]) >= base_cold
    ]
    blend = max(eligible, key=lambda x: (np.mean(blend_scores[x]), -x)) if eligible else 0.0
    summary = {str(x): float(np.mean(blend_scores[x])) for x in blend_values}
    print(
        f"[forward-fold] selected_rounds={rounds} blend={blend} blend_map={json.dumps(summary)} "
        f"restart=0.15 als_regularization=0.03"
    )
    return rounds, float(blend), reports


def prediction_features(
    snapshot: Snapshot,
    users: np.ndarray,
    generator: CandidateGenerator,
    features: FeatureBuilder,
) -> tuple[list[np.ndarray], dict, np.ndarray]:
    candidates, components = generator.generate(snapshot, users)
    x, _, _ = features.transform(snapshot, users, candidates, components)
    return candidates, components, x


def register_artifact(cache_root: Path, mode_dir: Path, key: str) -> None:
    path = cache_root / "artifacts.json"
    lock_path = cache_root / "artifacts.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        records = json.loads(path.read_text()) if path.exists() else []
        relative = str(mode_dir.relative_to(cache_root))
        if not any(x.get("path") == relative for x in records):
            records.append(
                {
                    "name": "lane2 sparse graph diffusion cache",
                    "path": relative,
                    "description": "Annual and exact CSR graphs, ALS factors, PPR candidate arrays, and scored predictions",
                    "content_key": key,
                    "rebuild_hint": "Run main.py with the matching fidelity; snapshots extend check-before-compute",
                }
            )
            path.write_text(json.dumps(records, indent=2))
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main() -> None:
    overall = time.time()
    debug = "--debug" in sys.argv
    config = BuildConfig.create(debug)
    cache_root = shared_cache_dir()
    data = StaticData(cache_root)
    builder = SnapshotBuilder(data, config)
    generator = CandidateGenerator(data, config)
    features = FeatureBuilder(data)
    train, val, test = data.load_tasks()
    if debug:
        checkpoints = [pd.Timestamp("2018-01-01")]
    else:
        annual = [pd.Timestamp(f"{year}-01-01") for year in range(2013, 2019)]
        exact_train = [pd.Timestamp(x) for x in sorted(train.timestamp.unique()) if pd.Timestamp(x) >= pd.Timestamp("2013-01-01")]
        checkpoints = sorted(set(annual + exact_train))
    train_pool = train[train.timestamp >= checkpoints[0]].copy()
    if debug:
        train_pool = train_pool[train_pool.timestamp >= pd.Timestamp("2018-01-01")]
    sampled = sample_queries(train_pool, config.train_per_origin, SEED, recency_weighted=not debug)
    sampled["checkpoint"] = [checkpoint_for(x, checkpoints) for x in sampled.timestamp]
    blocks: list[TrainingBlock] = []
    reusable: dict[str, Snapshot] = {}
    for checkpoint in checkpoints:
        snapshot = builder.build(checkpoint)
        current = sampled[sampled.checkpoint == checkpoint].drop(columns=["checkpoint"])
        if len(current):
            blocks.append(prepare_block(data, config, snapshot, current, generator, features))
        if debug:
            reusable[snapshot.key] = snapshot
        else:
            del snapshot
    rounds, blend, fold_reports = select_rounds(blocks, config.ranker_rounds, debug)
    ax, ay, ag = combine(blocks)
    model_a, _ = fit_ranker(ax, ay, ag, rounds)
    print(f"[model-a] queries={int(ag.size)} rows={len(ay)} rounds={rounds} elapsed={time.time() - overall:.1f}s")
    val_origin = pd.Timestamp(val.timestamp.iloc[0])
    if debug:
        val_snapshot = reusable[timestamp_key(checkpoints[0])]
    else:
        val_snapshot = builder.build(val_origin)
    val_users = val.user_id.to_numpy(np.int32)
    val_labels = [np.asarray(x, dtype=object) for x in val.place_id]
    val_candidates, val_components, val_x = prediction_features(val_snapshot, val_users, generator, features)
    val_predictions = rank_predictions(model_a, val_x, val_candidates, blend)
    val_event_recall, val_row_recall = candidate_recall(val_candidates, val_labels)
    val_map = map_at_10(val_predictions, val_labels)
    val_warm = val_snapshot.user_place_count[val_users] > 0
    diagnostics = {
        "mode": "debug" if debug else "full",
        "content_key": content_key(config),
        "ranker_rounds": rounds,
        "rank_blend": blend,
        "forward_folds": fold_reports,
        "validation_candidate_event_recall": val_event_recall,
        "validation_candidate_row_recall": val_row_recall,
        "validation_self_map": val_map,
        "validation_warm_count": int(val_warm.sum()),
        "validation_cold_count": int((~val_warm).sum()),
        "validation_warm_map": slice_map(val_predictions, val_labels, val_warm),
        "validation_cold_map": slice_map(val_predictions, val_labels, ~val_warm),
    }
    print(f"[model-a validation] {json.dumps(diagnostics, sort_keys=True)}")
    val_train_candidates = val_candidates
    val_train_x = val_x
    val_train_y = labels_for_rows(
        np.concatenate([np.full(len(x), int(u), dtype=np.int32) for u, x in zip(val_users, val_candidates)]),
        np.concatenate(val_candidates),
        val_users,
        val_labels,
    )
    val_block = TrainingBlock(
        val_origin,
        val_train_x,
        val_train_y,
        np.asarray([len(x) for x in val_train_candidates], dtype=np.int32),
        val_train_candidates,
        val_labels,
        val_users,
        val_warm,
        val_event_recall,
        val_row_recall,
        val_x,
        labels_for_rows(
            np.concatenate([np.full(len(x), int(u), dtype=np.int32) for u, x in zip(val_users, val_candidates)]),
            np.concatenate(val_candidates),
            val_users,
            val_labels,
        ),
        np.asarray([len(x) for x in val_candidates], dtype=np.int32),
        val_candidates,
    )
    extra_blocks: list[TrainingBlock] = []
    if not debug:
        origins = [pd.Timestamp("2018-12-01"), pd.Timestamp("2019-03-01"), pd.Timestamp("2019-06-01"), pd.Timestamp("2019-09-01")]
        episodes = exact_episodes(data, origins)
        episodes = sample_queries(episodes, config.train_per_origin, SEED + 91)
        episode_checkpoints = [val_origin] + origins
        episodes["checkpoint"] = [checkpoint_for(x, episode_checkpoints) for x in episodes.timestamp]
        for checkpoint in episode_checkpoints:
            current = episodes[episodes.checkpoint == checkpoint].drop(columns=["checkpoint"])
            if not len(current):
                continue
            episode_snapshot = val_snapshot if checkpoint == val_origin else builder.build(checkpoint)
            extra_blocks.append(prepare_block(data, config, episode_snapshot, current, generator, features))
            if checkpoint != val_origin:
                del episode_snapshot
    bx, by, bg = combine(blocks + [val_block] + extra_blocks)
    model_b, _ = fit_ranker(bx, by, bg, rounds)
    print(f"[model-b] queries={int(bg.size)} rows={len(by)} rounds={rounds} elapsed={time.time() - overall:.1f}s")
    if debug:
        test_snapshot = val_snapshot
    else:
        test_snapshot = builder.build(pd.Timestamp(test.timestamp.iloc[0]))
    test_users = test.user_id.to_numpy(np.int32)
    test_candidates, _, test_x = prediction_features(test_snapshot, test_users, generator, features)
    test_predictions = rank_predictions(model_b, test_x, test_candidates, blend)
    if val_predictions.shape != (547, 10) or test_predictions.shape != (351, 10):
        raise RuntimeError(f"prediction shape violation: {val_predictions.shape} {test_predictions.shape}")
    for name, values in (("val", val_predictions), ("test", test_predictions)):
        if np.any(values < 0) or np.any(values >= data.n_places):
            raise RuntimeError(f"{name} id range violation")
        if any(len(set(row.tolist())) != 10 for row in values):
            raise RuntimeError(f"{name} duplicate prediction violation")
    save_predictions(val_predictions.astype(np.int64), test_predictions.astype(np.int64))
    output = run_data_dir()
    diagnostics["elapsed_seconds"] = time.time() - overall
    (output / "metrics.json").write_text(json.dumps(diagnostics, indent=2))
    mode_dir = builder.cache
    np.save(mode_dir / "scored_val_predictions.npy", val_predictions.astype(np.int64))
    np.save(mode_dir / "scored_test_predictions.npy", test_predictions.astype(np.int64))
    register_artifact(cache_root, mode_dir, content_key(config))
    print(f"[complete] elapsed={time.time() - overall:.1f}s")


def timestamp_key(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


if __name__ == "__main__":
    main()
