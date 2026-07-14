"""Kapso RelBench starter-kit helpers.

Import these from your main.py to load the task correctly (sanitized cache,
download=False) and to save predictions exactly per the evaluation contract.

Typical skeleton:

    from kapso_datasets.common import load_task, save_predictions, is_debug, shared_cache_dir

    ctx = load_task()                     # -> TaskContext
    train, val, test = ctx.train, ctx.val, ctx.test
    db = ctx.db
    ...train on train (labels), tune on val (labels), predict val+test...
    save_predictions(val_pred, test_pred)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def is_debug() -> bool:
    return "--debug" in sys.argv


def run_data_dir() -> Path:
    d = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./kapso_output"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def shared_cache_dir() -> Path:
    d = Path(os.environ.get("KAPSO_SHARED_CACHE_DIR", "./shared_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class TaskContext:
    dataset_name: str
    task_name: str
    dataset: Any
    task: Any
    db: Any
    train: Any  # relbench Table (labels included)
    val: Any    # relbench Table (labels included; tuning allowed)
    test: Any   # relbench Table (seed ids + timestamps only)

    @property
    def target_col(self) -> str:
        return self.task.target_col


def load_task(upto_test_timestamp: bool = True) -> TaskContext:
    """Load dataset/task from the sanitized cache. Never pass download=True.

    For autocomplete tasks pass upto_test_timestamp=False when you need the
    post-test-cutoff entity rows as *inputs* (their targets are blanked).
    """
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    dataset = get_dataset(dataset_name, download=False)
    task = get_task(dataset_name, task_name, download=False)
    db = dataset.get_db(upto_test_timestamp=upto_test_timestamp)
    return TaskContext(
        dataset_name=dataset_name,
        task_name=task_name,
        dataset=dataset,
        task=task,
        db=db,
        train=task.get_table("train"),
        val=task.get_table("val"),
        test=task.get_table("test"),
    )


def save_predictions(val_pred: np.ndarray, test_pred: np.ndarray) -> None:
    """Write both prediction files. Call this in EVERY run mode, debug included."""
    out = run_data_dir()
    np.save(out / "val_predictions.npy", np.asarray(val_pred))
    np.save(out / "test_predictions.npy", np.asarray(test_pred))
    print(
        f"[starter_kit] saved predictions: val{np.asarray(val_pred).shape} "
        f"test{np.asarray(test_pred).shape} -> {out}"
    )


def assert_temporal_safety(df, time_col: str, seed_time) -> None:
    """Guard: every row used for features must satisfy time <= seed_time."""
    bad = (df[time_col] > seed_time).sum()
    if bad:
        raise RuntimeError(
            f"TEMPORAL LEAK: {bad} rows have {time_col} after the seed time {seed_time}"
        )


def eval_map_at_k(pred: np.ndarray, true_lists, k: int) -> float:
    """Fast MAP@K identical to relbench.metrics.link_prediction_map."""
    maps = []
    for row_pred, true in zip(pred, true_lists):
        true = set(true)
        if not true:
            continue
        hits = np.array([1 if p in true else 0 for p in row_pred[:k]])
        if hits.sum() == 0:
            maps.append(0.0)
            continue
        precision_at = np.cumsum(hits) / (np.arange(k) + 1)
        maps.append(float((precision_at * hits).sum() / min(len(true), k)))
    return float(np.mean(maps)) if maps else 0.0
