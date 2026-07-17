"""Self-check prediction files against the evaluation contract (read-only).

Run after writing predictions:  python kapso_evaluation/check_predictions.py
Exits non-zero with a specific message on the first violation.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def fail(msg: str) -> None:
    print(f"[check_predictions] FAIL: {msg}")
    sys.exit(1)


def main() -> None:
    from relbench.base import AutoCompleteTask, RecommendationTask, TaskType
    from relbench.tasks import get_task

    dataset_name = os.environ["RELBENCH_DATASET"]
    task_name = os.environ["RELBENCH_TASK"]
    out = Path(os.environ.get("KAPSO_RUN_DATA_DIR", "./kapso_output"))
    task = get_task(dataset_name, task_name, download=False)

    for split in ("val", "test"):
        n = len(task.get_table(split))
        path = out / f"{split}_predictions.npy"
        if not path.exists():
            fail(f"{path} missing")
        arr = np.load(path, allow_pickle=False)

        if isinstance(task, RecommendationTask):
            if arr.shape != (n, task.eval_k):
                fail(f"{split}: shape {arr.shape} != {(n, task.eval_k)}")
            if not np.issubdtype(arr.dtype, np.integer):
                fail(f"{split}: dtype {arr.dtype} is not integer")
            if arr.min() < 0 or arr.max() >= task.num_dst_nodes:
                print(f"[check_predictions] WARN {split}: ids outside [0,{task.num_dst_nodes}) never match")
            dup = sum(len(np.unique(r)) < len(r) for r in arr)
            if dup:
                print(f"[check_predictions] WARN {split}: {dup} rows contain duplicate ids")
        else:
            expected: tuple
            if task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
                c = getattr(task, "num_classes", None)
                expected = (n, c) if c else (n,)
            else:
                expected = (n,)
            if arr.shape != expected:
                fail(f"{split}: shape {arr.shape} != {expected}")
            if not np.all(np.isfinite(arr.astype(np.float64))):
                fail(f"{split}: non-finite values")
            if task.task_type == TaskType.BINARY_CLASSIFICATION and (
                arr.min() < 0 or arr.max() > 1
            ):
                fail(f"{split}: binary predictions outside [0,1] — apply a sigmoid")
        print(f"[check_predictions] OK {split}: shape {arr.shape}")

    print("[check_predictions] all checks passed")


if __name__ == "__main__":
    main()
