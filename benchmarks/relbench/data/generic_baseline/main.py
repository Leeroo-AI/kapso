"""Baseline candidate seeded into generic-search workspaces.

Purpose: (1) give the evaluation maintainer's registration-time calibration a
runnable candidate, (2) give parent_policy=best a legitimate starting parent.
Writes shape-correct, family-appropriate trivial predictions for both splits.
Replace me with a real solution.
"""

import os
import sys

import numpy as np

from relbench.base import RecommendationTask, TaskType
from relbench.tasks import get_task


def main() -> None:
    task = get_task(
        os.environ["RELBENCH_DATASET"], os.environ["RELBENCH_TASK"], download=False
    )
    n_val = len(task.get_table("val"))
    n_test = len(task.get_table("test"))

    if isinstance(task, RecommendationTask):
        k = task.eval_k
        row = np.arange(k, dtype=np.int64)
        val_pred = np.tile(row, (n_val, 1))
        test_pred = np.tile(row, (n_test, 1))
    elif task.task_type == TaskType.BINARY_CLASSIFICATION:
        val_pred = np.full(n_val, 0.5)
        test_pred = np.full(n_test, 0.5)
    elif task.task_type == TaskType.MULTICLASS_CLASSIFICATION:
        classes = int(getattr(task, "num_classes"))
        val_pred = np.full((n_val, classes), 1.0 / classes)
        test_pred = np.full((n_test, classes), 1.0 / classes)
    else:
        train = task.get_table("train", mask_input_cols=False)
        center = float(train.df[task.target_col].median())
        val_pred = np.full(n_val, center)
        test_pred = np.full(n_test, center)

    out = os.environ["KAPSO_RUN_DATA_DIR"]
    np.save(os.path.join(out, "val_predictions.npy"), val_pred)
    np.save(os.path.join(out, "test_predictions.npy"), test_pred)
    print(
        f"[baseline] wrote trivial predictions val{val_pred.shape} test{test_pred.shape} "
        f"(debug={'--debug' in sys.argv})"
    )


if __name__ == "__main__":
    main()
