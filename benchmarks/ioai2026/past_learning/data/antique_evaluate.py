#!/usr/bin/env python3
"""Antique evaluator — accuracy of ±1 predictions vs gold labels. Fail loud.

Submission contract (submission/solution.py):
    class Solution:
        def __init__(self, data_dir: str): ...   # dir with training_set.csv
        def predict(self, X) -> list[int]: ...    # X: [N,5] floats -> N in {-1,1}
"""

import argparse
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

FEATURES = ["feature1", "feature2", "feature3", "feature4", "feature5"]


def load_solution_class(spec: str):
    path, _, cls = spec.partition(":")
    if not os.path.isfile(path) or not cls:
        raise ValueError(f"--solution must be path.py:ClassName, got {spec!r}")
    module_spec = importlib.util.spec_from_file_location("submission_solution", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    if not hasattr(module, cls):
        raise AttributeError(f"{path} has no class {cls!r}")
    return getattr(module, cls)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True,
                        help="csv with feature1..5 + a 'label' column")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--solution", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if not set(FEATURES) <= set(df.columns) or "label" not in df.columns:
        raise ValueError(f"data must have {FEATURES} + 'label'; got {list(df.columns)}")
    X = df[FEATURES].to_numpy()
    y = df["label"].to_numpy()

    Solution = load_solution_class(args.solution)
    predictions = np.asarray(Solution(args.data_dir).predict(X)).flatten()
    if predictions.shape[0] != y.shape[0]:
        raise ValueError(f"{predictions.shape[0]} preds for {y.shape[0]} samples")
    if not np.isin(predictions, (-1, 1)).all():
        raise ValueError("predictions must all be -1 or 1")

    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")
    print(f"Samples: {len(y)}")


if __name__ == "__main__":
    main()
