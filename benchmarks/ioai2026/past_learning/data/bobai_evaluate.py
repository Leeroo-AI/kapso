#!/usr/bin/env python3
"""Help_BOBAI evaluator — macro-F1 over the 7-way classification.

Loads a submission's `Solution` class, runs it on a labeled split, and
reports macro-F1 (the task metric) plus old-vs-new-class breakdown. Fail
loud: a wrong shape or missing method raises, it is not worked around.

Submission contract (submission/solution.py):
    class Solution:
        def __init__(self, data_dir: str): ...   # visible dataset dir:
            #   train-dev_with_labels.pt (2473,1,769), base_classifier.pth
        def predict(self, X) -> list[int]: ...    # X: tensor [N,1,768],
            #   returns N predictions in 0..6
"""

import argparse
import importlib.util
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

OLD_CLASSES = (0, 1, 2, 3, 4)
NEW_CLASSES = (5, 6)


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
    parser.add_argument("--data", required=True, help="labeled .pt [N,1,769]")
    parser.add_argument("--data-dir", required=True, help="visible dataset dir")
    parser.add_argument("--solution", required=True, help="path.py:ClassName")
    args = parser.parse_args()

    dataset = torch.load(args.data, map_location="cpu", weights_only=False)
    if not (torch.is_tensor(dataset) and dataset.dim() == 3
            and dataset.shape[-1] == 769):
        raise ValueError(f"labeled data must be [N,1,769] tensor, "
                         f"got {type(dataset)} {getattr(dataset, 'shape', None)}")
    inputs = dataset[:, :, :-1]
    labels = dataset[:, :, -1].flatten().long().numpy()

    Solution = load_solution_class(args.solution)
    solution = Solution(args.data_dir)
    predictions = np.asarray(solution.predict(inputs)).flatten()
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"got {predictions.shape[0]} predictions for "
                         f"{labels.shape[0]} samples")
    if not np.isin(predictions, range(7)).all():
        raise ValueError("predictions must all be in 0..6")

    macro_f1 = f1_score(labels, predictions, average="macro")
    old_mask = np.isin(labels, OLD_CLASSES)
    new_mask = np.isin(labels, NEW_CLASSES)
    acc_old = (predictions[old_mask] == labels[old_mask]).mean()
    acc_new = (predictions[new_mask] == labels[new_mask]).mean()

    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Samples: {len(labels)} | acc_old(0-4): {acc_old:.4f} "
          f"| acc_new(5-6): {acc_new:.4f}")


if __name__ == "__main__":
    main()
