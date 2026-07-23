#!/usr/bin/env python3
"""Prepare the run root for the Help_BOBAI local task.

Splits the 2473-sample train-dev blob into a DISJOINT train (agent builds on
it) + dev (agent self-checks on it) stratified by label, keeps the 700-sample
test split PRIVATE for post-campaign scoring, installs the statement +
evaluator, and writes task_meta.json the runner reads.

Usage:
    python -m benchmarks.ioai_tasks.data.prepare_bobai \
        --root ~/bobai_run --source-dir ~/bobai_src
"""

import argparse
import json
import os
import shutil

import torch

HERE = os.path.dirname(__file__)
STATEMENT = os.path.join(HERE, "bobai_statement.md")
EVALUATOR = os.path.join(HERE, "bobai_evaluate.py")

REQUIRED = ["train-dev_with_labels.pt", "base_classifier.pth",
            "eval_dataset.pt", "test_with_labels.pt"]


def stratified_split(dataset, dev_fraction: float, seed: int):
    """Disjoint train/dev split, per-class proportional. Deterministic."""
    labels = dataset[:, 0, -1].long()
    generator = torch.Generator().manual_seed(seed)
    train_idx, dev_idx = [], []
    for cls in labels.unique().tolist():
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)]
        n_dev = max(1, int(round(len(perm) * dev_fraction)))
        dev_idx.append(perm[:n_dev])
        train_idx.append(perm[n_dev:])
    train_idx = torch.cat(train_idx)
    dev_idx = torch.cat(dev_idx)
    return dataset[train_idx], dataset[dev_idx]


def prepare(root: str, source_dir: str, dev_fraction: float, seed: int) -> str:
    source_dir = os.path.abspath(source_dir)
    for name in REQUIRED:
        if not os.path.isfile(os.path.join(source_dir, name)):
            raise FileNotFoundError(f"{source_dir}/{name} missing")

    root = os.path.abspath(root)
    dataset_dir = os.path.join(root, "task", "dataset")
    private_dir = os.path.join(root, "private")
    for d in (dataset_dir, private_dir):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    train_dev = torch.load(os.path.join(source_dir, "train-dev_with_labels.pt"),
                           map_location="cpu", weights_only=False)
    train, dev = stratified_split(train_dev, dev_fraction, seed)
    torch.save(train, os.path.join(dataset_dir, "train_with_labels.pt"))
    torch.save(dev, os.path.join(dataset_dir, "dev_with_labels.pt"))
    shutil.copy2(os.path.join(source_dir, "base_classifier.pth"),
                 os.path.join(dataset_dir, "base_classifier.pth"))
    shutil.copy2(os.path.join(source_dir, "eval_dataset.pt"),
                 os.path.join(dataset_dir, "eval_dataset.pt"))
    shutil.copy2(STATEMENT, os.path.join(dataset_dir, "statement.md"))
    shutil.copy2(EVALUATOR, os.path.join(dataset_dir, "evaluate.py"))
    shutil.copy2(os.path.join(source_dir, "test_with_labels.pt"),
                 os.path.join(private_dir, "test_with_labels.pt"))

    self_check = (
        "python3 dataset/evaluate.py --data dataset/dev_with_labels.pt "
        "--data-dir dataset --solution submission/solution.py:Solution"
    )
    task_meta = {
        "task_name": "Help_BOBAI",
        "metric_name": "Macro-F1",
        "self_check_command": self_check,
        "final_eval_command": [
            os.path.join(dataset_dir, "evaluate.py"),
            "--data", os.path.join(private_dir, "test_with_labels.pt"),
            "--data-dir", dataset_dir,
            "--solution", "submission/solution.py:Solution",
        ],
    }
    with open(os.path.join(root, "task", "task_meta.json"), "w") as f:
        json.dump(task_meta, f, indent=2)

    print(f"  task_dir:  {os.path.join(root, 'task')}")
    print(f"  train:     {len(train)}  dev: {len(dev)}  private test: 700")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.root, args.source_dir, args.dev_fraction, args.seed)


if __name__ == "__main__":
    main()
