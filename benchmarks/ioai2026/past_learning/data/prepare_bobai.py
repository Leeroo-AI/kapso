#!/usr/bin/env python3
"""Prepare the run root for Help_BOBAI from a contest/ + gold/ split.

Consumes the split produced by acquire_bobai.py:
  contest/  → the agent's dataset (train split + honest dev split, base model,
              statement, evaluator). NEVER contains answers.
  gold/     → sequestered: the test-with-labels becomes the PRIVATE scoring
              split (outside task/), and the reference solution is copied to
              <root>/gold/ for the aggregate step (never under task/).

    python -m benchmarks.ioai2026.past_learning.data.prepare_bobai \
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


def stratified_split(dataset, dev_fraction: float, seed: int):
    labels = dataset[:, 0, -1].long()
    generator = torch.Generator().manual_seed(seed)
    train_idx, dev_idx = [], []
    for cls in labels.unique().tolist():
        cls_idx = (labels == cls).nonzero(as_tuple=True)[0]
        perm = cls_idx[torch.randperm(len(cls_idx), generator=generator)]
        n_dev = max(1, int(round(len(perm) * dev_fraction)))
        dev_idx.append(perm[:n_dev])
        train_idx.append(perm[n_dev:])
    return dataset[torch.cat(train_idx)], dataset[torch.cat(dev_idx)]


def prepare(root: str, source_dir: str, dev_fraction: float, seed: int) -> str:
    source_dir = os.path.abspath(source_dir)
    contest = os.path.join(source_dir, "contest")
    gold = os.path.join(source_dir, "gold")
    for d in (contest, gold):
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{d} missing — run acquire_bobai.py first (needs contest/+gold/)"
            )
    for name in ("train-dev_dataset_with_labels.pt", "base_classifier.pth",
                 "eval_dataset.pt"):
        if not os.path.isfile(os.path.join(contest, name)):
            raise FileNotFoundError(f"contest/{name} missing")
    if not os.path.isfile(os.path.join(gold, "test_dataset_with_labels.pt")):
        raise FileNotFoundError("gold/test_dataset_with_labels.pt missing")

    root = os.path.abspath(root)
    dataset_dir = os.path.join(root, "task", "dataset")
    private_dir = os.path.join(root, "private")
    gold_out = os.path.join(root, "gold")
    for d in (dataset_dir, private_dir, gold_out):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # contest/ → agent-visible dataset (train/dev split + base model + eval)
    train_dev = torch.load(
        os.path.join(contest, "train-dev_dataset_with_labels.pt"),
        map_location="cpu", weights_only=False)
    train, dev = stratified_split(train_dev, dev_fraction, seed)
    torch.save(train, os.path.join(dataset_dir, "train_with_labels.pt"))
    torch.save(dev, os.path.join(dataset_dir, "dev_with_labels.pt"))
    shutil.copy2(os.path.join(contest, "base_classifier.pth"),
                 os.path.join(dataset_dir, "base_classifier.pth"))
    shutil.copy2(os.path.join(contest, "eval_dataset.pt"),
                 os.path.join(dataset_dir, "eval_dataset.pt"))
    shutil.copy2(STATEMENT, os.path.join(dataset_dir, "statement.md"))
    shutil.copy2(EVALUATOR, os.path.join(dataset_dir, "evaluate.py"))

    # gold/ → PRIVATE scoring split + reference solution (never under task/)
    shutil.copy2(os.path.join(gold, "test_dataset_with_labels.pt"),
                 os.path.join(private_dir, "test_with_labels.pt"))
    reference_solution = None
    ref_src = os.path.join(gold, "reference_solution.ipynb")
    if os.path.isfile(ref_src):
        reference_solution = os.path.join(gold_out, "reference_solution.ipynb")
        shutil.copy2(ref_src, reference_solution)

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
        "reference_solution": reference_solution,
    }
    with open(os.path.join(root, "task", "task_meta.json"), "w") as f:
        json.dump(task_meta, f, indent=2)

    print(f"  task_dir:  {os.path.join(root, 'task')}")
    print(f"  train: {len(train)}  dev: {len(dev)}  private test: 700")
    print(f"  gold reference solution: {reference_solution}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-dir", required=True,
                        help="dir with contest/ + gold/ from acquire_bobai.py")
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(args.root, args.source_dir, args.dev_fraction, args.seed)


if __name__ == "__main__":
    main()
