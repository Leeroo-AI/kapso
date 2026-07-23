#!/usr/bin/env python3
"""Acquire Help_BOBAI and split it into contest/ (agent-visible) vs gold/
(answers — reference solution, test labels, any precomputed predictions).

Past-contest repos ship the answer key next to the task. A harvest run must
NEVER let the agent see gold/ — neither on disk nor (separately) via web
search. This split makes the boundary explicit: prepare_bobai builds the run
root from contest/ only, uses gold/ labels for PRIVATE scoring, and stashes
gold/ the reference solution for the aggregate step.

    python -m benchmarks.ioai_tasks.data.acquire_bobai --out ~/bobai_src
"""

import argparse
import os
import urllib.request

BASE = ("https://raw.githubusercontent.com/IOAI-official/IOAI-2024/main/"
        "On-Site-Round/Help_BOBAI/")

# What a contestant is given — safe for the agent to see.
CONTEST = {
    "train-dev_dataset_with_labels.pt": "training_set/train-dev_dataset_with_labels.pt",
    "base_classifier.pth": "training_set/base_classifier.pth",
    "eval_dataset.pt": "Solution/validation_set/eval_dataset.pt",
}
# Answers — MUST be sequestered from the agent (used only for private scoring
# + fed to the aggregate step as the gold reference).
GOLD = {
    "reference_solution.ipynb": "Solution/Help_BOBAI_Solution.ipynb",
    "test_dataset_with_labels.pt": "Solution/test_set/test_dataset_with_labels.pt",
    "test_labels.txt": "Solution/test_set/test_labels.txt",
}


def _download(rel: str, dest: str) -> None:
    urllib.request.urlretrieve(BASE + rel, dest)
    if os.path.getsize(dest) == 0:
        raise RuntimeError(f"downloaded {rel} is empty")


def acquire(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    contest_dir = os.path.join(out_dir, "contest")
    gold_dir = os.path.join(out_dir, "gold")
    os.makedirs(contest_dir, exist_ok=True)
    os.makedirs(gold_dir, exist_ok=True)
    for name, rel in CONTEST.items():
        _download(rel, os.path.join(contest_dir, name))
    for name, rel in GOLD.items():
        _download(rel, os.path.join(gold_dir, name))
    print(f"  contest/ (agent-visible): {sorted(CONTEST)}")
    print(f"  gold/    (sequestered):   {sorted(GOLD)}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    acquire(args.out)


if __name__ == "__main__":
    main()
