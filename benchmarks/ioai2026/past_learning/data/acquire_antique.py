#!/usr/bin/env python3
"""Acquire Antique into contest/ (agent-visible) vs gold/ (answers).

    python -m benchmarks.ioai2026.past_learning.data.acquire_antique --out ~/antique_src
"""

import argparse
import os
import urllib.request

BASE = ("https://raw.githubusercontent.com/IOAI-official/IOAI-2025/main/"
        "Individual-Contest/Antique/")

CONTEST = {
    "training_set.csv": "training_set/training_set.csv",
    "validation_set.csv": "Solution/validation_set/validation_set.csv",
    "test_set.csv": "Solution/test_set/test_set.csv",
}
GOLD = {
    "reference_solution.ipynb": "Solution/Antique_Solution.ipynb",
    "label.csv": "Scoring/label.csv",  # columns: validation_label, testing_label
}


def _download(rel: str, dest: str) -> None:
    urllib.request.urlretrieve(BASE + rel, dest)
    if os.path.getsize(dest) == 0:
        raise RuntimeError(f"downloaded {rel} is empty")


def acquire(out_dir: str) -> str:
    out_dir = os.path.abspath(out_dir)
    contest = os.path.join(out_dir, "contest")
    gold = os.path.join(out_dir, "gold")
    os.makedirs(contest, exist_ok=True)
    os.makedirs(gold, exist_ok=True)
    for name, rel in CONTEST.items():
        _download(rel, os.path.join(contest, name))
    for name, rel in GOLD.items():
        _download(rel, os.path.join(gold, name))
    print(f"  contest/: {sorted(CONTEST)}")
    print(f"  gold/:    {sorted(GOLD)}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    acquire(args.out)


if __name__ == "__main__":
    main()
