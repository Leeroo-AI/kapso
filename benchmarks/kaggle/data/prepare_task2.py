#!/usr/bin/env python3
"""Prepare the run root for IOAI AI Models Track Practice Task 2.

Arranges the extracted Kaggle download (three pickles) into the kapso
run-root layout and installs the curated statement. No private split: the
evaluation scenarios ship unlabeled and the ground truth lives on the
Kaggle leaderboard.

Usage:
    python -m benchmarks.kaggle.data.prepare_task2 \
        --root ~/kaggle_run --source-dir ~/kaggle_task2_src
"""

import argparse
import json
import os
import pickle
import shutil

STATEMENT_PATH = os.path.join(os.path.dirname(__file__), "task2_statement.md")

REQUIRED_ENTRIES = ["train_demos.pkl", "valid_scenarios.pkl",
                    "test_scenarios.pkl"]


def prepare(root: str, source_dir: str, competition: str) -> str:
    source_dir = os.path.abspath(source_dir)
    for entry in REQUIRED_ENTRIES:
        if not os.path.isfile(os.path.join(source_dir, entry)):
            raise FileNotFoundError(
                f"{source_dir}/{entry} missing — point --source-dir at the "
                "extracted competition download"
            )

    root = os.path.abspath(root)
    dataset_dir = os.path.join(root, "task", "dataset")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    for entry in REQUIRED_ENTRIES:
        shutil.copy2(os.path.join(source_dir, entry),
                     os.path.join(dataset_dir, entry))

    shutil.copy2(STATEMENT_PATH, os.path.join(dataset_dir, "statement.md"))

    with open(os.path.join(root, "task", "kaggle.json"), "w") as f:
        json.dump({"competition": competition}, f, indent=2)

    with open(os.path.join(dataset_dir, "train_demos.pkl"), "rb") as f:
        demo_count = len(pickle.load(f)["trajectories"])
    with open(os.path.join(dataset_dir, "test_scenarios.pkl"), "rb") as f:
        test_count = len(pickle.load(f))
    print(f"  task_dir:        {os.path.join(root, 'task')}")
    print(f"  train demos:     {demo_count}")
    print(f"  test scenarios:  {test_count}")
    print(f"  competition:     {competition}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-dir", required=True,
                        help="Extracted competition download")
    parser.add_argument("--competition",
                        default="ioai-2026-ai-models-track-practice-task-2")
    args = parser.parse_args()
    prepare(args.root, args.source_dir, args.competition)


if __name__ == "__main__":
    main()
