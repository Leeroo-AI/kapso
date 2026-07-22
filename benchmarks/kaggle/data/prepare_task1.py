#!/usr/bin/env python3
"""Prepare the run root for IOAI AI Models Track Practice Task 1.

Arranges the extracted Kaggle download (audio/, model/, three CSVs) into the
kapso run-root layout and installs the curated statement. Unlike the Animal
Deduction preparer there is no private split to sequester: the evaluation
rows are unlabeled and the ground truth lives on the Kaggle leaderboard.

Usage:
    python -m benchmarks.kaggle.data.prepare_task1 \
        --root ~/kaggle_run --source-dir ~/kaggle_task1_src \
        --competition ioai-2026-ai-models-track-practice-task-1
"""

import argparse
import csv
import json
import os
import shutil

STATEMENT_PATH = os.path.join(os.path.dirname(__file__), "task1_statement.md")

REQUIRED_ENTRIES = ["audio", "model", "train.csv", "fine_tune.csv",
                    "submission.csv"]
MODEL_FILES = ["config.json", "model.safetensors", "preprocessor_config.json"]


def prepare(root: str, source_dir: str, competition: str) -> str:
    source_dir = os.path.abspath(source_dir)
    for entry in REQUIRED_ENTRIES:
        if not os.path.exists(os.path.join(source_dir, entry)):
            raise FileNotFoundError(
                f"{source_dir}/{entry} missing — point --source-dir at the "
                "extracted competition download"
            )
    for name in MODEL_FILES:
        if not os.path.isfile(os.path.join(source_dir, "model", name)):
            raise FileNotFoundError(f"{source_dir}/model/{name} missing")

    root = os.path.abspath(root)
    dataset_dir = os.path.join(root, "task", "dataset")
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    for entry in REQUIRED_ENTRIES:
        src = os.path.join(source_dir, entry)
        dst = os.path.join(dataset_dir, entry)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)

    shutil.copy2(STATEMENT_PATH, os.path.join(dataset_dir, "statement.md"))

    with open(os.path.join(root, "task", "kaggle.json"), "w") as f:
        json.dump({"competition": competition}, f, indent=2)

    labeled = 0
    for name in ("train.csv", "fine_tune.csv"):
        with open(os.path.join(dataset_dir, name)) as f:
            labeled += sum(1 for _ in csv.DictReader(f))
    audio_count = len(os.listdir(os.path.join(dataset_dir, "audio")))
    print(f"  task_dir:      {os.path.join(root, 'task')}")
    print(f"  labeled rows:  {labeled}")
    print(f"  audio files:   {audio_count}")
    print(f"  competition:   {competition}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-dir", required=True,
                        help="Extracted competition download")
    parser.add_argument("--competition",
                        default="ioai-2026-ai-models-track-practice-task-1")
    args = parser.parse_args()
    prepare(args.root, args.source_dir, args.competition)


if __name__ == "__main__":
    main()
