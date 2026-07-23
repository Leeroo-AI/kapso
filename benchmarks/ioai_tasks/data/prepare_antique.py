#!/usr/bin/env python3
"""Prepare the Antique run root from a contest/ + gold/ split.

Agent-visible dataset = training_set (semi-supervised) + validation_with_labels
(the labeled validation split, the contest's public board). Private scoring =
test_set + gold testing_label. The reference solution is stashed under
<root>/gold/ for the aggregate step; never under task/.

    python -m benchmarks.ioai_tasks.data.prepare_antique \
        --root ~/antique_run --source-dir ~/antique_src
"""

import argparse
import json
import os
import shutil

import pandas as pd

HERE = os.path.dirname(__file__)
STATEMENT = os.path.join(HERE, "antique_statement.md")
EVALUATOR = os.path.join(HERE, "antique_evaluate.py")


def prepare(root: str, source_dir: str) -> str:
    source_dir = os.path.abspath(source_dir)
    contest = os.path.join(source_dir, "contest")
    gold = os.path.join(source_dir, "gold")
    for name in ("training_set.csv", "validation_set.csv", "test_set.csv"):
        if not os.path.isfile(os.path.join(contest, name)):
            raise FileNotFoundError(f"contest/{name} missing (run acquire first)")
    if not os.path.isfile(os.path.join(gold, "label.csv")):
        raise FileNotFoundError("gold/label.csv missing")

    root = os.path.abspath(root)
    dataset_dir = os.path.join(root, "task", "dataset")
    private_dir = os.path.join(root, "private")
    gold_out = os.path.join(root, "gold")
    for d in (dataset_dir, private_dir, gold_out):
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    labels = pd.read_csv(os.path.join(gold, "label.csv"))  # validation_label, testing_label

    # contest/ → agent-visible: semi-supervised training + LABELED validation dev
    shutil.copy2(os.path.join(contest, "training_set.csv"),
                 os.path.join(dataset_dir, "training_set.csv"))
    val = pd.read_csv(os.path.join(contest, "validation_set.csv"))
    val["label"] = labels["validation_label"].to_numpy()
    val.to_csv(os.path.join(dataset_dir, "validation_with_labels.csv"), index=False)
    shutil.copy2(STATEMENT, os.path.join(dataset_dir, "statement.md"))
    shutil.copy2(EVALUATOR, os.path.join(dataset_dir, "evaluate.py"))

    # gold/ → PRIVATE test (features + gold testing_label) + reference solution
    test = pd.read_csv(os.path.join(contest, "test_set.csv"))
    test["label"] = labels["testing_label"].to_numpy()
    test.to_csv(os.path.join(private_dir, "test_with_labels.csv"), index=False)
    reference_solution = os.path.join(gold_out, "reference_solution.ipynb")
    shutil.copy2(os.path.join(gold, "reference_solution.ipynb"), reference_solution)

    self_check = (
        "python3 dataset/evaluate.py --data dataset/validation_with_labels.csv "
        "--data-dir dataset --solution submission/solution.py:Solution"
    )
    task_meta = {
        "task_name": "Antique",
        "metric_name": "Accuracy",
        "self_check_command": self_check,
        "final_eval_command": [
            os.path.join(dataset_dir, "evaluate.py"),
            "--data", os.path.join(private_dir, "test_with_labels.csv"),
            "--data-dir", dataset_dir,
            "--solution", "submission/solution.py:Solution",
        ],
        "reference_solution": reference_solution,
    }
    with open(os.path.join(root, "task", "task_meta.json"), "w") as f:
        json.dump(task_meta, f, indent=2)

    print(f"  task_dir: {os.path.join(root, 'task')}")
    print(f"  train 500 (4 labeled), val-dev 500 labeled, private test 500")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    prepare(args.root, args.source_dir)


if __name__ == "__main__":
    main()
