"""Growth-pattern classifier: train on the labelled tiles, predict a
probability for each of the four patterns on every holdout tile.

Contract (the evaluator calls exactly this):

    python predict.py --data kapso_datasets --out predictions.csv

reads <data>/train.csv (with the `pattern` label) and <data>/holdout.csv
(features only) and writes <out> with five columns, `tile_id` and one
probability per pattern (lepidic, acinar, papillary, solid), one row per
holdout tile. The baseline is a logistic regression on the standardised
tile embeddings. It also prints a validation score from a random split of
the training tiles.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL = "pattern"
ID = "tile_id"
PATTERNS = ["lepidic", "acinar", "papillary", "solid"]


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000)),
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="kapso_datasets")
    parser.add_argument("--out", default="predictions.csv")
    args = parser.parse_args()
    data = Path(args.data)

    train = pd.read_csv(data / "train.csv")
    holdout = pd.read_csv(data / "holdout.csv")
    labels = train.pop(LABEL)
    features = train.drop(columns=[ID, "patient_id"])

    fit_rows, check_rows, fit_labels, check_labels = train_test_split(
        features, labels, test_size=0.2, stratify=labels, random_state=0)
    check = build_pipeline().fit(fit_rows, fit_labels)
    validation = roc_auc_score(check_labels, check.predict_proba(check_rows), labels=check.classes_, multi_class="ovr")
    print(f"validation macro_auc: {validation:.4f}")

    pipeline = build_pipeline().fit(features, labels)
    probabilities = pd.DataFrame(pipeline.predict_proba(holdout.drop(columns=[ID, "patient_id"])), columns=pipeline.classes_)
    predictions = pd.concat([holdout[[ID]], probabilities[PATTERNS]], axis=1)
    predictions.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(holdout)} rows")


if __name__ == "__main__":
    main()
