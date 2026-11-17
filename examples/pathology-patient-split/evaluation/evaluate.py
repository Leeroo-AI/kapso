"""The evaluation Kapso protects for the pathology fixture.

    python kapso_evaluation/evaluate.py [--fast] [--timeout SECONDS]

Run from the workspace root. It runs the repository's prediction contract,
`python predict.py --data kapso_datasets --out predictions.csv`, validates
the file, scores it with macro one-vs-rest ROC AUC over the four growth
patterns against the labels under the private directory named in
`private_dir.txt` beside this script, writes `result.json` beside this
script in the shape the engine reads, and prints the metric as the last
line:

    macro_auc: 0.8712

Higher is better. Every holdout tile belongs to a patient absent from the
training data. Exit codes: 0 scored; 2 the predictions are missing or
malformed (the reason is printed); 3 `predict.py` failed (its last lines
are printed). A timeout raises. `--fast` scores the first 8 holdout
patients only: a smoke check whose number is not comparable with a full
score.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ID = "tile_id"
LABEL = "pattern"
PATTERNS = ["lepidic", "acinar", "papillary", "solid"]
FAST_PATIENTS = 8
PREDICT_COMMAND = [sys.executable, "predict.py", "--data", "kapso_datasets", "--out", "predictions.csv"]


def fail(code: int, reason: str) -> None:
    print(f"evaluation failed: {reason}")
    sys.exit(code)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", default=".", help="the workspace root; predict.py and kapso_datasets/ live there")
    parser.add_argument("--fast", action="store_true", help=f"score the first {FAST_PATIENTS} holdout patients only")
    parser.add_argument("--timeout", type=float, default=600.0, help="seconds predict.py may take")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    here = Path(__file__).resolve().parent
    private_dir = Path((here / "private_dir.txt").read_text(encoding="utf-8").strip())
    labels = pd.read_csv(private_dir / "holdout_labels.csv")
    holdout_ids = set(labels[ID])
    if args.fast:
        patients = labels[ID].str.split("-").str[0]
        labels = labels[patients.isin(patients.unique()[:FAST_PATIENTS])]

    predictions_path = workspace / "predictions.csv"
    for stale in (predictions_path, here / "result.json"):
        if stale.exists():
            stale.unlink()   # never score or report a stale file

    started = time.monotonic()
    run = subprocess.run(PREDICT_COMMAND, cwd=workspace, capture_output=True, text=True, timeout=args.timeout)
    if run.returncode != 0:
        print("\n".join((run.stdout + run.stderr).splitlines()[-20:]))
        fail(3, f"predict.py exited {run.returncode}")
    if not predictions_path.is_file():
        fail(2, "predict.py wrote no predictions.csv")

    predictions = pd.read_csv(predictions_path)
    if list(predictions.columns) != [ID, *PATTERNS]:
        fail(2, f"predictions.csv columns are {list(predictions.columns)}, expected {[ID, *PATTERNS]}")
    if predictions[ID].duplicated().any():
        fail(2, "predictions.csv repeats a tile_id")
    if not predictions[ID].isin(holdout_ids).all():
        fail(2, "predictions.csv has a tile_id that is not a holdout tile")
    if not all(pd.api.types.is_numeric_dtype(predictions[name]) for name in PATTERNS):
        fail(2, "every pattern probability must be a number")
    values = predictions[PATTERNS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        fail(2, "every pattern probability must be a finite number in [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0):   # the tolerance roc_auc_score itself applies
        fail(2, "each row's four probabilities must sum to 1")
    merged = labels.merge(predictions, on=ID, how="left")
    if merged[PATTERNS].isna().any(axis=None):
        fail(2, f"{int(merged[PATTERNS[0]].isna().sum())} holdout tiles have no prediction")

    ordered = sorted(PATTERNS)   # roc_auc_score takes the class labels in sorted order
    auc = float(roc_auc_score(merged[LABEL], merged[ordered], labels=ordered, multi_class="ovr", average="macro"))
    seconds = time.monotonic() - started
    summary = f"macro_auc: {auc:.4f} (tiles={len(merged)}, fast={args.fast}, seconds={seconds:.1f})"
    result = {
        "evaluation_script_path": "kapso_evaluation/evaluate.py",
        "evaluation_output": summary,
        "score": round(auc, 4),
    }
    (here / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"tiles: {len(merged)}  seconds: {seconds:.1f}  fast: {args.fast}")
    print(f"macro_auc: {auc:.4f}")


if __name__ == "__main__":
    main()
