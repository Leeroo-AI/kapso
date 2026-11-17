"""Record the fixture's baseline: the untouched repository scored by the
evaluator in a staged workspace, repeated, with a standard error of the
macro AUC from a bootstrap over patients (tiles of one patient move
together, so resampling tiles would understate the noise).

    python baseline.py [--repeats 3] [--bootstrap 1000]

Writes baseline.json beside this script: the score per repeat, the
bootstrap standard error, the evaluator's wall time, the platform, the
library versions and the data hashes from MANIFEST.json.
"""

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
ID = "tile_id"
LABEL = "pattern"
PATTERNS = ["acinar", "lepidic", "papillary", "solid"]   # sorted: roc_auc_score takes the labels in order


def stage_workspace(target: Path) -> None:
    shutil.copytree(ROOT / "repo", target, dirs_exist_ok=True)
    shutil.copytree(ROOT / "data", target / "kapso_datasets")
    shutil.copytree(ROOT / "evaluation", target / "kapso_evaluation")


def score_once(workspace: Path) -> tuple[float, float]:
    started = time.monotonic()
    run = subprocess.run(
        [sys.executable, "kapso_evaluation/evaluate.py"],
        cwd=workspace, capture_output=True, text=True, check=True,
    )
    seconds = time.monotonic() - started
    last = run.stdout.strip().splitlines()[-1]
    if not last.startswith("macro_auc: "):
        raise RuntimeError(f"evaluator's last line is not the metric: {last!r}")
    return float(last.split(": ", 1)[1]), seconds


def patient_bootstrap_se(workspace: Path, samples: int) -> float:
    private_dir = Path((ROOT / "evaluation" / "private_dir.txt").read_text(encoding="utf-8").strip())
    labels = pd.read_csv(private_dir / "holdout_labels.csv")
    merged = labels.merge(pd.read_csv(workspace / "predictions.csv"), on=ID)
    patients = merged[ID].str.split("-").str[0]
    rows_by_patient = [np.flatnonzero(patients == patient) for patient in patients.unique()]
    truth = merged[LABEL].to_numpy()
    scores = merged[PATTERNS].to_numpy()
    rng = np.random.default_rng(0)
    aucs = []
    for _ in range(samples):
        pick = np.concatenate([rows_by_patient[i] for i in rng.integers(0, len(rows_by_patient), len(rows_by_patient))])
        aucs.append(roc_auc_score(truth[pick], scores[pick], labels=PATTERNS, multi_class="ovr", average="macro"))
    return float(np.std(aucs, ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    manifest = json.loads((ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="pathology-baseline-") as temp:
        workspace = Path(temp) / "workspace"
        stage_workspace(workspace)
        repeats = [score_once(workspace) for _ in range(args.repeats)]
        se = patient_bootstrap_se(workspace, args.bootstrap)

    baseline = {
        "model": "repo/predict.py as shipped: logistic regression on the standardised tile embeddings",
        "metric": "macro one-vs-rest ROC AUC on new patients, higher is better",
        "macro_auc": [round(score, 4) for score, _ in repeats],
        "patient_bootstrap_se": round(se, 4),
        "evaluator_seconds": [round(seconds, 1) for _, seconds in repeats],
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "data_sha256": manifest["sha256"],
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (ROOT / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(baseline, indent=2))


if __name__ == "__main__":
    main()
