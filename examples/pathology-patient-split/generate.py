"""Deterministic synthetic pathology tiles, grouped by patient.

Byte-identical output for the same arguments. Each patient contributes
40 to 120 tiles, each tile a 32-number embedding standing in for a
foundation model's features, labelled with one of four lung adenocarcinoma
growth patterns. Three things make it a patient-level task:

- every patient carries a signature, a shift and a scale shared by all of
  their tiles (the stain and the scanner), unrelated to the pattern;
- every patient has a dominant mix of patterns, so a model that recognises
  the patient can guess the pattern without reading the tile;
- the holdout tiles come from 40 patients who appear nowhere in training.

A validation that shuffles tiles therefore puts each patient on both sides
and rewards recognising patients: gradient boosting scores about 0.98
macro ROC AUC that way and about 0.86 on the new patients. The shipped
logistic regression (repo/predict.py) lands near 0.84 on the new patients;
a model that reads each tile beside its patient's other tiles reaches about
0.90. Acinar and papillary also differ in the spread of the last eight
features, a cue a linear model misses.

The data is synthetic and illustrative: the pattern names give it a shape,
not a clinical meaning.

    python generate.py            # writes data/, private/, evaluation/private_dir.txt, MANIFEST.json

data/train.csv              agent-visible, with the label
data/holdout.csv            agent-visible, features only, the tiles to predict
private/holdout_labels.csv  the answers; nothing under private/ is copied into a workspace
evaluation/private_dir.txt  the absolute path of private/, read by the evaluator
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261116
TRAIN_PATIENTS = 120
HOLDOUT_PATIENTS = 40
TILES_PER_PATIENT = (40, 121)
FEATURES = 32
SPREAD_FEATURES = 8
PATTERNS = ["lepidic", "acinar", "papillary", "solid"]
CENTROID_SD = 0.9
SIGNATURE_SD = 3.0
SCALE_SD = 0.4
TILE_NOISE_SD = 1.8
MIX_CONCENTRATION = 0.3
PAPILLARY_SPREAD = 2.5
EMBEDDING_COLUMNS = [f"emb_{i:02d}" for i in range(FEATURES)]
FEATURE_COLUMNS = ["tile_id", "patient_id", *EMBEDDING_COLUMNS]
LABEL = "pattern"


def make_centroids(rng: np.random.Generator) -> np.ndarray:
    centroids = rng.normal(0, CENTROID_SD, (len(PATTERNS), FEATURES))
    # Acinar and papillary share all but the last eight centroid features,
    # and papillary differs in their spread too.
    centroids[PATTERNS.index("papillary"), :-SPREAD_FEATURES] = centroids[PATTERNS.index("acinar"), :-SPREAD_FEATURES]
    return centroids


def make_frame(rng: np.random.Generator, centroids: np.ndarray, patients: int, first_patient: int) -> pd.DataFrame:
    frames = []
    for number in range(first_patient, first_patient + patients):
        tiles = int(rng.integers(*TILES_PER_PATIENT))
        mix = rng.dirichlet(np.full(len(PATTERNS), MIX_CONCENTRATION))
        labels = rng.choice(len(PATTERNS), tiles, p=mix)
        shift = rng.normal(0, SIGNATURE_SD, FEATURES)
        scale = np.exp(rng.normal(0, SCALE_SD))
        tissue = centroids[labels] + rng.normal(0, TILE_NOISE_SD, (tiles, FEATURES))
        tissue[:, -SPREAD_FEATURES:] *= np.where(labels == PATTERNS.index("papillary"), PAPILLARY_SPREAD, 1.0)[:, None]
        frame = pd.DataFrame(scale * tissue + shift, columns=EMBEDDING_COLUMNS)
        frame.insert(0, "patient_id", f"P{number:04d}")
        frame.insert(0, "tile_id", [f"P{number:04d}-T{i:03d}" for i in range(tiles)])
        frame[LABEL] = [PATTERNS[label] for label in labels]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.3f", lineterminator="\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent), help="the fixture directory")
    args = parser.parse_args()
    root = Path(args.root).resolve()

    rng = np.random.default_rng(SEED)
    centroids = make_centroids(rng)
    train = make_frame(rng, centroids, TRAIN_PATIENTS, 1)
    holdout = make_frame(rng, centroids, HOLDOUT_PATIENTS, TRAIN_PATIENTS + 1)

    write_csv(train, root / "data" / "train.csv")
    write_csv(holdout[FEATURE_COLUMNS], root / "data" / "holdout.csv")
    write_csv(holdout[["tile_id", LABEL]], root / "private" / "holdout_labels.csv")
    (root / "evaluation" / "private_dir.txt").write_text(str(root / "private") + "\n", encoding="utf-8")

    manifest = {
        "seed": SEED,
        "patients": {"train": TRAIN_PATIENTS, "holdout": HOLDOUT_PATIENTS},
        "tiles": {"train": len(train), "holdout": len(holdout)},
        "features": FEATURES,
        "label": LABEL,
        "patterns": PATTERNS,
        "pattern_share": {
            "train": {name: round(float(share), 4) for name, share in train[LABEL].value_counts(normalize=True).reindex(PATTERNS).items()},
            "holdout": {name: round(float(share), 4) for name, share in holdout[LABEL].value_counts(normalize=True).reindex(PATTERNS).items()},
        },
        "sha256": {
            "data/train.csv": sha256(root / "data" / "train.csv"),
            "data/holdout.csv": sha256(root / "data" / "holdout.csv"),
            "private/holdout_labels.csv": sha256(root / "private" / "holdout_labels.csv"),
        },
    }
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
