# Pathology tiles, scored on new patients

The data is synthetic. `generate.py` makes every row from a seed; no row
comes from a real patient or slide, and the example makes no clinical
claim.

A build of about three quarters of an hour on the predictive-modelling
harness that raises a growth-pattern classifier's macro ROC AUC on new
patients from the shipped logistic regression's 0.842 toward the brief's
0.88 target. Every tile belongs to a patient and the holdout patients are
new. A validation that shuffles tiles puts each patient on both sides of
the split, so on this data it scores recognising the patient as well as
reading the tile, and reads higher than the score on new patients.

Each row is a tile, standing for a small patch of one patient's slide,
described by 32 numbers in place of a pathology foundation model's
features and labelled with one of four growth-pattern names (lepidic,
acinar, papillary, solid). The names give the task a shape, not a
clinical meaning.

## What you get

- `repo/predict.py`, the starting model, improved on a branch of its own
  inside `campaign/`, one branch per experiment.
- A score measured by `evaluation/evaluate.py`: macro one-vs-rest ROC AUC
  on 3,092 tiles from 40 patients whose labels never enter the workspace.
- `campaign/.kapso/solution.json`, the record; `Solution.open("campaign")`
  reads it from any process.

## Run it

```bash
pip install -r requirements.txt   # the generator, the model, the evaluator and the harness's pyarrow
python generate.py                # the data: 120 training patients, 40 held out, seeded
kapso harness check predictive-modelling
kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other. The brief's `Harness` line names the
harness, so the command needs no `--harness`.

## What to look at

- The shipped `predict.py` prints a validation score from a random split
  of the training tiles when it runs (the evaluator prints only its own
  score): 0.885, against 0.842 on the new patients, both measured on the
  generated data when the example was built, 2026-11-17. The gap is the
  patients it saw on both sides of that split.
- `kapso trace campaign`: how each attempt validated and what the audit
  found. The harness's cv-protocol skill tells a session to group folds by
  the repeated unit when the test holds unseen units, and its
  leakage-auditor reports a fold that splits one.
- `solution.explain()` and `solution.diff()`: the delivered candidate, its
  measurement, and its code against the seed.

## Change it

- The target: edit `Target` and the objective. A model that reads each
  tile beside its patient's other tiles scored 0.895 under the evaluator
  (0.893 when validated by patient, 0.995 when validated on shuffled
  tiles), measured on the generated data when the example was built,
  2026-11-17, so 0.88 sits between the baseline and a measured model and
  above the baseline's noise. The
  brief allows it: a tile may be read beside the other tiles of its own
  patient, the way a slide is scored whole.
- The budget: the brief's `Budget minutes=45 experiments=3`, or
  `--budget-minutes` and `--budget-experiments` on the command, which win
  over it.
- The evaluator: `evaluation/evaluate.py` runs `python predict.py --data
  kapso_datasets --out predictions.csv`, checks the file (the five
  columns, numbers in [0, 1], each row summing to 1, one row per holdout
  tile), scores it against the private labels, prints `macro_auc: 0.8418`
  as its last line, writes `kapso_evaluation/result.json`, exits 2 on
  malformed predictions and 3 when `predict.py` fails; `--fast` scores the
  first 8 patients, a smoke check.
- The data: `generate.py` (seed 20261116) writes `data/`, `private/` and
  `MANIFEST.json` with the hash of every file; `baseline.py` records
  `baseline.json` (0.842, a standard error of 0.021 from a bootstrap over
  patients, since tiles of one patient move together).

The hold-out is not sealed by this example: the labels sit outside the two
directories the engine copies into the workspace, and nothing more.
