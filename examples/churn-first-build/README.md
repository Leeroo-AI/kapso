# Your first build: churn, with an evaluator present

A first build, about three quarters of an hour on the shipped config, that
raises a churn model's out-of-sample ROC AUC from the shipped logistic
regression's 0.797 toward the brief's 0.85 target, and hands back a `Solution`: the winning branch, its score under an evaluator the
campaign cannot edit, and the record of the whole campaign.

## What you get

- `repo/predict.py`, the starting model, improved on a branch of its own
  inside `campaign/`, one branch per experiment.
- A score measured by `evaluation/evaluate.py`: ROC AUC on 3,000 held-out
  rows whose labels never enter the workspace.
- `campaign/.kapso/solution.json`, the record; `Solution.open("campaign")` reads
  it from any process, and `kapso deploy campaign` ships it.

## Run it

```bash
pip install numpy pandas scikit-learn   # the generator, the model and the evaluator
python generate.py        # the data: 10,000 training rows, 3,000 held out, seeded
kapso doctor build
kapso build --brief kapso-brief.md --evaluation evaluation --data data --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other. It blocks until the build returns and
prints `solution.explain()`. `--evaluation evaluation` copies the directory
to `kapso_evaluation/` inside the campaign, which is why the brief names
it there.

## What to look at

- `kapso-brief.md`: the objective, how it is measured, the baseline, the
  target, the rules the judge enforces, the budget. Everything the campaign
  knows about the task is in it, and all of it reaches the judge.
- The summary block: `COMPLETED` with the candidate, the score and the
  stop reason, or `WAITING ON YOU` with what to answer (`kapso inbox
  campaign`). A detached run writes it to `campaign.log` beside the
  campaign; `kapso watch` shows the live view, not this block.
- `solution.explain()` and `solution.diff()`: the delivered candidate, its
  measurement, and its code against the seed.
- `git -C campaign branch`: every experiment, kept.
- `kapso trace campaign`: every attempt with its sessions and runs, its
  score and the judge's words, what was rejected and why, what it cost.

## Change it

- The target: edit `Target` and the objective. A target the data cannot
  reach ends in `max_iterations`, not in a result.
- The budget: the brief's `Budget minutes=45 experiments=3`, or
  `--budget-minutes` and `--budget-experiments` on the command, which win
  over it. The ideation step runs ten to fifteen minutes before the first
  candidate is implemented, so a budget under half an hour starves the
  implementation, and each further experiment needs about twenty-five
  minutes of its own.
- The evaluator: `evaluation/evaluate.py` runs `python predict.py --data
  kapso_datasets --out predictions.csv`, scores the predictions against the
  private labels, prints `auc: 0.8412` as its last line, writes
  `kapso_evaluation/result.json`, exits 2 on malformed predictions and 3
  when `predict.py` fails; `--fast` scores 600 rows, a smoke check. Change
  the metric there and in `Measured by` together.
- The data: `generate.py` (seed 20261024) writes `data/`, `private/` and
  `MANIFEST.json` with the hash of every file; `baseline.py` records
  `baseline.json` (0.797, bootstrap standard error 0.010, measured on the
  generated data when the example was built, 2026-10-24). A gradient
  boosting model with no feature work scores 0.832 on the same data, so
  there is room.

The hold-out is not sealed by this example: the labels sit outside the two
directories the engine copies into the workspace, and nothing more. Sealing
is the engine's job (evaluation integrity and permissions); until then this
example measures the flow, not the boundary.
