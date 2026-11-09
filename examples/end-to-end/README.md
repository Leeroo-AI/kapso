# End to end: Spaceship Titanic

A 45-minute build that takes `repo/train.py` from a DummyClassifier at
about 0.50 accuracy toward the brief's 0.78 on the Kaggle Spaceship Titanic
data, with the repository's own `evaluate.py` as the judge, on the bundled
`predictive-modelling` harness the brief names, and then the rest of the
sequence on what it delivered: the review, a lesson learned into
your bank, a second build served that bank, and the delivered model
deployed as running software.

Recorded result: not recorded yet.

## What you get

- `repo/train.py`, whose `train_model()` and `predict_with_model()` keep
  their signatures and get new bodies: features, a model, hyperparameters,
  on a branch of its own inside `campaign/`.
- A score from `repo/evaluate.py --data-dir ./data --seed 0`: accuracy on a
  90/10 train/validation split of `train.csv`.
- `campaign/.kapso/solution.json`, the record, read back by `Solution.open("campaign")`.
- A card in your bank, written by `learn()` from the first campaign with
  the evidence that earned it; `kapso experience list` shows the bank.
- `campaign-2/`, the second build, whose launch record names `mine` as the
  experience it was served, pinned at the bank's head; under it,
  `.kapso/deployments/1/`, the deployment's record with the handle
  `deploy()` returned.

## Run it

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
kaggle competitions download -c spaceship-titanic && unzip spaceship-titanic.zip -d repo/data/
kapso doctor build
kapso harness check predictive-modelling
python run.py
```

Without the Kaggle CLI, download `train.csv` and `test.csv` from
https://www.kaggle.com/competitions/spaceship-titanic/data into `repo/data/`.
`run.py` is the whole sequence in twelve lines: build, the review, learn,
the bank loaded, build again, deploy. The same sequence from the shell:

```bash
kapso build --brief kapso-brief.md --repo repo --output campaign --detach
kapso watch campaign --follow
kapso trace campaign
kapso learn campaign
kapso build --brief kapso-brief.md --repo repo --output campaign-2 --experience mine --detach
kapso watch campaign-2 --follow
kapso deploy campaign-2 --target local
```

A detached build returns at once; `deploy` opens the campaign's solution
record, which exists only once the build has finished, so the second
`watch --follow` waits for it (`run.py` builds synchronously and needs no
wait).

The second build starts from what the first one established. Whether it
ends higher is the experiment; this file will carry the numbers once a run
is recorded.

## What to look at

- `kapso-brief.md`: the rules say the two functions keep their signatures
  and `test.csv` is never fitted on; the judge holds every candidate to them.
- `solution.explain()`: the score, the stop reason, the cost.
- `cd campaign && python evaluate.py --data-dir ./data --seed 0`: re-measured by hand.
- `kapso trace campaign`: every attempt with its sessions and runs, its
  score and the judge's words, what was rejected and why, what it cost.
- `lesson.explain()`: the lesson's summary, the bank's head before and
  after, the cards created or updated; the card itself, with its
  reliability and its evidence, is in the bank (`kapso experience list`
  names it) and the lesson's reports are beside the trajectory.
- `campaign-2/.kapso/launch.json`: the second build's experience list, the
  bank pinned at its head; `kapso trace campaign-2` shows what its sessions
  read from it.
- `campaign-2/.kapso/deployments/1/`: the deployment's record, which
  `software.record` names; the handle prints its number, target and state,
  `software.proof` is the call the session proved it with, and
  `software.run(inputs)` makes another (a `local` deployment is a function
  call, with no address).

## Change it

- The target: `Target` in the brief.
- The budget: the brief's `Budget minutes=45 experiments=4`, or the command's
  `--budget-minutes` and `--budget-experiments`, which win over it.
- Protect the evaluator from the start by moving `evaluate.py` into a
  directory of its own and passing it with `--evaluation`; the brief's
  rule and the harness's verification hold it until you do.
- The deployment target: `target="local"` in `run.py`, or `--target` on the
  command: `docker`, `modal`, `bentoml`, `langgraph`, or `auto` to let the
  session choose.
