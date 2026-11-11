# A build with no evaluator: fraud scoring

A build of about three quarters of an hour on labelled transactions and a rule-based scorer,
with no evaluator anywhere. The brief states the protocol instead; the
campaign's first node builds the evaluator from it in `kapso_evaluation/`,
and the judge reads every candidate's evaluation against that protocol.
Registering and protecting it file by file is the evaluation maintainer's
job, which the shipped config does not turn on yet; see
[Evaluation](https://docs.leeroo.com/docs/build/evaluation).

## What you get

- `repo/score.py`, three hand-written rules, replaced on a branch of its own
  inside `campaign/` by a model fitted on the 80 percent split.
- An evaluator the campaign wrote: `campaign/kapso_evaluation/evaluate.py`,
  following `Measured by` in the brief.
- `solution.score`, the precision-recall AUC on the held-out 20 percent,
  and `solution.feedback`, the judge's last verdict on how the evaluation
  was done.

## Run it

```bash
pip install numpy pandas scikit-learn   # the generator and the starting scorer
python generate.py        # 20,000 labelled transactions, seeded, about 5 percent fraud
kapso doctor build
kapso build --brief kapso-brief.md --data data --repo repo --output campaign --detach
kapso watch campaign --follow
```

`python run.py` is the same build from Python, into the same `campaign/`
directory: run one or the other.

## What to look at

- `kapso-brief.md`: `Measured by` is a protocol, not a path, and `Baseline`
  is `unknown`. The campaign's first measurement replaces the unknown.
- `campaign/kapso_evaluation/evaluate.py` after the first node: the split, the
  seed, the metric, exactly as the protocol said.
- `solution.explain()` and `solution.feedback`: the score, and the judge's
  reading of the evaluation behind it.
- `kapso trace campaign`: every attempt with its sessions and runs, its
  score and the judge's words, what was rejected and why, what it cost.

## Change it

- The protocol: change the split, the seed or the metric in `Measured by`;
  the evaluator the campaign builds follows the new text.
- The target: `pr_auc above 0.50` is proposed with headroom (the shipped rules score 0.09 under the protocol, gradient boosting with no feature work 0.55, measured 2026-10-24 on the generated data); a target the
  data cannot reach ends in `max_iterations`.
- Bring your own evaluator instead: write it, pass it with `--evaluation`,
  and it is protected from the start; `Measured by` then names the file.
- The data: `generate.py` (seed 20261025) writes `data/transactions.csv`
  and `MANIFEST.json` with its hash.
