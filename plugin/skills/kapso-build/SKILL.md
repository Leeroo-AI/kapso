---
name: kapso-build
description: Turn what the user wants built into a Kapso campaign — discover the repository's data, evaluation and domain in silence, write the brief (objective, how it is measured, baseline, target, rules, budget), get the evaluation approved through kapso-evaluation, and launch detached with the handoff. Use when the user says build X and verify it, I need Y that meets Z, improve W, make this pass the benchmark, get the metric above a number, ship a model that meets a bar, or runs /kapso:build. Do not use for a one-file change with an obvious fix, for operating a campaign already running (the kapso skill), or inside a Kapso campaign session.
---

# From a wish to a launched campaign

Kapso builds and verifies an objective end to end: it is not a metric-improvement
tool, and "improve the AUC" is one shape of objective among "build a pipeline that
produces X and prove it" or "make this pass the benchmark". The objective is the
user's; your job is to make it buildable and verifiable with as little talking as
the repository allows: discover everything first, in silence, then speak once.

## 1. Listen for two things

What the user wants to exist at the end, and how they would know it is done. Both
may be vague ("a better churn model", "a working ingestion pipeline"): you propose
the measurable form rather than asking for it. A wish that names no task ("build
something useful here") gets the same one message as any other (section 4): the
findings, a proposal line that says what will exist and what it does for the user,
one alternative line, the brief, and the one approval question it ends on — within
the same ten lines. The evaluation is built as the evaluation skill says: a
repository with no data gets fixtures the proposal makes and an evaluation that
scores against them, so the campaign starts the moment the user says go.

## 2. Discover, in silence, in under a minute

Read the repository: README and docs, data files and directories, notebooks and
their metric cells, test suites, scripts named like evaluators or benchmarks, CI,
dependencies. Infer the domain and the harness that fits (`kapso harness list
--config kapso-config.yaml`; or confirm the one loaded), and read that harness's
setup page (`instructions/setup.md`): what a task statement in this domain must
record becomes questions to the repository, not to the user. Note which experience
sources exist (`kapso experience list --config kapso-config.yaml`: a bank with
cards, a Leeroopedia key, an ingested index).

## 3. The evaluation

Hand what you found to the `kapso-evaluation` skill (`/kapso:kapso-evaluation`),
which owns the one thing the product promise rests on: an evaluation Kapso can
protect. Its three branches: an evaluator found is audited against the contract and
run once for the baseline; a criterion without an evaluator gets one built from the
criterion and the data; neither gets candidate criteria proposed from the
repository with their evidence, one recommended. This happens before the user hears
anything, except the single approval it needs. When the user asks what the
candidates were, the comparison goes in one fenced block (a table of the criteria,
their evidence and figures), which the message's line budget exempts, and the prose
around it is at most three lines — the recommendation and why, ending there with no
question: the user asked to see, and the one approval was already asked.

## 4. One message

What was found (data, evaluator, baseline, domain), with the evidence each rests on
("`evaluate.py` prints macro-F1 on `data/dev.csv`, 1,204 rows"); the brief as the
file `kapso-brief.md`; the evaluation's protocol for approval when one was built or
changed; and the single thing the user must confirm or edit. Short: the brief is the
message, and the prose around its fence is at most ten lines in all — one line for
each thing found (the data, the evaluator, the baseline, the domain), its evidence in
that same line; when an evaluation was built or changed, two lines, what it measures
and how it was checked; and the one line that asks the approval. The four findings
are the message's worth and are never dropped for brevity: the data with its rows and
label, the evaluator with what it prints on what, the baseline with the figure it
measured, the domain with the harness loaded — each a fact the user did not have, with
its figure or file. No headings, no sub-bullets, no paragraph, and no line about this
skill's own rules: a line each. The brief, one field per line, indented lines
continuing the field above:

```
Objective    Raise the churn model's out-of-sample AUC above 0.85
             without changing the prediction contract.
Measured by  evaluation/evaluate.py, ROC AUC on the held-out rows
Baseline     0.797 (logistic regression, bootstrap SE 0.010)
Target       0.85
Rules
  - the evaluator is never modified
  - no fitting on the held-out rows
Budget       minutes=60 experiments=10
Harness      predictive-modelling
Experience   mine, leeroopedia
```

- Only `Objective` is required; whatever the user gives goes in verbatim, and the
  whole brief reaches the judge, which stops the campaign only when the objective
  reads as fully achieved and scores every experiment from what the evaluation
  prints. So name the metric and a number, and the judge.
- `Baseline` is measured when that is cheap, else `unknown`. `Target` is the user's,
  or proposed with headroom below any ceiling you can see and the reasoning in one
  line; a target the data cannot reach ends in the inbox, not in a result.
- Rules go in, methods do not: the judge enforces data rules and prohibitions as
  invariants for the whole campaign, and would freeze a method preference the same
  way.
- `Harness` names the domain layer (a bundled name, a directory, a git reference);
  `Experience` names what the build is served, each source existing already
  (`kapso experience add` makes one).

## 5. At most one more round

The user edits a line or says go. A second round only when the edit changes the
evaluation. Never ask twice; never say you cannot proceed.

## 6. Launch, and hand off

```bash
kapso doctor build --json --config kapso-config.yaml
kapso build --brief kapso-brief.md --repo . --evaluation evaluation --data data --output ../churn-campaign --budget-minutes 60 --budget-experiments 10 --config kapso-config.yaml --detach
```

- `--evaluation` only when an evaluator exists or was approved (it is copied and
  protected); without one the campaign builds its own from `Measured by`.
- `--output` is an empty sibling directory, never inside the repository; `--config
  kapso-config.yaml` whenever the project root holds that file.
- The handoff, in the same reply, at most eight lines and no headings: the brief
  you passed; what you assumed and that the user's own criterion can replace it; the
  harness and its version when one is loaded; `kapso watch ../churn-campaign --follow
  --config kapso-config.yaml`; and that `WAITING ON YOU` is a pause — `kapso inbox
  ../churn-campaign` shows the request, and once what its fix line names is in place
  `kapso inbox reply ../churn-campaign <id> "<note>" --detach` answers it
  (`/kapso:inbox` does both). The run outlives your tool call; do not poll it.

## Rules

- The criterion comes from the user or the repository, never invented; every
  proposal carries its evidence.
- Never says it cannot proceed; asks one question in all — the approval of the
  proposal, which states every choice it made (the criterion, the evaluation's split
  and metric, the target) as a default the user may change in their answer — and
  never asks twice; a second round only answers what the user changed; the whole
  exchange under five minutes.
- A one-file change with an obvious fix is done by hand, and you say so instead of
  launching.
- Never launch a campaign the user did not ask for, and never from inside a Kapso
  campaign session.
