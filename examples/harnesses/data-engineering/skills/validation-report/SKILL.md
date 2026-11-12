---
name: validation-report
description: How a senior data engineer states what a change proved, with confidence as a list rather than an adjective (what was validated and how, the outcomes, dev against prod, what was not inspected, skips with reasons, assumptions for an expert), the kind of change and the proof each kind obliges, the three labels safe to merge, unsure and won't do and the script that measures a change and computes the label it earns (evidence.py environment, baseline, prove, label, verify, lower), "the reference is wrong" as a first-class outcome, the results/proof.json format every attempt writes, and the phrasing. Use in setup to name the kind of change and check the environment file, before changing a ticket's first file and after its last, when writing the proof at the end of an implementation, when auditing a report in verification, and when weighing attempts in feedback.
---
# The validation report

A validation report states which declared obligations the measurements establish and
which remain unproved. Its label follows those obligations even when a score, a reviewer
or a stakeholder disagrees: a "safe" that fails costs more than an "unsure" that passes.

## Confidence is a list

A defensible claim lists: the resources and commands validated; compile, lint, run and
test outcomes; the differences between development and production; the downstream
resources not run or inspected; every failure, warning and skipped check with the reason
for the skip; and every assumption that still needs a subject-matter expert. It says what
was computed ("net revenue, calendar quarter three, as of 10:42"), gives a quantified bound
with a named cause rather than a hedge ("there is some uncertainty" lowers trust; a stated
range with its reason does not), and never argues from effort.

## The kind of change decides the proof

State `change_kind` as `fix`, `refactor`, `performance`, `new`, `migration` or
`no_change`. A reproduced defect is required only for `fix`.

| Kind | What the proof must show |
| --- | --- |
| fix | a cause reproduced by a query and named with file and lines; checks that failed before and pass after; checks that passed in both states, listed as preserved |
| refactor, performance | the output equal to the reference's row by row, on rows shown unique; the structure preserved; for performance the requested metric and the billed quantity under recorded conditions (warehouse-cost) |
| new | an acceptance oracle independent of the dataset's own code: reconciliation to sources, the grain's uniqueness, the measures recomputed another way |
| migration | the data preserved row by row and the consumers' contract preserved or migrated, with each consumer named; a migration that changes the grain or the columns on purpose cannot show that equality and stays `unsure`, with its mapping and each consumer's evidence for the reviewer |
| no_change | `changed_files` empty, and `decision_evidence` naming the artifacts that contradict the premise, show the information missing, or state the business decision required |

Every kind but `no_change` adds the downstream comparison with each changed dataset
explained, the replay comparison and the runbook when a load changed, and the engines
executed (engine-portability).

## The labels

| Label | What it says |
| --- | --- |
| safe to merge | every obligation of the kind is met by a measurement, and verification took the measurements again |
| unsure | the change passes what it can prove, and one thing is named that it cannot: a measurement not taken, or a question only the ticket's author can answer (a business definition, an engine out of reach, a downstream change the ticket does not settle) |
| won't do | `change_kind` `no_change` with its evidence: the premise shown wrong by a query, a decision that belongs to the business, or the data shown absent; a refusal needs the same proof as a fix |

A missing obligation moves the label down, never the reasoning up. "Unsure" is not a
failure; it is the honest state of a change whose last check is still owed.

## The label is computed

Nobody chooses a label. One script measures the change and says what the measurements
earn; the stop hook, verification and the publication of a pull request each ask it
again, so a label typed into a proof changes nothing.

```
E="python3 $KAPSO_HARNESS_ROOT/skills/validation-report/scripts/evidence.py"
$E environment [--ticket <ticket>]           # setup: is prepared/environment.json usable; how one acceptance run is read
$E baseline --ticket <ticket> --kind <kind> --select <model> [<model> ...] [--snapshot <name> ...] --project-dir <dbt project>
$E prove    --ticket <ticket> [--with-downstream]
$E label    --ticket <ticket> --write        # or --all; --json for the obligations as data
$E verify   --all --out <file>               # verification: measure every closed ticket again
$E lower    --in <file> --ticket <ticket> --reason "<finding>"   # verification: a reviewer's finding lowers a label
```

- `baseline`, before any file of the ticket changes. Every dataset of the ticket has its
  spec first (results/<ticket>/spec_<dataset>.json, the invariants skill): the oracle
  predates the change, and its hash is recorded. The command records the commit and
  what already differs from it, runs the reset command, and unless the kind is `new`
  builds the code as it stands and runs the specs on it. A dataset that does not build
  is recorded with its error when the error is its own; a project broken elsewhere is
  refused until that is fixed: under its own ticket when the backlog has one, otherwise
  reported as the first finding, which the task's owner decides on. A second baseline needs
  `--again "<what was wrong with the spec>"`. One ticket is measured in one dbt project
  (`--project-dir`, a path inside the repository): a file it changes in another project
  is still named in `changed_files`, and its effect is measured through the ticket's own
  datasets. prepared/environment.json holds one set of targets and one reset command,
  read for every project of the repository. Where two projects do not share target
  names, measure in the project the file was written for and say in the proof that the
  other was not measured.
- `prove`, after the change. It puts the namespace back to the reference state, so every
  measurement of a candidate starts from the same place whatever was built by hand. It
  builds the datasets with one `dbt build` (models and snapshots in the graph's order,
  no tests) and reads dbt's own record of that command,
  `<project>/target/run_results.json`, for each one's success; a build that leaves no
  record there measures nothing. It runs each spec on the build and on the reference
  target, fingerprints both, compares the rows by `except` when the kind promises an
  unchanged output, runs the acceptance check when one is pinned, and hashes every
  report and every changed file. With `--with-downstream` it rebuilds everything that
  reads the datasets, snapshots included, and screens each against production's by its
  fingerprint: the same fingerprint is recorded as that, not as equal rows. A
  downstream dataset that was not built was not compared. For one whose own build
  failed, the script then builds the same selection once more from the code as it
  stood at the baseline's commit, exported beside the workspace, on the reset
  namespace. A consumer that fails there with the same error cannot be built with or
  without this change: it is named in the pull request as broken before it and is not
  held against the ticket. Any other consumer that was not compared keeps the label at
  `unsure`: one that was skipped, one that fails differently, and every one when the
  code as it stood builds nothing at all (nothing built before the change, so there is
  no earlier state to compare with). Say in the proof what you found about
  that consumer, for the reviewer who decides.
- `label` prints each obligation as met or missing, from the hashed reports and never
  from a summary: the record is whole; the proof's kind is the baseline's; a fix names
  its cause; reports and specs are unchanged; the code is as it was measured (every
  file the measurement saw changed, named by the proof or not), the proof's files are
  among them and changed after the baseline, and each of them is named by a closed
  ticket's proof (this ticket's, or the proof of the ticket that changed it later);
  every dataset was built and checked; every check passes and none regressed or went
  missing; production was measured or shown absent; by kind a check that moved (fix),
  or the reference's columns by name and kind with rows equal to production's and no
  duplicate row (refactor, performance, migration); for every dataset production does
  not hold yet, a declared grain and a passing check against another relation; every
  downstream dataset was rebuilt and screened, each changed one has a reason, and
  none is left unmeasured but one that fails with the same error on the code as it
  stood; no assumption and no unknown; a changed load has its replay comparison, taken
  after the baseline and free of duplicate rows, and its runbook; the acceptance check
  passes. Anything missing gives `unsure` with that as its reason.

What the script does not compute, the reviewers of verification judge, and a finding of
theirs lowers the label through `lower`: a spec that cannot fail or restates the code, a
reason for a downstream change that names no mechanism, the requested metric of a
performance change, an engine the task names that was not run, a shared layer or a test
changed without independent evidence.

The reports sit in the workspace, where the session that made the change can write. The
label is a consistency check of its inputs; it is believed because verification builds
and measures again from the code. `verify` takes again the build, the specs on both
sides, the fingerprints, the row comparison and the acceptance check, into a directory
of its own, on the code as it is delivered. That takes about as long as the build
session's own `prove` runs together: when the verification window the Kapso config gives
is shorter, name the tickets (`--only`) and say in the verdict's summary that the window,
not the evidence, left the others `unsure`, so that the user can raise
`harness.verification_minutes`. That measurement is the ticket's after-state
from then on: the label is what it earns, and the pull request is published on it, its
label computed once more at publication against the files as they are then
(`label --ticket <ticket> --evidence <verification's evidence file>`). The build
session's own measurement is history by then: a later ticket's commit may have changed
what it measured, and what it said differently lowers the label only when the code was
the same in both. `verify` does not take again a load's replay comparison or the
measurement of the code as it stood before the change: the reviewers read those
reports, and a finding lowers the label. The scripts drive dbt: on another stack they
cannot take the measurements, the label stays `unsure` with that as its reason, and no
report written by hand stands in for them.

`acceptance_at_a_time` in prepared/environment.json says the task's check builds a state
of its own, away from this checkout: it neither reads the build namespace nor runs dbt
in the workspace's project directories (a service that is sent the files, a copy of the
tree). Only then does it run beside a measurement, several at once, and on each
pull-request branch alone. Without that key a ticket's check runs after its measurement
has built the namespace and before any control replaces that build, one at a time, and
no branch is checked alone.

Without `reference_target` nothing can be shown against production, and no ticket reads
`safe to merge`; without an acceptance command the label is computed from the specs alone.
Both are ordinary states of a repository, not faults of the attempt: say so in the proof.
An acceptance command pinned without a pass pattern is read by its exit status.

An acceptance check someone else owns, whose definitions are not shown, is listed under
`not_inspected`: it is not an assumption of the change and not a question for the
author. `assumptions` holds what the change itself takes for granted; `unknowns` holds
the questions only the ticket's author can answer, and either one keeps the label at
`unsure` on purpose.

## The reference may be wrong

Test the expectation against the authoritative definition and an independent
reproduction; a stakeholder's number, a dashboard and a reference answer deserve the same
scepticism as the pipeline. When the discrepancy belongs to the reference, report that
evidence and leave the correct transformation unchanged: "the daily totals reconcile to
the settled ledger entries within the declared tolerance on every day of the range; the
discrepancy is in the
dashboard's filter, not in the dataset" is a complete, labelled outcome.

## results/proof.json

```json
{
  "ticket": "the ticket's key",
  "model": "merchant_payments_daily",
  "change_kind": "fix",
  "label": "written by evidence.py label --write",
  "label_reasons": ["written by evidence.py label --write"],
  "validated": {"commands": ["the project's build command for the dataset", "the task's acceptance check for the ticket"],
                "outcomes": "built; 9 of 9 checks pass; the acceptance check passes",
                "not_inspected": ["finance_export_feed", "the acceptance check's own definitions"],
                "skipped": [{"check": "production-engine execution", "reason": "no credentials in this environment"}],
                "assumptions": []},
  "root_cause": {"summary": "one sentence a reviewer can check", "file": "models/marts/merchant_payments_daily.sql", "lines": "41-47",
                 "reproducing_query": "select ... "},
  "blast_radius": {"changed_models": [{"model": "merchant_payments_weekly", "reason": "sums the corrected daily amounts"},
                                      {"model": "merchant_rollup_*", "reason": "every rollup reads the corrected amount column"}]},
  "needs": "another-ticket",
  "rerun": "results/<ticket>/replay_compare.json",
  "runbook": {"one_time_actions": ["full refresh of merchant_payments_daily"], "order": [], "verification": ["row count equals the key count"], "recovery": []},
  "portability": [{"engine": "duckdb", "evidence": "compared"}, {"engine": "snowflake", "evidence": "compiled"}],
  "performance": {"before": "results/<ticket>/performance_before.json", "after": "results/<ticket>/performance_after.json"},
  "decision_evidence": [],
  "changed_files": ["models/marts/merchant_payments_daily.sql"],
  "unknowns": []
}
```

The session writes what only it knows: the cause, the validation list, what the change
takes for granted, the questions for the author, and:

- `changed_files`: the files of this ticket and no other's, all of them (a macro it
  touched as much as a model): they must have changed between its baseline and its proof
  run, and a changed file of the project that no proof names keeps the label at `unsure`.
- `blast_radius.changed_models`: a reason for every downstream dataset `prove` found
  changed. An entry names one dataset, or a family by a name prefix of three characters
  or more followed by `*`; no other pattern is read. The reason says what changed in the
  input and what that does to the consumer.
- `needs`: the one other ticket without whose change this ticket's code does not build
  (it introduces something this ticket's code refers to). The pull requests are
  stacked from it. A ticket is still measured on what production holds: when its checks
  can pass only on data that other ticket's change produces, its own baseline selects
  that ticket's datasets too, so they are built and checked in its measurement.
- `rerun`, when the change touched a load (a snapshot, an incremental model): keep a
  copy of the load's output, run the load again on the same inputs, then compare by row
  with `fingerprint.py --except "<the copy>" "{{ ref('<the load>') }}" --save
  results/<ticket>/replay_compare.json`; the label reads that report.
- `runbook`, for a changed load or a migration: what to run once, in what order, how to
  see that it worked, how to recover; four lists, at least one filled.

The measurements are not retyped: they are in results/<ticket>/evidence.json, and the
pull-request body is rendered from there. Every `skipped` entry is an object with `check`
and `reason`. `performance` is present for a cost or speed ticket, `decision_evidence`
for `no_change`; `root_cause.file` and `lines` may be null for any kind but `fix`.

## A backlog: one proof per ticket

With several tickets (ticket-backlog) each ticket's proof is its own file,
results/proofs/<ticket>.json, in the format above, and its specs and measurements live
under results/<ticket>/. `ticket` is the ticket's key. The label is earned per ticket;
the backlog has no label of its own, and results/backlog.json is its ledger. A single
ticket uses results/proof.json and the ticket name `change`.
