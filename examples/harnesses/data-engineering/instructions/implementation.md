# Implementation

On a dbt project a ticket is closed by these steps, in this order, whatever the plan lists,
for every ticket the session closes; on another stack, the same measurements by hand (invariants).

```bash
E="python3 $KAPSO_HARNESS_ROOT/skills/validation-report/scripts/evidence.py"
# 1. the oracle first: results/<ticket>/spec_<dataset>.json for every dataset of the ticket (invariants)
$E baseline --ticket <t> --kind fix|new|refactor|performance|migration --select <model> ... --project-dir <dir>  # 2. before any file changes
# 3. change one dataset, build it with selection (dbt-workflow), look at its rows
$E prove --ticket <t> [--with-downstream]   # 4. after: a reset, the build, the specs on it and on production, the comparison, the acceptance check
# 5. the proof file (validation-report), then the label, computed; then commit the ticket:
$E label --ticket <t> --write
```

- The label is computed from those measurements, never chosen: the stop hook,
  verification and the delivery compute it again. Passing the task's check, where it has
  one, earns `unsure` by itself; a measurement still within reach is work left undone.
- Where setup pinned a reset, `baseline` and `prove` each put the build namespace back to
  the reference state, so a ticket is measured with no other ticket's change or hand build
  around it; without one, the proof says what else the namespace held. A file edited before
  its baseline cannot be claimed. Code that needs another ticket's change names it in `needs`.
- With a backlog (ticket-backlog): one ticket at a time through the steps, then its
  proof (results/proofs/<ticket>.json), its commit and the ledger (results/backlog.json);
  a ticket that resists is parked with its note. Last, the task's whole acceptance run.
- Look at the data first (the suspect node's rows, key cardinality, null rates, the
  control total against production), then bisect (lineage-bisection): at each edge state
  the expected filtering, multiplicity and aggregation, stop at the first that departs,
  and name the line with data-bug-mechanisms. Fix the cause, not the number.
- One dataset at a time; keep its columns, materialization and grain unless the ticket
  asks otherwise (grain-and-modelling). A refactor or an optimisation starts from the
  row comparison (data-diff) and reports the billed quantity under recorded conditions
  (warehouse-cost); a changed load gets its replay comparison (pipeline-reliability) and
  its runbook in the proof; what depends on engine behaviour runs on every engine the
  task names (engine-portability).
- Ask the root-cause-reviewer when the fix lives downstream but the cause smells upstream,
  or when the ticket's symptom cannot be reproduced.
- A constant from the ticket in place of a mechanism, a write to source data, an output
  the ticket did not ask for, a test or a shared layer changed without independent
  evidence and a line in the proof: each is a finding against the attempt.
