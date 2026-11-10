---
name: lineage-bisection
description: How a senior data engineer finds where a wrong number is born, with the stakeholder's figure reproduced first, the upstream-change question before the SQL question, the lineage walked edge by edge with the expected filtering, multiplicity and aggregation stated at each edge and the residual measured, the compiled SQL read at the first edge that departs, the rule for an inconclusive walk, and a script that prints a dbt model's lineage with row counts. Use at the start of every ticket, in setup for the statement, and whenever a fix downstream does not move the symptom.
---
# Lineage bisection

On a dbt project:

```
python3 $KAPSO_HARNESS_ROOT/skills/lineage-bisection/scripts/lineage.py <model> \
  [--project-dir .] [--profiles-dir DIR] [--target NAME] [--counts] [--downstream] [--json]
```

Prints the upstream tree from the project's graph (depth, resource type, materialization,
file), with `--counts` the row counts of the dataset and its direct parents (materialized
relations only; tests and ephemeral nodes stay graph entries), with `--downstream` the
consumers instead. It refuses an ambiguous model name (two packages or versions). On
another stack the graph comes from the orchestrator (a DAG's task dependencies, an asset
graph, a job's input tables, a warehouse's view dependencies) and the walk is the same.

## The method

1. **Reproduce the stakeholder's figure exactly**, with the same filter, period, time zone
   and grain, and write the query down; the stakeholder's report is usually the only
   detector that fired. If it cannot be reproduced, say so: the premise may be wrong, and
   that is a finding, not a failure.
2. **Ask what changed upstream before what is wrong in the SQL.** Inspect producer, schema
   and load changes before selecting a transformation to edit: the sources' row counts,
   schema and freshness, the producer's releases, recent backfills and ad-hoc writes.
3. **Walk the lineage edge by edge at a comparable grain.** At each edge state the expected
   filtering, multiplicity and aggregation, then measure the residual on both sides: row
   count, key cardinality, null rate on the suspect column, the measure's control total.
   Compare with a known-good state when one exists; otherwise with an independently
   justified source calculation. Inspect both branches of a join, and separate a changed
   input population from a wrong transformation. Stop at the first edge that departs
   while its inputs do not.
4. **Check the grain at the suspect node.** Row count before and after each join; `group by
   key having count(*) > 1`. A fan-out doubles measures silently; a predicate on a
   left-joined column drops rows silently; an incremental boundary drops late rows.
5. **Read the compiled SQL** of that node (under `target/compiled` on a dbt project) and
   name the line with the data-bug-mechanisms skill; copy the compiled query into the
   warehouse to see the error where it occurs, materializing a chain as tables temporarily
   when the engine hides which step fails.
6. **Write the reproducing query into the proof** beside the line; verification reruns it.

## Rules

- The proximal symptom is rarely the cause. Continue until a query distinguishes the
  proposed cause from the nearest plausible alternative; if the evidence cannot, name the
  unresolved alternative and report the walk as inconclusive.
- Several bugs look like one symptom. After each fix, rerun the original reproduction and
  the walk: the next layer's error was hidden under the first.
- A cause upstream of the named dataset is reported as such; the fix goes where the cause
  is only when the ticket allows it, and the diff says what that moves.
- A load, a schedule, a cache and an extract are layers too: a measure that is right in the
  warehouse and wrong in the report may have moved in the refresh.
- Bisection in time is the same method: the mismatch rate over a date column shows when an
  incremental model was backfilled wrongly.
