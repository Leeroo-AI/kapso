---
name: data-diff
description: The before-and-after comparison that reviewers rely on, with the audit-first method for a refactor, fingerprints of every modified dataset and its downstream (row count, numeric sums, distinct counts) as a screening step, the keyed comparison that proves equality, the replay comparison, why lineage says where to look but only the diff says whether it is right, and a script for dbt projects. Use before any refactor, after every change before writing the proof, in verification to check the proof, and to show that a rerun does not change results.
---
# Data diff

Structural assertions can miss semantic changes: a currency column where both values are
valid passes every test while a change flips the currency. Pair the declared invariants
with a comparison of the affected outputs, screened by fingerprints and proven by key.

## The audit-first method for a refactor

Write the comparison before changing anything, confirm that the old and the new agree,
change one small piece, run, compare, commit, repeat. Compare columns, types and
materialization first, then rows: a varchar against a bigint blocks the row comparison.
The comparison reports rows only in A, rows only in B, and per-column mismatches side by
side, keyed by the dataset's grain after its uniqueness is proven.

## The script

On a dbt project:

```
python3 $KAPSO_HARNESS_ROOT/skills/data-diff/scripts/fingerprint.py --select "<model>+" \
  [--project-dir .] [--profiles-dir DIR] [--target NAME] --save results/fingerprints_after.json
python3 $KAPSO_HARNESS_ROOT/skills/data-diff/scripts/fingerprint.py \
  --compare results/<ticket>/fingerprints_reference.json results/<ticket>/fingerprints_after.json [--json]
python3 $KAPSO_HARNESS_ROOT/skills/data-diff/scripts/fingerprint.py \
  --except "{{ ref('merchant_payments_daily') }}" "audit.merchant_payments_daily_before" [--project-dir .] [--target NAME] [--save FILE]
```

`--save` fingerprints every dataset the selector lists (the dataset and its downstream
closure by default): the row count, the rounded sum of every numeric column, the distinct
count of every other column, through `dbt show`, with the measurement context. The
datasets are measured together, a group per statement, and one by one where a group's
statement fails. For a ticket, the validation-report skill's `evidence.py prove` takes
these fingerprints on the build and on the reference target (what production holds) and
compares them; `--with-downstream` rebuilds what reads the changed dataset first. `--compare`
prints the datasets whose fingerprint changed, with the columns that moved, and the
datasets added or removed; a measurement that failed or was not built is an error, never
"unchanged", and counts are compared exactly. A dataset is recorded as absent only when
the adapter itself says no such relation exists; absent on one side is a change.
`--except` is a set comparison: it counts the rows of each relation that the other does
not hold, with both row counts and both distinct-row counts, and it shows row equality
only when neither side holds a duplicate row; a count that does not come back stops it.
It is the proof for a refactor, a migration or a replay, and `--save` writes the report
the label reads, which refuses a comparison with duplicate rows.
On another stack the same numbers, taken with the engine's client over the consumer
list from the orchestrator's graph, give the same comparison.

## Reading the comparison

- Unequal fingerprints identify a change; equal fingerprints do not establish equality.
  For a refactor, a migration or a replay claim, first prove key uniqueness, then compare
  outputs by that key (`--except`, or the project's audit helpers) with null-safe
  per-column comparisons and the keys missing on either side.
- The named dataset changed: expected. Its row count moved: the change touched the grain or
  the exclusion; say which in the proof.
- A downstream dataset changed: expected when it consumes the fixed column; each one goes
  into the proof with the reason. One you cannot explain is a second bug or a symptom
  patch.
- Nothing downstream changed while the dataset did: the consumers do not read the fixed
  column, or they are tables that were not rebuilt. Rebuild the downstream set with
  selection before fingerprinting; a view reflects the change without a rebuild, a table
  does not.
- A rerun on identical inputs that differs disproves idempotence (pipeline-reliability);
  equal fingerprints after a rerun need the keyed comparison before idempotence is claimed.
- Two fingerprints taken by hand are comparable only when the downstream set was rebuilt
  in both states, in the same measurement context. `prove` rebuilds the downstream set
  once, after the change, and screens it against what production holds: production is
  the reference, not a second rebuild, so a production table last built before its
  inputs last moved shows as changed for a reason that is not the ticket's. Name that
  reason in the proof where it applies.

## Lineage says where, the diff says whether

Check whether the lineage provider includes indirect dependencies through joins, filters
and grouping; when that coverage is unknown, expand the impact assessment to every
consumer of the relation. Lineage identifies possible impact; measured comparisons
establish what changed. Use lineage to choose what to diff, never as proof that a change
is safe.

## Limits

Sums and distinct counts miss a change that preserves them (two rows swapping values);
that is what the row comparison is for, and why equal fingerprints never stand in for
it. The grouped measurement is written for Snowflake and DuckDB and falls back to one
dataset at a time elsewhere. Every uninspected column and consumer is listed in the
proof, never counted as unchanged.
