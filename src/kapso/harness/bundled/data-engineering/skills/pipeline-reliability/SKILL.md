---
name: pipeline-reliability
description: What makes a pipeline rerunnable and complete, with a task as a pure function of code and partition, one logical effect per retry and the write shapes that give it, write-audit-publish, exactly-once as a sink property, lateness as a declared window, backfills through the production code path, duplicates from at-least-once delivery, partitions and schema evolution, orchestration traps, and the replay protocol that proves a load. Use for a ticket about reruns, duplicates, missing late rows, incremental loads, backfills or scheduling, and before labelling any change that touches a load.
---
# Pipeline reliability

## A task is a pure function of code and partition

The same input gives the same output on every rerun; a job treats its input as immutable
and completely replaces its output, which buys inspection of every stage, provenance and
cheap recovery (fix the failed job in a chain and restart there). A task never reads "the
latest available data" but the partition keyed by its interval, never lets the wall clock
drive a critical computation, and leaves no partial output. A retry must have one logical
effect: partition replacement publishes atomically; delete-and-insert commits together or
recovers without exposing a partial partition; a keyed merge resolves duplicates and
version order deterministically. Unprotected append on replay duplicates output; an
append protocol with an enforced operation identity can be retry-safe, and a plain
`INSERT` on a rerun path is a question to answer before anything else.

## Audit before publish

After a build-then-test, the corrupted data has already shipped. Write to an audit
location (a staging table, a branch, a hidden snapshot), run the checks, then publish by
swapping or fast-forwarding. Check completeness against the expected input population
and the transformation's declared exclusions: an unexplained drop is a failure, a
legitimate reduction is not. On corruption the order is stop the bad data entering first,
then reprocess, because stale data is almost always better than incorrect data.

## Exactly-once is a sink property

Transports guarantee exactly-once only within their own scope; a call to a remote store,
a replayed offset, a lost response all break it. Assume at-least-once everywhere and make
the sink idempotent: apply by key, latest version wins, with an end-to-end operation id
under a uniqueness constraint. For change-data-capture the consumer's job is "apply by
key, latest wins", not suppressing every duplicate; duplicates inside one batch are
deduplicated too, or history-only deduplication lets both copies land.

## Lateness is a declared window

Record the engine's watermark definition, the operator that uses it and its late-event
policy; the guarantee is one-sided (data inside the window is kept, later data may or may
not be) and its exact semantics belong to the engine, not to a rule of thumb. For a batch
lookback, declare which timestamp selects rows and how corrections older than the
lookback are recovered; size the window from measured lateness, monitor it, and test both
sides of the boundary, since `>` against `>=` at the edge duplicates or drops it.
Late-arriving dimensions: a fact whose dimension row has not arrived is dropped by an
inner join; keep the fact with an unknown member and bound the orphan rate.

## Backfills, partitions, schemas, orchestration

- A backfill is bounded by a partition range, named in the ticket, and runs through the
  production code path (the orchestrator's catch-up or backfill command over the missed
  intervals); a backfill with different logic leaves a seam that invariants show as a step
  change at the boundary date. Follow a backfill with count and sum reconciliation across
  the backfilled and untouched ranges.
- Filter on the partition column itself; the partition's time zone is the data's. After
  a replay compare file counts, sizes and logical row identities: extra files alone do
  not show duplicate logical output.
- A column added mid-history is null before its birth date; a renamed source field
  silently nulls the target; a compatibility mode dictates the upgrade order (backward:
  consumers first; forward: producers first).
- A sensor sees partial files unless the writer renames at the end; a schedule in local
  time runs twice or never on the two clock-change days; a dependency is declared, not
  assumed.

## The replay protocol that proves a load

A repeated unchanged run proves retry stability only for the state it exercised; a
pipeline that consistently ignores late corrections or deletes passes it. On frozen
inputs, compare the incremental output with an independent full recomputation by key
(data-diff `--except`). Then exercise: duplicate delivery within one batch and across
batches; equal timestamps at the boundary; a late insert; an update to an old event; an
out-of-order update; a delete followed by replay of an older update; an interruption
before publication. Require identical final business rows and one logical effect per
input; document intentionally variable audit metadata separately. Row count equals the
distinct key count; no source row with an event time inside the lookback is absent from
the target; the orphan rate is bounded and explained; no step change at the backfill
boundary; the two clock-change days present; partitions per day stable across a rerun.

The proof of a changed load names the row comparison of one replay (`rerun`): keep a
copy of the load's output, run the load again on the same inputs, and compare the two
with data-diff's `--except ... --save`. The computed label reads that one report and
asks of it: no row on one side only, the same row count, no duplicate row on either
side, taken after the ticket's baseline. The rest of the protocol it does not check:
the cases above that apply to the load are run by hand and listed under `validated`,
the ones not run under `skipped` with the reason, and verification reads both. The
runbook says what the first run in production needs (validation-report).
