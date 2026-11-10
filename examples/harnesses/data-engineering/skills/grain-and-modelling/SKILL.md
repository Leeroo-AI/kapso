---
name: grain-and-modelling
description: The modelling canon a senior data engineer applies before touching a dataset, with the grain as a declared contract, fact-table types and what may be summed across which dimensions, the two prohibitions that prevent fan-out and double counting, keys, nulls, history and time, the layers that apply each transformation once, and the materialization ladder with the pitfalls of incremental loads and history tables. Use in setup to declare the grain, in implementation before changing a dataset's shape, and when a ticket reads "does not reconcile".
---
# Grain and modelling

## The grain is a contract

Declare the output grain and test its key before changing joins or aggregation: what one
row means, the key named after it (`user_per_day_id`), and the first check on any dataset,
that the key is unique and not null. Grains are never mixed in one table. A deliberate
grain change is a migration: its consumers move with it, and the before and after carry
separate contracts.

## Three fact types, and what may be summed

Transaction facts (one row per event), periodic snapshots (one row per entity per period,
dense even when nothing happened) and accumulating snapshots (one row per process,
revisited as it progresses) are the three; which one is being loaded says whether updates
in place are legitimate. For each measure declare the dimensions it is additive across.
Across time a balance may need the closing value, an opening value or a time-weighted
average according to its definition; a cumulative needs the final observation of its reset
period; a ratio is stored as its additive components and computed last. A column that is
already cumulative is never summed again.

## The two prohibitions that prevent the commonest wrong numbers

- Before joining two fact datasets, prove each side's multiplicity at the join grain. When
  either side has several rows per key, aggregate each to the common grain first, then
  join, and reconcile each measure before and after.
- Every dimension attached to a fact is single-valued at the fact's grain; a legitimately
  multivalued one goes through a bridge table constrained to a moment in time.

The failures these prevent run without any error: a fan trap multiplies one parent's
measure down a chain of one-to-many joins; a chasm trap multiplies two independent
many-side branches through a shared one-side table.

## Keys, nulls, history, time

- Dimension keys are the warehouse's own surrogates. Declare whether a missing measure
  means unknown, absent or zero, and test the intended treatment in sums, counts and
  denominators; a fact whose dimension is missing keeps an explicit Unknown member rather
  than a null key, because engines group and constrain on nulls inconsistently.
- For a history-bearing attribute, declare whether reports need the value at event time or
  the current value, and test that late facts select the intended version.
- Late-arriving facts are matched to the dimension rows effective when the event occurred;
  late-arriving dimensions get a placeholder row completed later. Store UTC and convert at
  the edge; keep the event's zone or offset where its interpretation needs it, and resolve
  offsets from the event date and the zone's rules, never from the zone name alone.

## Layers exist to apply each transformation once

A landing layer that keeps the source as-is with load metadata; a cleaning layer
one-to-one with sources, since re-graining there loses source data that will be needed;
a layer that re-grains and isolates joins; and consumer-facing datasets, one per
concept. Several consumers of one intermediate is a signal to look at its grain. These
are the worked project's conventions (dbt-workflow has them in dbt terms); preserve the
task's architecture when it differs.

## The materialization ladder

A view, until it is too slow to query; a table, until it is too slow to build; incremental
only then. Views recompute at every query, tables trade storage for build compute, and
incremental loads carry the largest documented correctness risk: the filter must be valid
whether or not the load is incremental; a null or non-unique key duplicates rows or fails
the merge; a schema change leaves new columns empty for old rows unless backfilled; an
unbounded merge scans the whole destination; a fixed lookback misses rows later than it
forever, and the honest options are a longer window, a periodic full refresh, or
independent idempotent batches with an explicit lookback. A history table cannot be rebuilt
from configuration: its key must be unique, its validity intervals must not overlap, hard
deletes need an explicit policy, and it runs on a schedule or records nothing.
