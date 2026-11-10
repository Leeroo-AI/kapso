---
name: warehouse-cost
description: Make a warehouse workload or a pipeline cheaper or faster the way an expert does, with the task's objective and the engine's billing model identified separately, attribution before optimisation, the three places money goes (idle time, recomputation, data nobody reads), pruning verified on the engine's plan, accelerators judged by a measured break-even, plan reading, and the honesty rules for a speed or spend claim. Use for a cost or speed ticket, in setup to record the baseline, and before labelling any change that alters materialization, partitioning or clustering.
---
# Warehouse cost

## First question: what is asked, and what does the engine bill?

Identify the task's objective and the billing model separately: latency, throughput,
bytes processed, allocated compute time or storage. Record the applicable minimums, idle
allocation and concurrency: one engine bills compute-seconds with a per-resume minimum and
an idle tail until auto-suspend, so a short query that wakes a warehouse pays for minutes
and two concurrent queries share what two serial ones pay twice; another bills bytes in
the selected columns with a per-table minimum, so `LIMIT` saves nothing on an unclustered
table and `select *` by ten users is the whole table ten times; an always-on cluster
bills the same whatever the query does unless it shrinks. A latency improvement is a
speed result; call it a cost saving only when the billed quantity and the applicable
price support the claim. Write both in the domain statement and measure both.

## Attribute before optimising

Build query-level attribution (tags, the query history joined to the metering history,
idle time allocated across each hour's queries) before any change; the distribution is
skewed and the top few queries, tables or definitions carry most of the cost. On a
transformation project, measure the orchestration overhead too: one task per model can
spend most of the compute on parse and startup, and batching is the fix.

## Where the money goes

- **Idle time**: a warehouse billed while nothing runs; measure billed idle time and
  resume overhead before choosing an auto-suspend.
- **Recomputation**: an incremental merge with no bound on the destination scans the
  whole table on every run; a view re-read by every consumer and every schedule; an
  unbounded view over years of history read daily.
- **Data nobody reads**: tables maintained for consumers who left; find them in the query
  history before paying for them again.

## The levers, in order of yield

1. Stop re-reading: materialize an input where measured reuse justifies it; bound reads
   by the range the consumer needs, without discarding history it requires; share the
   input across definitions.
2. Partition and cluster by the filter actually used, and verify pruning on the actual
   engine and predicate with the plan or the dry-run estimate: prefer predicates the
   optimiser can map to partition or clustering boundaries; which casts and functions
   keep pruning varies by engine. Clustering suits large, read-mostly tables with
   selective frequent queries and few columns from low to high cardinality, and its
   maintenance is billed, so measure it before adding it.
3. Pre-aggregate the many side before joining (fan-out before aggregation is a cost bug
   and a correctness bug at once); decorrelate subqueries the planner cannot; give the
   planner statistics and memory before touching cost knobs.
4. Materialize heavy shared views as tables, incremental on the big facts, sorted or
   clustered by date; a view is recomputed by every consumer, so count the consumers.
5. Accelerators last, each judged by a measured break-even on the workload's own
   queries: a price premium needs a time reduction to match it, and per-query gains do
   not add up to a bill reduction; materialized views carry maintenance and staleness.
6. Settings last of all: thread counts and compilation flags, measured inside the
   run-to-run spread.

## Reading a plan

The first thing to look for is whether estimated row counts are close to reality; a sort
or a hash that spilled to disk; a nested loop over a large input; a scan where a prune was
expected. Results on toy tables do not transfer.

## The honesty rules for a speed or spend claim

Compare paired runs on the same input snapshot and equivalent resources; record every
measured run, the concurrency, the cache state, the result-transfer boundary and the
billed quantity. Measure cold and warm behaviour separately when both matter in
production. If variability obscures the effect, collect more observations or report the
result as inconclusive. Prove output equivalence by key before claiming a
result-preserving optimisation (data-diff), and disclose materialization and maintenance
costs. Pre-computed answers keyed to known queries, sampling that changes results, stale
caches and a different engine returning different numbers are not speedups.
