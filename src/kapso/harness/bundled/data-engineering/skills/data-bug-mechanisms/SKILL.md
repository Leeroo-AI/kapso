---
name: data-bug-mechanisms
description: The mechanisms behind silently wrong data, with cardinality, null semantics, status, time semantics, ordering and frames, and the data read wrong, each with how it shows, the revealing query, the usual cause in the code, the shape of the fix and the invariant that proves it, and how a symptom maps to them. Use when a ticket describes a symptom, when choosing the next cause to test, when reading a suspect node's SQL, and when writing the invariant spec for a change.
---
# Data bug mechanisms

Silently wrong SQL is long in its catalogue and short in its mechanisms. Work the first
three whenever a total disagrees with its source; the revealing queries are cheap and run on every
engine.

| Mechanism | How it shows | Revealing query | Usual cause | Fix shape | Proving invariant |
| --- | --- | --- | --- | --- | --- |
| **Cardinality**: fan-out and chasm traps, de-duplicating unions, non-deterministic row numbers, replay duplicates | duplicate keys; totals above the source; counts that grow with a join; numbers that change on rerun | `select key, count(*) from t group by key having count(*) > 1`; row count before and after each join; `union` where `union all` was meant; run twice and compare | a join on a non-unique key (variants per product, lines per order, versions per record); `row_number` over a non-unique order; an append on retry; a de-duplication against history that misses duplicates inside one batch | aggregate the many side before the join or join on the full key; a deterministic tie-breaker; merge by key; de-dupe within the batch too | unique(grain); reconciliation to the source total; keyed comparison stable across two runs |
| **Null semantics**: outer-join filters, `NOT IN` with nulls, counts and sums over nulls, coalesce masking a bug | rows missing; counts below the source; zero-activity entities gone; a rate wrong only where a value is absent | anti-join the source key against the output; read every predicate on a left-joined column; `not in` against a column that can be null returns nothing | a `WHERE` on the non-preserved side of a left join; `7 = NULL` is null, so `=` and `<>` both miss; `count(col)` skips nulls and `sum` of all nulls is null; `> 0` or `is not null` dropping valid zero or null rows | move the predicate into the join; `is distinct from`; coalesce by business rule, not to hide a join | completeness(source key in output); row count equals the filtered source; not-null on keys |
| **Status semantics** (a cardinality-and-filter special case) | totals include cancelled, refunded, test, draft or soft-deleted rows | `select status, count(*), sum(amount) from source group by status` | no status predicate, a value list missing a spelling, a soft-delete flag ignored | one explicit predicate from the source's own vocabulary | reconciliation to the source filtered the same way |
| **Time semantics**: session-zone casts, inclusive `BETWEEN`, DST, mixed formats, watermarks, overlapping validity | daily totals wrong at the day boundary; times collapsed to the start of the day; nulls after a cast; two sources off by hours; a day with zero or double duration; yesterday's late rows missing | `extract(hour from ts)` distribution; parsed nulls against non-null raw; the same event's timestamp in two sources; counts per hour across the clock-change day; overlap test on validity windows | casting to a date in the session's zone; `BETWEEN` double-counting a boundary; a strict `>` on the watermark; an offset taken from a zone name instead of the event date; a cast that drops the time; a partition in another zone | store UTC and convert at the edge; half-open ranges; a lookback window sized from observed lateness; parse with the engine's null-on-failure form and count what it left null | the hour-of-day distribution matches the source's; parsed nulls only where raw is null; alignment within the offset; expected period coverage; completeness within the lookback |
| **Ordering and frames**: default window frames, ties, cumulative sums, gaps | a running sum that decreases or jumps at ties; the last cumulative differs from the total; days missing when nothing happened | lag over (partition order) finding decreases; last cumulative against the sum; lead over the date finding differences above one day | default frames depend on the engine and the function (peer-sensitive cumulative aggregates and full-partition value functions need an explicit frame); no partition; ties without a tie-breaker; aggregating only the rows that exist | order by (date, id); an explicit `ROWS` frame; a date spine with coalesce to zero | monotonic only where increments are non-negative; successive differences equal the increments; final equals total; expected entity-period coverage |
| **The data read wrong**: a cumulative summed, a code misunderstood, integer division, a rank or percentile over the wrong population, conditions evaluated in the wrong order | plausible numbers an order of magnitude off; a category missing or dominant; a ratio truncated to 0 | recompute the measure from a sample by hand; the distribution over the categories; recompute the rank or percentile in SQL and compare row by row | `sum` over a column that is already cumulative; integer operands; a ranking population, null-ordering rule or percentile function inconsistent with the measure's definition; `CASE` branches out of the stated order; `<` for `<=` | read the column's definition before aggregating it (the final observation per entity and reset period; `max` only after proving monotonicity); cast before dividing; identify the function's null semantics, then implement the defined population and tie behaviour; branches in the stated order | custom recomputation agreement; bounds; structure preserved (columns, types, materialization) |

## Reading a symptom into mechanisms

- A total that disagrees with its source: cardinality, then null semantics, then status.
- A figure wrong only near a period boundary, or records that arrive late and never show:
  time.
- A value outside its possible range (non-finite, negative, a ratio above one): the data
  read wrong, then nulls.
- Rows absent from the output: null semantics, then history when the dataset is a snapshot,
  then the incremental boundary when the load is incremental (pipeline-reliability).
- A result that changes between runs on the same input: cardinality from replays; one
  logical effect per retry.
- A running or period-over-period measure that is off: ordering and frames, then the grain
  of the partition key.
- Parts that do not add up to their whole: conservation, then parsing of the key the parts
  are made on.

After a fix, rerun the original reproduction and the independent checks; any remaining
discrepancy needs another explanation, and the walk in lineage-bisection finds it.
