# Repeat purchase cohort revenue: rebuild the mart

**Area:** revenue · **Reported by:** Growth analytics · **Kind:** rebuild a report as a new model

## What is wrong

The repeat purchase cohort revenue mart needs repairing. It is rebuilt as a new model beside the old one, so
that consumers can switch over once it reconciles.

## What we need

One model, in the warehouse project:

- Model: `warehouse/models/marts/customer/rpt_repeat_purchase_cohort_revenue_fixed.sql`
- Output: `main.rpt_repeat_purchase_cohort_revenue_fixed`, a table
- Source: `main.int_sales__orders_enriched`, which is already built in the warehouse

### Output columns

- `cohort_month` (DATE): first purchase month (first day of the month)
- `months_since_cohort` (INTEGER): 0 for the cohort month; never negative
- `cohort_size` (INTEGER): distinct customers per cohort (constant within a cohort)
- `active_customers` (INTEGER): distinct customers ordering in the months-since bucket
- `orders` (INTEGER): distinct orders in the bucket
- `revenue` (DECIMAL(18,2)): bucket revenue
- `cumulative_revenue` (DECIMAL(18,2)): running revenue per cohort ordered by `months_since_cohort`

### Business rules (strict)

- Source relation: `main.int_sales__orders_enriched`. This table may contain several rows per `order_id`.
  Deduplicate to one row per order before computing cohorts. When deduplicating, aggregate: `MIN(customer_id)`
  to resolve conflicting customer ids, `MIN(ordered_at)` for timestamps, and `SUM()` for monetary values (to
  preserve total revenue). This keeps total revenue reconciled with the source.
- Cohort definition: `cohort_month = date_trunc('month', min(ordered_at))::date` per `customer_id`.
- NULL `customer_id`: some orders have a NULL `customer_id`. These orders must still be included in the total
  `orders` counts and `revenue` totals (for reconciliation against the source), but must not count toward
  `cohort_size`, which counts real customers with a non-NULL `customer_id` only. Consider assigning synthetic
  customer keys to the NULL rows so that they stay in the dataset.
- Per-order fields:
  - `order_month = date_trunc('month', ordered_at)::date`
  - `months_since_cohort = DATEDIFF('month', cohort_month, order_month)`
  - Ignore orders with a NULL `ordered_at`; keep `grand_total` as it is (no negatives expected).
- Aggregation grain `(cohort_month, months_since_cohort)`:
  - `active_customers = count(distinct customer_id)`
  - `orders = count(distinct order_id)`
  - `revenue = cast(sum(grand_total) as decimal(18,2))`
- `cohort_size = count(distinct customer_id)` per `cohort_month`, attached to every bucket.
- `cumulative_revenue = sum(revenue) over (partition by cohort_month order by months_since_cohort rows
  unbounded preceding)`.

### Quality and invariants (enforced)

- No NULLs in `cohort_month` or `months_since_cohort`; `months_since_cohort >= 0`.
- `active_customers <= cohort_size` for every row.
- `revenue >= 0` and `cumulative_revenue >= revenue`.
- `cumulative_revenue` is non-decreasing within a cohort.
- For each cohort, `cohort_size` equals the number of distinct customers in that cohort's first-order month
  (recomputed from the source).
- Global reconciliation against the source: total `orders` matches to within 1; total `revenue` matches within
  0.5% relative error.
- Cumulative check: for every cohort and month bucket, `cumulative_revenue` equals the sum of all prior-or-equal
  bucket revenues (tolerance 0.01).
- Idempotent and deterministic: reruns do not change totals.

## Notes

- Reference the source in the `main` schema with `adapter.get_relation()`.
- Do not modify the staging models; rely on the source as it is built in the warehouse.
- Avoid fan-out: aggregate once, and window over the aggregated data.
- The model materializes in the `main` schema, the default schema of the warehouse project's profile.

## Done when

- The model builds and every rule and invariant above holds.
- The ticket check `repeat-purchase-cohorts` passes.
