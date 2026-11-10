# Customer history snapshot is wrong; add a current-customer dimension on top of it

**Area:** customer · **Reported by:** Analytics · **Kind:** fix a snapshot and build a new model

## What is wrong

The analytics team reports data quality issues in the customer snapshot's output. The snapshot is supposed to
implement SCD Type 2 tracking of customer data, but several problems have been identified. They affect both
the snapshot's configuration and the completeness of the data it holds.

The team also needs a `dim_customer_current` model: one row per customer, with analytical columns derived
from the snapshot's history. It must be a standard dbt model (not a snapshot) that reads from the corrected
snapshot.

## What we need

### Part 1: fix the customer snapshot

Investigate the customer snapshot to identify all of its problems. Examine both its configuration and the data
it produces. There may be more than one issue.

- Snapshot: `warehouse/snapshots/customer_snapshot.sql`
- Output: `snapshots.customer_snapshot`

### Part 2: create `dim_customer_current`

After fixing and running the snapshot, create a new dbt model called `dim_customer_current` that reads from
the `customer_snapshot` snapshot and produces a single-row-per-customer analytical table. Place it in the
warehouse project's `models/` directory.

The model includes all columns from the current snapshot record, plus these derived columns:

1. `days_since_last_update`: the number of days between `dbt_valid_from` and the current date, computed only
   for the current record (where `dbt_valid_to IS NULL`). Use `DATEDIFF`.
2. `total_versions`: the total count of snapshot versions (rows) that exist for each customer across all
   history. This needs an aggregation over the full snapshot table grouped by `customer_id`.
3. `is_recently_changed`: a boolean that is `true` when the customer's current record was created (its
   `dbt_valid_from`) within the last 30 days, and `false` otherwise.

The model includes only current records (where `dbt_valid_to IS NULL`) and produces exactly one row per
`customer_id`. It is built with `dbt run` after the snapshot completes.

## Data sources

- `stg_customers` (in the `main` schema): the customer data
- `customer_snapshot` (in the `snapshots` schema): the SCD Type 2 snapshot fixed in part 1

## Done when

1. The snapshot tracks every customer from the source.
2. The `dim_customer_current` model exists with the right columns and data.
3. `dim_customer_current` has exactly one row per customer.
4. Every derived column computes correctly (non-negative days, positive version counts, consistent boolean
   logic).
5. The ticket check `customer-snapshot` passes.
