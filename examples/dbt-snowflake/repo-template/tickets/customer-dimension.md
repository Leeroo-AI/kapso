# Customer dimension: numbers do not reconcile with upstream data

**Area:** customer · **Reported by:** Analytics · **Kind:** fix an existing model

## What is wrong

The analytics team flagged the `dim_customers` model as producing incorrect results. The numbers do not
reconcile with upstream data. There may be more than one issue.

## What we need

Find and fix the `dim_customers` model in the mart layer. Investigate the data lineage and compare the model's
output against its upstream sources. Fix every issue you find.

- Model: `warehouse/models/marts/core/dim_customers.sql`
- Output: `main.dim_customers`

## Done when

- All customer-level metrics reconcile with the upstream source data.
- The ticket check `customer-dimension` passes.

## Constraints

- Do not modify upstream staging models.
- Do not change the model's materialization type.
- Do not remove or rename existing columns.
