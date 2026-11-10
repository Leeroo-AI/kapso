# Daily cohort retention report

**Area:** customer · **Reported by:** CRM analytics · **Kind:** new model

## What we need

Build a dbt model called `rpt_daily_cohorts` that analyses customer retention by grouping customers into
cohorts by the date they first placed an order.

- Model: `warehouse/models/marts/analytics/rpt_daily_cohorts.sql`
- Output: `main.rpt_daily_cohorts`, a table

### Output columns

- `cohort_date`
- `cohort_size`
- `day_0_customers`
- `day_7_customers`
- `day_30_customers`
- `day_90_customers`
- `day_0_retention`
- `day_7_retention`
- `day_30_retention`
- `day_90_retention`

The `day_N_customers` columns are cumulative: each is the count of customers who made a purchase within N days
of the `cohort_date` (inclusive). For example, `day_7_customers` counts every customer with at least one order
within 7 days of their cohort date, which always includes all the day-0 customers. So
`day_7_customers >= day_0_customers`, and so on.

## Data sources

The warehouse project already has staging models. Look through them to find the right source of orders, the
columns it offers, and how to use them for the cohort analysis.

## Notes

- Materialize the model as a table.
- Think carefully about data types and edge cases in the calculations.

## Done when

1. The model compiles and runs without errors.
2. The retention metrics are logically consistent.
3. Each cohort date appears exactly once in the output.
4. The total number of customers across all cohorts accounts for every customer in the source data.
5. The ticket check `daily-cohorts` passes.
