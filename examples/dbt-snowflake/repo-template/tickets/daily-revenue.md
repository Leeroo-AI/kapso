# Daily revenue report: missing days and a wrong running total

**Area:** revenue · **Reported by:** Finance · **Kind:** fix an existing model

## What is wrong

Finance tracks daily revenue from `rpt_order_daily_summary`. The report has data quality problems: some
expected records are missing, and calculations appear incorrect.

## What we need

Fix the `rpt_order_daily_summary` model so that it produces correct output.

- Model: `warehouse/models/marts/sales/rpt_order_daily_summary.sql`
- Output: `main.rpt_order_daily_summary`
- Source data: `stg_orders__orders`

## Done when

- The output is complete and accurate.
- Every date in the range is present (no gaps).
- Cumulative revenue is calculated correctly.
- The ticket check `daily-revenue` passes.

## Constraints

- Do not modify upstream staging models.
- Do not change the model's materialization.
- Preserve all output columns.
