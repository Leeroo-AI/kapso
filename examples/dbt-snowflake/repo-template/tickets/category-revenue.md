# Category performance report: suspicious numbers

**Area:** revenue · **Reported by:** Retail analytics · **Kind:** fix an existing model

## What is wrong

The retail analytics team has flagged `rpt_category_performance` as producing suspicious numbers. Several
stakeholders raised concerns during recent reviews, but nobody has pinpointed exactly what is wrong. The report
is used for monthly business reviews and category-level planning.

## What we need

Investigate the `rpt_category_performance` model, identify the root cause of the data quality issue or issues,
and fix the model.

- Model: `warehouse/models/marts/product/rpt_category_performance.sql`
- Output: `main.rpt_category_performance`
- Source data: the model references several staging tables; explore the warehouse and the model's SQL to
  understand how they relate.

## Done when

- The model produces accurate results that are consistent with the underlying source data.
- The ticket check `category-revenue` passes.

## Constraints

- Do not modify upstream staging models.
- Do not change the model's materialization.
- Preserve all output columns.
