# Return reconciliation report: invalid values, missing returns, and a new velocity tier

**Area:** revenue · **Reported by:** Operations · **Kind:** fix an existing model and add a column

## What is wrong

Operations rely on `rpt_return_reconciliation` to track how efficiently returns are processed and to find
bottlenecks in the refund process. The model currently has issues that put invalid data into reports:

1. Some calculated metrics show invalid values (infinity, NaN) in production reports.
2. Row counts do not match expectations: some returns are missing from the output.
3. A new classification column is needed to prioritise improvements to return processing.

## What we need

Fix the `rpt_return_reconciliation` model so that it produces correct output.

- Model: `warehouse/models/marts/sales/rpt_return_reconciliation.sql`
- Output: `main.rpt_return_reconciliation`
- Sources:
  - `stg_orders__returns`: return headers, with identifiers, customer details, return metadata, monetary
    amounts and timestamp fields. Explore the table to see the columns available.
  - `stg_orders__return_lines`: return line detail. Explore it to find the columns relevant for aggregation.

### 1. Fix the data quality issues

The model should include all returns from the source table, with no invalid numeric values:

- no infinity or NaN values in any calculated column;
- all returns included in the output (check whether any filter excludes valid data);
- every ratio and rate calculation handles its edge cases.

### 2. Add a processing efficiency classification

Operations want a new column, `refund_velocity_tier`, that classifies returns by how efficiently they were
processed, to identify the returns that experienced delays.

Use a waterfall classification: check the conditions in order and assign the first match.

1. `critical_delay`
   - bottom 25% slowest processors (`PERCENT_RANK <= 0.25`, that is the 25% with the highest
     `total_processing_days`)
   - and took more than 10 days to process
2. `needs_improvement`
   - bottom 50% slowest processors (`PERCENT_RANK <= 0.50`)
   - or refund completion rate below 80%
3. `acceptable`
   - refund completion rate >= 85%
   - and total processing days <= 7
4. `efficient`
   - top 25% fastest processors (`PERCENT_RANK >= 0.75`)
   - and refund completion rate >= 90%
5. default: all other cases are `acceptable`

Implementation notes:

- Use the `PERCENT_RANK()` window function ordered by `total_processing_days DESC`.
- Calculate percentiles only for non-NULL processing days.
- NULL handling in the tier classification:
  - NULL `refund_completion_rate`: treat as 0 (worst case).
  - NULL `total_processing_days` in the `> 10` check: treat as 0 (best case, does not trigger critical).
  - NULL `total_processing_days` in the `<= 7` check: treat as a very high value (worst case, does not pass).
- Check the conditions in the order listed above (waterfall logic).

## Expected output columns

`return_id`, `return_number`, `order_id`, `customer_id`, `return_type`, `refund_method`, `status`,
`total_return_amount`, `total_refund_amount`, `return_line_count`, `requested_at`, `received_at`,
`processed_at`, `days_to_receive`, `days_to_process`, `total_processing_days`, `refund_completion_rate`,
`processing_efficiency`, `refund_velocity_tier` (new).

## Done when

1. The model compiles and runs without errors.
2. No numeric column holds an infinity or NaN value.
3. The row count matches the number of returns in the source table.
4. The new tier column holds valid values distributed across more than one category.
5. The classification distinguishes fast processors from slow ones.
6. The ticket check `return-reconciliation` passes.
