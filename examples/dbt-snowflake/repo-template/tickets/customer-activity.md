# Customer activity table for churn analysis

**Area:** customer · **Reported by:** CRM analytics · **Kind:** new model

## What we need

Build a dbt model called `fct_customer_activity` that tracks customer order history for churn analysis.

- Model: `warehouse/models/marts/customer/fct_customer_activity.sql`
- Output: `main.fct_customer_activity`, a table
- Source: `stg_orders__orders`

### Columns

- `customer_id`: customer identifier
- `email`: customer email address
- `order_id`: order identifier
- `order_date`: order date (DATE type, not a timestamp)
- `previous_order_date`: the date of this customer's previous order (DATE type). Use the DATE-typed
  `order_date` in the window function, not the raw timestamp.
- `days_since_last_order`: days between consecutive orders (from the DATE values)
- `is_reactivated_order`: true if the customer had been inactive for over 90 days

One row per order.

## Notes

- Make the ordering deterministic when several orders share the same date.

## Done when

- `main.fct_customer_activity` exists as a table with the columns above, one row per order.
- The ticket check `customer-activity` passes.
