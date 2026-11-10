# Cumulative product sales tracking

**Area:** inventory · **Reported by:** Merchandising · **Kind:** new model

## What we need

Build a dbt model called `fct_inventory_balance` that tracks cumulative sales for products over time.

- Model: `warehouse/models/marts/inventory/fct_inventory_balance.sql`
- Output: `main.fct_inventory_balance`, a table
- Schema file: `warehouse/models/marts/inventory/schema.yml`, defining a test that `cumulative_revenue` is
  never negative

### Columns

- `product_id`: product identifier from the products table
- `product_name`: product name from the products table
- `order_date`: order date (cast the `ORDERED_AT` column of the orders table to DATE)
- `order_line_id`: order line identifier
- `units_sold`: quantity from order lines (use `quantity_ordered`)
- `revenue`: line total from order lines
- `cumulative_units_sold`: running total of units per product up to and including the current date
- `cumulative_revenue`: running total of revenue per product up to and including the current date
- `prev_day_cumulative_revenue`: the maximum cumulative revenue from the previous calendar date for this
  product (NULL for the first date). All rows on the same date show the same previous-day value.
- `daily_change`: the difference between the current cumulative revenue and the previous day's
- `rank_in_day`: ranking of order lines within the same product and date, ordered by `order_line_id` ascending
- `days_since_last_sale`: number of days between the current `order_date` and the previous distinct
  `order_date` for this product (NULL for the first date). Use LAG over distinct (`product_id`, `order_date`)
  pairs, not over raw rows.
- `running_avg_revenue_per_line`: cumulative revenue divided by cumulative units sold (handle division by zero)
- `cumulative_percentile`: `PERCENT_RANK` of the product's current cumulative revenue compared with all
  products' current cumulative revenue on the same date

### Rules

- **Data filtering:** only include order lines whose order status is not 'cancelled' or 'refunded'. Check the
  `status` column of the orders table.
- **Join path:** use the staging tables `stg_orders__order_lines`, `stg_orders__orders`,
  `stg_product__product_variants`, `stg_product__products`. Join order lines to orders on `order_id`, then to
  product variants on the order line's `variant_id`, then to products on the product variant's `product_id`.
- **Window functions:**
  - For the cumulative columns, use `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` so that same-date
    values match.
  - For `prev_day_cumulative_revenue`, calculate the maximum cumulative from the previous date with a self-join
    or a window function.
  - For `rank_in_day`, use `ROW_NUMBER` partitioned by product and date, ordered by `order_line_id`.
  - For `days_since_last_sale`, use LAG on `order_date`, then `DATEDIFF`.
  - For `cumulative_percentile`, use `PERCENT_RANK` partitioned by `order_date`, ordered by
    `cumulative_revenue`.

## Data sources

Explore each to see the columns available:

- `stg_orders__order_lines`: order line details, with quantities, amounts and identifiers
- `stg_orders__orders`: order headers, with timestamps and status
- `stg_product__product_variants`: maps variants to products
- `stg_product__products`: product information

## Done when

- `main.fct_inventory_balance` exists as a table with the columns and rules above, and the schema file defines
  the test.
- The ticket check `inventory-balance` passes.
