# Product performance metrics

**Area:** inventory · **Reported by:** Merchandising · **Kind:** new model

## What we need

Build a product performance summary from sales data: a model called `fct_product_metrics` that aggregates the
key performance indicators for each product.

- Model: `warehouse/models/marts/product/fct_product_metrics.sql`
- Output: `main.fct_product_metrics`, a table, sorted by revenue descending

Explore the staging models of the warehouse project to find the tables for products, variants, orders and
order line items. You will need to understand how these tables relate and decide which columns to use.

The model gives a product-level summary with identification fields, order counts, unit volumes, revenue,
average pricing and variant information.

### Rules

- **Order filtering:** exclude orders with status `CANCELLED`, `RETURNED` or `FAILED`. Every other order
  status is a valid completed sale.
- **Output columns:** `product_id`, `product_name`, `category`, `total_orders`, `units_sold`,
  `product_revenue`, `avg_unit_price`, `total_variants`
- **Average price:** `avg_unit_price` is total revenue divided by total units sold (`product_revenue /
  units_sold`, a weighted average), not a simple `AVG(unit_price)`.
- All products appear in the output, including those without any sales.

## Guidelines

- Mind the grain of the model: joining across several dimensions can fan rows out.

## Done when

- `main.fct_product_metrics` exists as a table with the columns and rules above.
- The ticket check `product-metrics` passes.
