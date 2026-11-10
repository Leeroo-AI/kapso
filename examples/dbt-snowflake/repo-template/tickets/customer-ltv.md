# Customer lifetime value is overstated

**Area:** customer · **Reported by:** Finance · **Kind:** investigate, then publish a corrected model

## What is wrong

The company's customer lifetime value (LTV) metrics are incorrect. Finance reports that customer LTV values
are overstated, which is affecting customer segmentation and marketing spend decisions.

## What we need

Investigate the warehouse to understand the data model and identify what is causing LTV to be overstated.
Then create a dbt model that produces a corrected `customer_ltv` table in the `main` schema.

LTV should only include revenue from orders that resulted in actual revenue for the company. Investigate the
data to understand which orders should be excluded from the calculation.

- Model: `warehouse/models/marts/customer/customer_ltv.sql`
- Output: `main.customer_ltv`

### Required output: `main.customer_ltv`

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | VARCHAR | Unique customer identifier |
| `total_orders` | INTEGER | Count of valid orders |
| `lifetime_value` | DECIMAL | Sum of order value for valid orders only |
| `avg_order_value` | DECIMAL | Average order value |
| `first_order_date` | DATE | Date of first valid order |
| `last_order_date` | DATE | Date of most recent valid order |
| `value_segment` | VARCHAR | Customer segment based on LTV |

### Value segment rules

Assign customers to segments by their corrected `lifetime_value`:

- `'VIP'`: lifetime_value >= 1000
- `'High Value'`: lifetime_value >= 500 and < 1000
- `'Medium Value'`: lifetime_value >= 100 and < 500
- `'Low Value'`: lifetime_value > 0 and < 100
- `'No Value'`: lifetime_value = 0 or NULL

## Notes

- Round decimal values to 2 decimal places.
- Include all customers, even those with no valid orders.
- Customers with no valid orders have `lifetime_value = 0`, `total_orders = 0` and `value_segment = 'No Value'`.
- `avg_order_value` is 0 for customers with no orders.
- Do not modify upstream staging models.
- Preserve all output columns.

## Done when

- `main.customer_ltv` exists with the columns and rules above.
- The ticket check `customer-ltv` passes.
