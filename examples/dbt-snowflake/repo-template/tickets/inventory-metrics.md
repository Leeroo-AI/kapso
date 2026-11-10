# Product inventory metrics: invalid values, and two missing classifications

**Area:** inventory · **Reported by:** Supply chain · **Kind:** fix an existing model and add columns

## What is wrong

The `rpt_product_inventory_metrics` model has bugs and lacks business-critical classification logic.

## What we need

- Model: `warehouse/models/marts/inventory/rpt_product_inventory_metrics.sql`
- Output: `main.rpt_product_inventory_metrics`

### 1. Fix the existing bugs

The model produces invalid values (`inf`, `nan`) in calculated metrics. Identify and fix all the issues.

### 2. Add an inventory health tier

Add an `inventory_health_tier` column using waterfall logic (the first match wins):

| Tier | Criteria |
|------|----------|
| `critical` | Top 10% turnover + less than 1 week of supply + has recent sales |
| `at_risk` | High stockout rate (>15%) OR low supply (<2 weeks with sales activity) OR the majority of locations stocked out |
| `healthy` | Above-median turnover + low stockout (<5%) + adequate supply (2-12 weeks) |
| `overstock` | Everything else |

### 3. Add a velocity score

Add a `velocity_score` column (0-100 scale) combining:

- Turnover performance (40% weight): use `PERCENT_RANK() OVER (ORDER BY turnover_ratio)` to normalise turnover
  to 0-1.
- Sell-through efficiency (30% weight): calculated as
  `units_sold_365d / NULLIF(units_sold_365d + GREATEST(total_quantity_on_hand, 0), 0)` (the ratio of sold to
  total throughput).
- Picking recency (30% weight): a stepped function on `days_since_last_pick`:
  - <= 7 days: 1.0
  - <= 30 days: 0.7
  - <= 90 days: 0.4
  - > 90 days: 0.1

Final score: `(turnover_component + sell_through_component + recency_component) * 100`

## Expected output

All existing columns, plus:

| Column | Type | Description |
|--------|------|-------------|
| `inventory_health_tier` | varchar | Health classification (critical/at_risk/healthy/overstock) |
| `velocity_score` | decimal | Inventory velocity score (0-100) |

## Requirements

- The model compiles successfully.
- No invalid numeric values.
- All source products are included in the output.
- Tiers are distributed across more than one category.
- Use the `PERCENT_RANK()` window function for the tier calculation.

## Guidelines

- Do not use `CURRENT_DATE` for rolling window calculations (for example the 365-day transaction filter,
  `days_since_last_pick`, `days_since_last_receipt`). Use `MAX(transaction_date)` from the inventory
  transactions table as the reference date instead, so that results are the same whenever the model runs.

## Done when

- The requirements above hold.
- The ticket check `inventory-metrics` passes.
