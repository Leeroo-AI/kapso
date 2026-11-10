# CAC payback waterfall broke: rebuild it

**Area:** marketing · **Reported by:** Marketing finance · **Kind:** rebuild a report as a new model

## What is wrong

The CAC payback waterfall report is broken and is rebuilt as a new model. Marketing spend data exists in the
warehouse, among the models the warehouse project materializes; find the right source table by exploring them.

## What we need

- Model: `marketing/models/marts/marketing/rpt_cac_payback_waterfall_fixed.sql` (the marketing reports'
  project)
- Output: `analytics.rpt_cac_payback_waterfall_fixed`, a table

### Output columns

- `campaign_id` (STRING)
- `channel_key` (INTEGER)
- `cohort_date` (DATE): the first date on which we observe spend for the campaign
- `days_since_cohort` (INTEGER): 0 for the cohort date
- `daily_spend` (NUMERIC(12,2))
- `daily_revenue_attributed` (NUMERIC(12,2))
- `cumulative_spend` (NUMERIC(12,2))
- `cumulative_revenue` (NUMERIC(12,2))
- `payback_ratio` (NUMERIC(10,4)): a payback ratio showing how much of the investment has been recouped
- `is_paid_back` (BOOLEAN): true when the campaign has recouped its investment

### Business rules

- Explore the source to see which fields are available. You will need to find the marketing spend data among
  the models materialized in the warehouse.
- Build the waterfall per `campaign_id` (grain: one row per `campaign_id` + `days_since_cohort`).
- **Channel key handling:** each campaign has exactly one `channel_key` in the output. If the source data has a
  campaign appearing under several `channel_key` values, pick the `channel_key` with the highest total
  `spend_amount` for that campaign. Aggregate all spend and revenue across channel keys into a single campaign
  row before building the waterfall.
- Include all campaign-days in the data, and compute daily and cumulative values.
- Round currency fields to 2 decimals and `payback_ratio` to 4 decimals.

### Quality requirements

- No negative spend or revenue.
- `cumulative_spend` and `cumulative_revenue` are non-decreasing within each campaign.
- Reconciliation: for each campaign, the last row's cumulative totals equal the total source values within
  0.01.

## Done when

- `analytics.rpt_cac_payback_waterfall_fixed` exists with the columns and rules above.
- The ticket check `cac-payback` passes.
