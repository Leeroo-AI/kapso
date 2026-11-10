# Paid search attribution report broke: rebuild it

**Area:** marketing · **Reported by:** Marketing analytics · **Kind:** rebuild a report as a new model

## What is wrong

The paid search attribution mart is broken and is rebuilt as a new model. The warehouse already holds a
GA4-style sessions and events dataset. The new model produces a clean daily paid search attribution table.

## What we need

- Model: `marketing/models/marts/marketing/rpt_paid_search_attribution_fixed.sql` (the marketing reports'
  project)
- Output: `analytics.rpt_paid_search_attribution_fixed`, a table

### Columns

1. `attribution_date` (DATE)
2. `channel` (STRING)
3. `medium` (STRING)
4. `campaign` (STRING)
5. `sessions` (INTEGER)
6. `conversions` (INTEGER)
7. `attributed_revenue` (NUMERIC with 2 decimals)
8. `conversion_rate` (NUMERIC between 0 and 1 with 4 decimals)

### Business rules

- Use the GA session data in the warehouse.
- Limit to the last 90 days of data relative to the most recent session start date in the dataset: use a
  subquery such as `CAST(session_start AS DATE) >= (SELECT MAX(CAST(session_start AS DATE)) FROM
  <sessions_source>) - INTERVAL '90 days'`, not `CURRENT_DATE`.
- Treat a missing `utm_source` as `direct`, and a missing `utm_medium` or `utm_campaign` as `none`.
- Count sessions by distinct `session_id`.
- `conversions`: count the sessions where `is_converted = true`.
- `attributed_revenue`: the sum of purchase revenue from events joined to converted sessions. Prefer
  `int_sessions_events_joined` if it is available; otherwise fall back to `stg_ga__events` and `event_value`.
- `conversion_rate`: conversions / sessions (rounded to 4 decimals), 0 when there are no sessions.
- Focus on paid search quality: the output may include several channels, but paid search (for example
  google / cpc) must not be lost or double-counted.

### Schema

The table has to land in exactly the `analytics` schema, not `main_analytics`. The marketing project is set
up for that: its `generate_schema_name` macro uses a custom schema as written, and its project file gives
every model the schema `analytics`.

## What the ticket check expects

- The model `rpt_paid_search_attribution_fixed.sql` under `models/marts/marketing/`.
- `dbt run --select rpt_paid_search_attribution_fixed` succeeds and creates
  `analytics.rpt_paid_search_attribution_fixed`.
- The columns and the data quality match the business rules above.

## Tips

- Build the reference models first: `dbt run --select stg_ga__sessions stg_ga__events
  int_sessions_events_joined` in `warehouse/`.
- In the model, prefer the relations in the `main` schema (that is where the reference models materialize).
- Use `adapter.get_relation` and `adapter.get_columns_in_relation` to detect the relations and columns that
  are available, as a robust dbt macro would.
- Round revenue to 2 decimals and the conversion rate to 4 decimals to avoid precision issues.

## Done when

- `analytics.rpt_paid_search_attribution_fixed` exists with the columns and rules above.
- The ticket check `paid-search-attribution` passes.
