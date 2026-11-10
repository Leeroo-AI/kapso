# Email attribution report broke: rebuild it

**Area:** marketing · **Reported by:** Marketing analytics · **Kind:** rebuild a report as a new model

## What is wrong

The email attribution reporting model is broken: it failed after upstream schema changes in the GA4 data
pipeline. It is rebuilt as a new, robust model that tracks email marketing performance accurately.

## What we need

Build a dbt model `rpt_email_attribution_fixed` that produces a daily attribution report for email marketing
campaigns.

- Model: `marketing/models/marts/marketing/rpt_email_attribution_fixed.sql` (the marketing reports' project)
- Output: `analytics.rpt_email_attribution_fixed`, a table

### Output schema

The table holds exactly these columns:

| Column | Type | Description |
|--------|------|-------------|
| `attribution_date` | DATE | Date of the session (from `session_start`) |
| `channel` | VARCHAR | Marketing channel (`utm_source`, default `'direct'`) |
| `medium` | VARCHAR | Marketing medium (`utm_medium`, default `'none'`) |
| `campaign` | VARCHAR | Campaign name (`utm_campaign`, default `'none'`) |
| `sessions` | BIGINT | Count of distinct sessions |
| `conversions` | BIGINT | Count of sessions that converted |
| `attributed_revenue` | DECIMAL(12,2) | Total revenue attributed to these sessions |
| `conversion_rate` | DECIMAL(10,4) | Conversion rate (conversions/sessions), between 0 and 1 |

## Business rules

### Data source

- Primary source: the `main.stg_ga__sessions` table.
- Revenue source: prefer `main.int_sessions_events_joined` if it is available (check for a `t2_event_value`
  column); otherwise fall back to `main.stg_ga__events`.
- These three are models of the warehouse project. Build them first:
  `dbt run --select stg_ga__sessions stg_ga__events int_sessions_events_joined` in `warehouse/`.

### Time filtering

- Include only sessions whose `session_start` date is on or after 2025-10-04 (a fixed cutoff):
  `CAST(session_start AS DATE) >= DATE '2025-10-04'`.
- Exclude sessions after the data generation date: `CAST(session_start AS DATE) <= DATE '2026-01-02'`.

### UTM parameter normalisation

- `channel`: use `utm_source`; if NULL or an empty string, default to `'direct'`.
- `medium`: use `utm_medium`; if NULL or an empty string, default to `'none'`.
- `campaign`: use `utm_campaign`; if NULL or an empty string, default to `'none'`.
- Always trim whitespace before checking for empty strings.
- Handle both the NULL and the empty-string case explicitly.

### Session counting

- Count distinct `session_id` values.
- Filter out NULL or empty `session_id` values.
- Group by `attribution_date`, `channel`, `medium`, `campaign`.
- Each combination appears exactly once.

### Conversion logic

- A session is converted if `is_converted = true`.
- Handle the various boolean representations: `true`, `'1'`, `'true'`, `1`, and so on.
- Count distinct converted sessions per dimension combination.
- Conversions never exceed sessions in any row.

### Revenue attribution

- Attribute revenue only to converted sessions (`is_converted = true`).
- Sum `event_value` from purchase events (`event_name = 'purchase'`).
- When using `int_sessions_events_joined`, use `t2_event_value` and `t2_event_name`.
- Round revenue to 2 decimal places: `ROUND(..., 2)`.
- Keep values non-negative (use `GREATEST(..., 0)`).

### Conversion rate

- Formula: `conversions / sessions`.
- Round to 4 decimal places: `ROUND(..., 4)`.
- If `sessions = 0`, set `conversion_rate = 0.0`.
- It lies between 0.0 and 1.0 inclusive.
- Cast to `DECIMAL(10,4)` for precision.

## Implementation requirements

- Materialization: `table`, in the `analytics` schema (the marketing project's configuration does both).
- Use CTEs for clarity and maintainability.
- Handle edge cases (NULLs, empty strings, type coercion).
- Use `adapter.get_relation()` to check that a table is available, and `adapter.get_columns_in_relation()` to
  check for specific columns; handle a missing relation properly.
- Use `DISTINCT` where it prevents duplicates, filter early in the CTEs, and use a LEFT JOIN for the optional
  revenue data. Consider a `HAVING` clause to drop zero-session rows.

## What the ticket check covers

1. Model compilation and table creation
2. Required columns and data types
3. Date range constraints (90 days)
4. UTM parameter normalisation
5. Conversion rate calculation accuracy
6. Revenue handling and precision
7. Session aggregation correctness
8. Data quality and integrity
9. Email channel presence
10. Conversion count accuracy

## Constraints

- Do not modify upstream staging models.
- Do not change the model's materialization.

## Tips

1. Start with the reference models: build them first, so that the source tables exist.
2. Test incrementally: build and test each CTE on its own where possible.
3. Handle NULLs explicitly, with `COALESCE` and `CASE`.
4. Cast types explicitly to match the expected schema (DATE, BIGINT, DECIMAL).
5. Precision matters: use the exact precision for the DECIMAL types, (12,2) and (10,4).
6. Convert the various boolean representations to a consistent true/false.
7. Revenue fallback: check for the joined table first, then fall back to the events table.
8. Empty results: make sure the CTEs return the proper structure even when empty (the `WHERE 1=0` pattern).

## Common pitfalls

- Forgetting empty strings (not only NULLs).
- Incorrect date filtering (including future dates, or dates older than 90 days).
- Double-counting sessions across dimension combinations.
- Incorrect precision for revenue (2 decimals) or conversion rate (4 decimals).
- Not handling `sessions = 0` (`conversion_rate` should be 0.0).
- Missing revenue data when the joined table does not exist.
- Incorrect boolean conversion logic.

## Done when

- `analytics.rpt_email_attribution_fixed` exists with the columns and rules above.
- The ticket check `email-attribution` passes.
