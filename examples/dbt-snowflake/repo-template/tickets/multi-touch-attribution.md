# Multi-touch attribution report broke: rebuild it

**Area:** marketing · **Reported by:** Marketing analytics · **Kind:** rebuild a report as a new model

## What is wrong

The multi-touch marketing attribution mart is broken and is rebuilt as a new model. The warehouse holds
GA4-style sessions and events data. The new model outputs daily attribution metrics for five attribution
models, with a channel grouping and assist metrics.

## What we need

- Model: `marketing/models/marts/marketing/rpt_multi_touch_attribution_fixed.sql` (the marketing reports'
  project)
- Output: `analytics.rpt_multi_touch_attribution_fixed`, a table

### Output schema

| Column | Type | Description |
|--------|------|-------------|
| `attribution_date` | DATE | Session date (from `session_start`) |
| `channel` | STRING | `utm_source`, default `direct` |
| `medium` | STRING | `utm_medium`, default `none` |
| `campaign` | STRING | `utm_campaign`, default `none` |
| `channel_group` | STRING | One of: `Direct`, `Paid`, `Organic`, `Referral` |
| `attribution_model` | STRING | One of: `last_touch`, `first_touch`, `linear`, `time_decay`, `position_based` |
| `sessions` | INTEGER | Distinct `session_id` count |
| `conversions` | INTEGER | Sessions where `is_converted = true` |
| `attributed_revenue` | NUMERIC(12,2) | Sum of purchase revenue for converted sessions |
| `conversion_rate` | NUMERIC | conversions/sessions, in [0,1], 4 decimals; 0 when sessions=0 |
| `assist_sessions` | INTEGER | Count of distinct sessions that are assists (see the business rules) |

## Source tables

- `main.stg_ga__sessions`: `session_id`, `visitor_id`, `customer_id`, `session_start`, `is_converted`,
  `utm_source`, `utm_medium`, `utm_campaign`
- `main.stg_ga__events`: `event_name`, `event_value`
- `main.int_sessions_events_joined` (optional): `t2_event_name`, `t2_event_value`; prefer it for revenue when
  it is available

Use `adapter.get_relation` and `adapter.get_columns_in_relation` to detect `int_sessions_events_joined` and
`t2_event_value`.

## Business rules

### Time window (deterministic)

- A fixed 91-day window: `CAST(session_start AS DATE) BETWEEN '2025-10-02'::DATE AND '2025-12-31'::DATE`
  (inclusive). Do not use `CURRENT_DATE` or `INTERVAL`.

### UTM and channel_group

- `channel`: `COALESCE(utm_source, 'direct')` (missing is `direct`)
- `medium`: `COALESCE(utm_medium, 'none')`; `campaign`: `COALESCE(utm_campaign, 'none')`
- `channel_group` (compare on `LOWER(TRIM(COALESCE(...,'')))`):
  - `Paid`: `LOWER(TRIM(COALESCE(utm_medium,'')))` IN (`cpc`, `ppc`, `paid`, `cpm`)
  - `Organic`: `LOWER(TRIM(COALESCE(utm_medium,'')))` = `organic`
  - `Direct`: `LOWER(TRIM(COALESCE(utm_source,'')))` IN (`''`, `direct`) AND
    `LOWER(TRIM(COALESCE(utm_medium,'')))` IN (`''`, `none`, `(none)`)
  - `Referral`: everything else

### Sessions, conversions, revenue

- **Sessions:** `COUNT(DISTINCT session_id)`
- **Conversions:** sessions where `is_converted = true`
- **Attributed revenue:** the sum of purchase `event_value` (or `t2_event_value` from
  `int_sessions_events_joined` when available) for converted sessions; round to 2 decimals
- **Conversion rate:** `conversions / sessions` rounded to 4 decimals; 0 when sessions=0; within [0,1]

### User identification

- Use `visitor_id`; if it is NULL, fall back to `session_id`: `COALESCE(visitor_id, session_id)`.

### assist_sessions

For each (`attribution_date`, `channel`, `medium`, `campaign`): count the distinct `session_id` values where
the session is in at least one converting path and is not the last touch of that path.

- The converting path of a converted session C: all sessions of the same `visitor_id` with
  `session_start <= C.session_start` within the window. The last touch is C. Every other touch is an assist.
- A session that converts can be an assist for a later conversion by the same user.

### Attribution models

- **last_touch:** attribute conversions and revenue to the session's own channel, medium and campaign.
- **first_touch:** attribute to the earliest session per user (by `session_start`) in the window. All of that
  user's conversions and revenue go to the first session's channel, medium and campaign.
- **linear:** spread evenly across the user's sessions in the window. With N sessions, each gets 1/N of the
  user's conversions and revenue.
- **time_decay:** for each converting session C, build the path (same `visitor_id`,
  `session_start <= C.session_start`). For each touch, `days_before_conversion` = (C's `session_start` date
  minus the touch's `session_start` date) in days. Weight = `POW(2, -days_before_conversion / 7.0)` (a 7-day
  half-life). Normalise the weights to sum to 1 per path, and attribute that conversion and its revenue in
  proportion. When `days_before_conversion` = 0, the weight is 1.
- **position_based** (U-shaped): for each path, order by `session_start` (1 = first, N = last). N=1: 100% to
  the only touch; N=2: 50% each; N>=3: 40% to the first, 40% to the last, 20% split evenly among the (N-2)
  middle touches, each of which gets `20/(N-2)`.

## Implementation requirements

- `dbt run --select rpt_multi_touch_attribution_fixed` in `marketing/` succeeds and creates
  `analytics.rpt_multi_touch_attribution_fixed`.
- The output has rows for each of the five `attribution_model` values.
- Every row includes `channel_group` and `assist_sessions`.

## Tips

- Build the reference models first: `dbt run --select stg_ga__sessions stg_ga__events
  int_sessions_events_joined` in `warehouse/`.
- Prefer the relations in the `main` schema for staged data.
- Round revenue to 2 decimals and `conversion_rate` to 4 decimals.
- For date arithmetic, use `DATEDIFF('day', start, end)`.

## Done when

- The implementation requirements above hold.
- The ticket check `multi-touch-attribution` passes.
