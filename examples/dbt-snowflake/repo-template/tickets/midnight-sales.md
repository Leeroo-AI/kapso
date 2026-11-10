# Daily sales are wrong around midnight

**Area:** revenue · **Reported by:** Operations · **Kind:** fix an existing model

## What is wrong

The warehouse consolidates orders from two systems, SAP and POS. Operations report that daily sales
aggregations are incorrect, particularly around midnight hours.

Data from the two source systems is being processed inconsistently. Investigation shows that timestamps from
POS and SAP are not aligned properly, which gives incorrect aggregations when the data is combined.

## What we need

Fix the `int_sales__orders_enriched` model so that it produces correct output.

- Model: `warehouse/models/intermediate/sales/int_sales__orders_enriched.sql`
- Output: `main.int_sales__orders_enriched`
- Sources:
  - `stg_sap__vbak`: SAP order data with timestamps, amounts and status fields. Explore the table to see the
    columns available.
  - `stg_pos__transactions`: POS transaction data with timestamps, amounts and status fields. Explore the
    table to see the columns available.

## What we know

1. The two source systems may store timestamps in different time zones. There is a 5-hour difference between
   the systems. Any of these resolves it:
   - adjusting POS timestamps forward 5 hours, or
   - adjusting SAP timestamps backward 5 hours, or
   - any other adjustment that leaves both systems aligned (a 5-hour total correction).
2. The timestamp fields in the staging tables use mixed formats; the data quality is inconsistent. You will
   find:
   - standard SQL format: `YYYY-MM-DD HH:MM:SS`
   - American format: `mm/dd/yyyy` or `mm/dd/yyyy HH:MM:SS` (for example `07/23/2020`, `11/27/2023`)
   - European format: `dd/mm/yyyy` or `dd/mm/yyyy HH:MM:SS`
   - compact format: `YYYYMMDD`
   - Unix timestamps: numeric values (seconds since epoch)

   All of these formats have to be handled. Try the American format before the European one.

Snowflake notes: `TRY_TO_TIMESTAMP(...)` parses with a format and returns NULL when it does not match;
`REGEXP_REPLACE` replaces every occurrence by default.

## Done when

- The output produces correct aggregations.
- The ticket check `midnight-sales` passes.

## Constraints

- Do not modify staging models.
- Do not change the model's materialization (keep it a view).
- Preserve all output columns.
