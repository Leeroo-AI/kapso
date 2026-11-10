#!/bin/bash
# Build for the sales-around-midnight ticket (benchmark task dbt-fix-timezone-sales).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/intermediate/sales/int_sales__orders_enriched.sql" warehouse/models/intermediate/sales/int_sales__orders_enriched.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run --select int_sales__orders_enriched
