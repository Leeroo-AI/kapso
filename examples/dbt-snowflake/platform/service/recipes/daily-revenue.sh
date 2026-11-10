#!/bin/bash
# Build for the daily revenue ticket (benchmark task dbt-fix-daily-revenue).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/marts/sales/rpt_order_daily_summary.sql" warehouse/models/marts/sales/rpt_order_daily_summary.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run --select rpt_order_daily_summary
