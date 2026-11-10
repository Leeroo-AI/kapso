#!/bin/bash
# Build for the daily cohorts ticket (benchmark task dbt-fix-daily-cohorts).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
export DBT_PROFILES_DIR="$MAIN_PROJECT"
require_file "$MAIN_PROJECT/models/marts/analytics/rpt_daily_cohorts.sql" warehouse/models/marts/analytics/rpt_daily_cohorts.sql
cd "$MAIN_PROJECT"
dbt deps
dbt run --select rpt_daily_cohorts
