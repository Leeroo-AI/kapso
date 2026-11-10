#!/bin/bash
# Build for the category performance ticket (benchmark task dbt-fix-category-revenue). The reference keeps every
# model in the profile's schema on Snowflake, so the same override is written here.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
export DBT_PROFILES_DIR="$MAIN_PROJECT"
cd "$MAIN_PROJECT"
schema_macro_default_only "$MAIN_PROJECT"
require_file "$MAIN_PROJECT/models/marts/product/rpt_category_performance.sql" warehouse/models/marts/product/rpt_category_performance.sql
dbt deps
dbt run --select rpt_category_performance
