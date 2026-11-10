#!/bin/bash
# Build for the customer lifetime value ticket (benchmark task dbt-customer-ltv-fix). The reference makes a
# model's custom schema apply exactly, so that the output lands in main; the same override is written here.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
schema_macro_custom_exact "$MAIN_PROJECT"
require_file "$MAIN_PROJECT/models/marts/customer/customer_ltv.sql" warehouse/models/marts/customer/customer_ltv.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps || true
dbt run --select customer_ltv
