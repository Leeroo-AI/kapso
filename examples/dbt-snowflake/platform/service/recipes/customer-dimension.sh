#!/bin/bash
# Build for the customer dimension ticket (benchmark task dbt-fix-customer-ltv).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/marts/core/dim_customers.sql" warehouse/models/marts/core/dim_customers.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run --select dim_customers
