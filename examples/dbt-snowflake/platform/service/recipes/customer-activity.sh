#!/bin/bash
# Build for the customer activity ticket (benchmark task dbt-fix-customer-churn).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/marts/customer/fct_customer_activity.sql" warehouse/models/marts/customer/fct_customer_activity.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run --select fct_customer_activity
