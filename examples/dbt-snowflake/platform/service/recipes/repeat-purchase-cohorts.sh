#!/bin/bash
# Build for the repeat purchase cohort revenue ticket (benchmark task dbt-fix-repeat-purchase-cohort-revenue).
# On Snowflake the reference builds the model inside the warehouse project and makes that project reachable at
# /app/dbt_project as well.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
ln -sfn "$MAIN_PROJECT" "$SIDE_PROJECT"
require_file "$MAIN_PROJECT/models/marts/customer/rpt_repeat_purchase_cohort_revenue_fixed.sql" warehouse/models/marts/customer/rpt_repeat_purchase_cohort_revenue_fixed.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps || say "dbt deps reported a problem"
dbt run --select rpt_repeat_purchase_cohort_revenue_fixed
