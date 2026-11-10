#!/bin/bash
# Build for the email attribution ticket (benchmark task dbt-fix-email-attribution): the reference models from
# the warehouse project first, then the report from the marketing project at /app/dbt_project, schema analytics.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
schema_macro_default_only "$MAIN_PROJECT"
build_reference_models
side_project dbt_project
write_profile "$SIDE_PROJECT" analytics dbt_project retail_dw_master
require_file "$SIDE_PROJECT/models/marts/marketing/rpt_email_attribution_fixed.sql" marketing/models/marts/marketing/rpt_email_attribution_fixed.sql
cd "$SIDE_PROJECT"
export DBT_PROFILES_DIR="$SIDE_PROJECT"
dbt run --select rpt_email_attribution_fixed --full-refresh
