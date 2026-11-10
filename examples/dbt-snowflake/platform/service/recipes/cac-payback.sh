#!/bin/bash
# Build for the CAC payback ticket (benchmark task dbt-fix-cac-payback-waterfall): the marketing project at
# /app/dbt_project, the analytics schema created first as the reference creates it, the report built there.
source "$(dirname "$0")/lib.sh"
overlay_main_project
side_project retail_dw_master
write_profile "$SIDE_PROJECT" main retail_dw_master dbt_project
create_schema '"analytics"'
require_file "$SIDE_PROJECT/models/marts/marketing/rpt_cac_payback_waterfall_fixed.sql" marketing/models/marts/marketing/rpt_cac_payback_waterfall_fixed.sql
cd "$SIDE_PROJECT"
export DBT_PROFILES_DIR="$SIDE_PROJECT"
dbt deps || true
dbt run --select rpt_cac_payback_waterfall_fixed
