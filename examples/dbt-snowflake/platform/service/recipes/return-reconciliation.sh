#!/bin/bash
# Build for the return reconciliation ticket (benchmark task dbt-fix-refund-reconciliation).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/marts/sales/rpt_return_reconciliation.sql" warehouse/models/marts/sales/rpt_return_reconciliation.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run -s rpt_return_reconciliation
