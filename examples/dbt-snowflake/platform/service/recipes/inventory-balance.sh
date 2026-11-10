#!/bin/bash
# Build for the cumulative product sales ticket (benchmark task dbt-fix-inventory-balance): the model, then the
# project's own tests on it, as the reference runs them.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
export DBT_PROFILES_DIR="$MAIN_PROJECT"
cd "$MAIN_PROJECT"
dbt deps
require_file "$MAIN_PROJECT/models/marts/inventory/fct_inventory_balance.sql" warehouse/models/marts/inventory/fct_inventory_balance.sql
require_file "$MAIN_PROJECT/models/marts/inventory/schema.yml" warehouse/models/marts/inventory/schema.yml
dbt run --select fct_inventory_balance
dbt test --select fct_inventory_balance
