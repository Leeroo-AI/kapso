#!/bin/bash
# Build for the product inventory metrics ticket (benchmark task dbt-fix-inventory-model).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
require_file "$MAIN_PROJECT/models/marts/inventory/rpt_product_inventory_metrics.sql" warehouse/models/marts/inventory/rpt_product_inventory_metrics.sql
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps
dbt run -s rpt_product_inventory_metrics
