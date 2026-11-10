#!/bin/bash
# Build for the product metrics ticket (benchmark task dbt-fix-product-metrics).
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
export DBT_PROFILES_DIR="$MAIN_PROJECT"
require_file "$MAIN_PROJECT/models/marts/product/fct_product_metrics.sql" warehouse/models/marts/product/fct_product_metrics.sql
cd "$MAIN_PROJECT"
dbt deps
dbt run --select fct_product_metrics
