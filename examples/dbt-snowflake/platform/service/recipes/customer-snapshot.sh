#!/bin/bash
# Build for the customer snapshot ticket (benchmark task dbt-fix-customer-snapshot-and-build-dimension): the
# snapshots schema, the staging model the snapshot reads, the snapshot, then the dimension built on it.
source "$(dirname "$0")/lib.sh"
overlay_main_project
write_profile "$MAIN_PROJECT" main retail_dw_master
export DBT_PROFILES_DIR="$MAIN_PROJECT"
create_schema snapshots
require_file "$MAIN_PROJECT/snapshots/customer_snapshot.sql" warehouse/snapshots/customer_snapshot.sql
cd "$MAIN_PROJECT"
dbt deps --profiles-dir . || true
dbt run --select stg_customers --profiles-dir .
dbt snapshot --select customer_snapshot --profiles-dir .
dbt run --select dim_customer_current --profiles-dir .
