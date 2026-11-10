#!/bin/bash
# Build for the paid search attribution ticket (benchmark task dbt-fix-paid-search-attribution). On Snowflake
# the reference builds this report inside the warehouse project's directory: the reference models first, then a
# minimal project file, a schema override that uses a custom schema exactly, the analytics schema created, and
# the report under models/marts/marketing. The submitted report comes from the marketing project; its folder
# keeps that project's configuration (schema analytics, table) and the marketing project's own macros are
# carried beside the warehouse's, which are the two additions to the reference.
source "$(dirname "$0")/lib.sh"
overlay_main_project
mkdir -p "$MAIN_PROJECT/models/marts/marketing" "$MAIN_PROJECT/macros/utils"
write_profile "$MAIN_PROJECT" main retail_dw_master
build_reference_models
cat > "$MAIN_PROJECT/dbt_project.yml" << 'EOF'
name: 'dbt_project'
version: '1.0.0'
config-version: 2
profile: 'retail_dw_master'
model-paths: ["models"]
macro-paths: ["macros"]
models:
  dbt_project:
    marts:
      marketing:
        rpt_paid_search_attribution_fixed:
          +schema: analytics
          +materialized: table
EOF
schema_macro_custom_exact "$MAIN_PROJECT"
carry_marketing_macros
create_schema '"analytics"'
require_file "$CANDIDATE/marketing/models/marts/marketing/rpt_paid_search_attribution_fixed.sql" marketing/models/marts/marketing/rpt_paid_search_attribution_fixed.sql
cp "$CANDIDATE/marketing/models/marts/marketing/rpt_paid_search_attribution_fixed.sql" "$MAIN_PROJECT/models/marts/marketing/rpt_paid_search_attribution_fixed.sql"
cd "$MAIN_PROJECT"
export DBT_PROFILES_DIR="$MAIN_PROJECT"
dbt deps || say "dbt deps reported a problem"
dbt run --select rpt_paid_search_attribution_fixed
