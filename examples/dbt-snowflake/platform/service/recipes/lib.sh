#!/bin/bash
# Shared steps of the ticket build recipes.
#
# A recipe is the benchmark's reference scaffolding for one task (the connection profile, the schema steps, the
# dbt commands its solution runs) with the submitted project in place of the reference's own files. Each function
# below reproduces one such step as the reference solutions write it, so the task's tests meet the environment
# they were validated in. A recipe never writes a model: what is built is what was submitted.
set -e

if [ -f /tmp/snowflake_env.sh ]; then
    source /tmp/snowflake_env.sh
fi

CANDIDATE=/candidate
MAIN_PROJECT=/app/dbt_models_snowflake
SIDE_PROJECT=/app/dbt_project
PRIVATE_KEY_PATH=/tmp/snowflake_private_key.p8

echo "$SNOWFLAKE_PRIVATE_KEY" | base64 -d > "$PRIVATE_KEY_PATH"
chmod 600 "$PRIVATE_KEY_PATH"
# From here on the key lives only in that file. A submitted model can read the build's environment
# (env_var) but not its files, so nothing secret stays in the environment dbt runs with; the tests, which run
# after the build in a shell of their own, still get the container's variables.
unset SNOWFLAKE_PRIVATE_KEY SNOWFLAKE_PRIVATE_KEY_PASSPHRASE SNOWFLAKE_PASSWORD

say() { echo "CHECK: $*"; }

# The submitted warehouse project replaces the image's copy: its models, macros, seeds, snapshots, tests and
# analyses and its project files. Build output and any connection profile in the submission stay out.
overlay_main_project() {
    for part in models macros seeds snapshots tests analyses; do
        rm -rf "${MAIN_PROJECT:?}/$part"
        if [ -d "$CANDIDATE/warehouse/$part" ]; then
            cp -a "$CANDIDATE/warehouse/$part" "$MAIN_PROJECT/$part"
        fi
    done
    for file in dbt_project.yml packages.yml package-lock.yml; do
        if [ ! -f "$CANDIDATE/warehouse/$file" ]; then
            say "the submission has no warehouse/$file"
            exit 3
        fi
        cp "$CANDIDATE/warehouse/$file" "$MAIN_PROJECT/$file"
    done
    say "submitted warehouse project in place"
}

# The side project the marketing tasks expect at /app/dbt_project: the reference's project file for the task
# (named dbt_project, on the profile the task names), the submitted marketing models and macros, and the
# marketing project's shipped configuration (every model a table in the schema analytics, exactly). The tasks
# differ in the profile they name, so the project file is written here and the submitted one is not used.
# Argument: the profile name the task's reference uses.
side_project() {
    local profile=$1
    rm -rf "$SIDE_PROJECT"
    mkdir -p "$SIDE_PROJECT/models" "$SIDE_PROJECT/macros"
    if [ -d "$CANDIDATE/marketing/models" ]; then
        cp -a "$CANDIDATE/marketing/models/." "$SIDE_PROJECT/models/"
    fi
    if [ -d "$CANDIDATE/marketing/macros" ]; then
        cp -a "$CANDIDATE/marketing/macros/." "$SIDE_PROJECT/macros/"
    fi
    find "$SIDE_PROJECT/macros" -name 'generate_schema_name.sql' -delete
    cat > "$SIDE_PROJECT/dbt_project.yml" <<EOF
name: 'dbt_project'
version: '1.0.0'
config-version: 2
profile: '$profile'
model-paths: ["models"]
macro-paths: ["macros"]
models:
  dbt_project:
    +schema: analytics
    +materialized: table
EOF
    schema_macro_custom_exact "$SIDE_PROJECT"
    say "submitted marketing models in place"
}

# The submitted marketing macros, for a report the reference builds inside the warehouse project's directory:
# every macro file under marketing/macros is carried, except the schema override (the recipe writes the
# reference's) and a file defining a macro the warehouse project already has (dbt refuses two of one name).
carry_marketing_macros() {
    [ -d "$CANDIDATE/marketing/macros" ] || return 0
    mkdir -p "$MAIN_PROJECT/macros/marketing_side"
    find "$CANDIDATE/marketing/macros" -name '*.sql' ! -name 'generate_schema_name.sql' | sort | while read -r file; do
        clash=""
        for name in $(grep -oE '\{%-?[[:space:]]*macro[[:space:]]+[A-Za-z_][A-Za-z0-9_]*' "$file" | awk '{print $NF}'); do
            if grep -rqE "\{%-?[[:space:]]*macro[[:space:]]+$name[[:space:]]*\(" "$MAIN_PROJECT/macros" --include='*.sql' --exclude-dir=marketing_side; then
                clash=$name
            fi
        done
        if [ -n "$clash" ]; then
            say "marketing macro file $(basename "$file") not carried: the warehouse project already defines $clash"
        else
            cp "$file" "$MAIN_PROJECT/macros/marketing_side/"
        fi
    done
    say "submitted marketing macros in place"
}

# A file the ticket asks for must be in the submission; without it the build stops here and the tests still run.
require_file() {
    if [ ! -s "$1" ]; then
        say "the submission has no file at $2"
        exit 3
    fi
}

# The key-pair profile the reference solutions write. Arguments: directory, default schema, profile names.
write_profile() {
    local directory=$1 schema=$2
    shift 2
    : > "$directory/profiles.yml"
    for name in "$@"; do
        cat >> "$directory/profiles.yml" <<PROFILES
$name:
  target: dev
  outputs:
    dev:
      type: snowflake
      account: ${SNOWFLAKE_ACCOUNT}
      user: ${SNOWFLAKE_USER}
      private_key_path: ${PRIVATE_KEY_PATH}
      database: ${SNOWFLAKE_DATABASE}
      schema: $schema
      warehouse: ${SNOWFLAKE_WAREHOUSE}
      role: ${SNOWFLAKE_ROLE:-}
      threads: 4
PROFILES
    done
}

# The reference override that keeps every model in the profile's schema (category revenue, email attribution).
schema_macro_default_only() {
    mkdir -p "$1/macros/utils"
    cat > "$1/macros/utils/generate_schema_name.sql" << 'GENMACRO'
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {{ default_schema }}
{%- endmacro %}
GENMACRO
}

# The reference override that uses a model's custom schema exactly (customer LTV, paid search, CAC payback).
schema_macro_custom_exact() {
    mkdir -p "$1/macros/utils"
    cat > "$1/macros/utils/generate_schema_name.sql" << 'MACRO'
{% macro generate_schema_name(custom_schema_name, node) -%}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
MACRO
}

# The reference step that creates a schema in the clone before dbt needs it. Argument: the schema, written as
# the reference writes it (the snapshot task unquoted, the marketing tasks quoted in lower case).
create_schema() {
    SCHEMA_SQL=$1 PRIVATE_KEY_PATH=$PRIVATE_KEY_PATH python3 - << 'SCHEMA_SCRIPT'
import os

import snowflake.connector
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization

with open(os.environ["PRIVATE_KEY_PATH"], "rb") as handle:
    key = serialization.load_pem_private_key(handle.read(), password=None, backend=default_backend())
connection = snowflake.connector.connect(
    account=os.environ["SNOWFLAKE_ACCOUNT"], host=os.environ["SNOWFLAKE_HOST"], user=os.environ["SNOWFLAKE_USER"],
    private_key=key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()),
    warehouse=os.environ["SNOWFLAKE_WAREHOUSE"], role=os.environ["SNOWFLAKE_ROLE"], database=os.environ["SNOWFLAKE_DATABASE"])
database = os.environ["SNOWFLAKE_DATABASE"]
connection.cursor().execute(f"create schema if not exists {database}.{os.environ['SCHEMA_SQL']}")
print(f"CHECK: schema {os.environ['SCHEMA_SQL']} ready in {database}")
connection.close()
SCHEMA_SCRIPT
}

# The reference models the marketing tasks build first, from the submitted warehouse project, best effort as the
# reference solutions run them.
build_reference_models() {
    cd "$MAIN_PROJECT"
    export DBT_PROFILES_DIR="$MAIN_PROJECT"
    dbt deps || say "dbt deps reported a problem"
    dbt run --select stg_ga__sessions stg_ga__events int_sessions_events_joined || say "the reference models did not all build"
}
