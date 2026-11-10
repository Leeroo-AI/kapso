"""Create the production database, the release staging database and the release identity, as the account admin.

  the source      the pristine image (`snowflake.source_database`). The check service clones it and nothing
                  ever writes to it.
  production      (`mirrors.production_database`) a zero-copy clone of the pristine image. The release role owns
                  the modelled layer (schema MAIN) and may create tables in ANALYTICS and new schemas; the
                  engineer role reads everything and writes nothing.
  staging         (`mirrors.staging.database`) where a release is rehearsed: a clone of production with the same
                  ownership, put back to production's state before every release (service/release.py). The
                  engineer role reads it.
  release         (`snowflake.release`) the identity of the release service. Its key stays on the platform's
                  machine and never reaches a build machine: a build session can read production and cannot
                  write it.

The platform's role (which creates, resets and drops these databases) inherits the release and engineer roles,
so a reset can replace what they own. Run it after setup_principals.py. Idempotent.

Usage: python setup_production.py            # create what is missing
       python setup_production.py --reset    # production and staging start over from the pristine image
"""
import pathlib
import sys
import time

from admin import CONFIG, SNOWFLAKE, connect, public_key_body, records

PRISTINE = SNOWFLAKE["source_database"]
PRODUCTION = CONFIG["mirrors"]["production_database"]
STAGING = CONFIG["mirrors"]["staging"]["database"]
RELEASE_USER = SNOWFLAKE["release"]["user"]
RELEASE = SNOWFLAKE["release"]["role"]
RELEASE_WAREHOUSE = SNOWFLAKE["release"]["warehouse"]
RELEASE_KEY = pathlib.Path(SNOWFLAKE["release"]["private_key_file"])
ENGINEER = SNOWFLAKE["engineer"]["role"]
PLATFORM_ROLE = SNOWFLAKE["pool"]["role"]
MODELLED_SCHEMA = "MAIN"
REPORT_SCHEMA = "ANALYTICS"

reset = "--reset" in sys.argv[1:]
cur = connect().cursor()


def run(statement):
    started = time.time()
    cur.execute(statement)
    print(f"  {time.time() - started:6.1f}s  {statement[:150]}", flush=True)


def shape(database):
    """Ownership and read access inside a production-shaped database."""
    schemas = [row["name"] for row in records(cur, f"show schemas in database {database}") if row["name"] not in ("INFORMATION_SCHEMA", "PUBLIC")]
    for schema in schemas:
        owner = RELEASE if schema == MODELLED_SCHEMA else PLATFORM_ROLE
        run(f"grant ownership on schema {database}.{schema} to role {owner} revoke current grants")
        run(f"grant ownership on all tables in schema {database}.{schema} to role {owner} revoke current grants")
    run(f"grant usage, create schema on database {database} to role {RELEASE}")
    run(f"grant usage on all schemas in database {database} to role {RELEASE}")
    run(f"grant select on all tables in database {database} to role {RELEASE}")
    run(f"grant create table, create view on schema {database}.{REPORT_SCHEMA} to role {RELEASE}")
    run(f"grant usage on database {database} to role {ENGINEER}")
    run(f"grant usage on all schemas in database {database} to role {ENGINEER}")
    run(f"grant select on all tables in database {database} to role {ENGINEER}")
    run(f"grant select on all views in database {database} to role {ENGINEER}")
    # What a release creates later (a rebuilt table, a new report, the snapshot's schema) is readable too.
    run(f"grant usage on future schemas in database {database} to role {ENGINEER}")
    run(f"grant select on future tables in database {database} to role {ENGINEER}")
    run(f"grant select on future views in database {database} to role {ENGINEER}")


print(RELEASE_USER)
run(f"create role if not exists {RELEASE} comment = 'Releases merged changes into the production warehouse'")
run(f"grant usage on warehouse {RELEASE_WAREHOUSE} to role {RELEASE}")
run(f"create user if not exists {RELEASE_USER} type = service default_role = {RELEASE} default_warehouse = {RELEASE_WAREHOUSE} "
    f"comment = 'Releases merged changes into the production warehouse'")
cur.execute(f"alter user {RELEASE_USER} set rsa_public_key = '{public_key_body(RELEASE_KEY)}'")
run(f"grant role {RELEASE} to user {RELEASE_USER}")
run(f"grant role {RELEASE} to role {PLATFORM_ROLE}")
run(f"grant role {ENGINEER} to role {PLATFORM_ROLE}")

existing = {row["name"] for row in records(cur, "show databases")}
for database, source in ((PRODUCTION, PRISTINE), (STAGING, PRODUCTION)):
    if database in existing and not reset:
        print(f"{database}: exists, left as it is")
        continue
    print(database)
    run(f"create or replace database {database} clone {source} comment = "
        + ("'The production retail warehouse'" if database == PRODUCTION else "'Release rehearsal: production as it stands plus the release under audit'"))
    run(f"grant ownership on database {database} to role {PLATFORM_ROLE} revoke current grants")
    shape(database)

for database in (PRODUCTION, STAGING):
    owners = records(cur, f"""select table_schema, table_owner, count(*) as n from {database}.information_schema.tables
                              where table_schema <> 'INFORMATION_SCHEMA' group by 1, 2 order by 3 desc""")
    print(f"{database} table owners:", [(row["TABLE_SCHEMA"], row["TABLE_OWNER"], row["N"]) for row in owners][:6])
for role in (RELEASE, ENGINEER):
    databases = sorted({row["name"].split(".")[0] for row in records(cur, f"show grants to role {role}") if row["granted_on"] == "DATABASE"})
    print(f"databases {role} can use:", databases)
