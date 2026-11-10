"""Create the build sessions' development database (`mirrors.development.database`), as the account admin.

A zero-copy clone of production (see setup_production.py) in which the engineer role owns the modelled layer
(schema MAIN), can create tables and views in ANALYTICS, and can create schemas (the customer snapshot wants
one); the source and raw schemas are readable and nothing more. Besides this database the role reads production
and the release staging database, and writes neither. Run it again to start the database over (it is dropped
and cloned afresh).

The database is left as a plain clone: nothing is built into it here. `python service/mirror.py reset development`
then records it as being in production's state, and puts it back there whenever a build session asks.

Usage: python setup_dev_database.py
"""
import time

from admin import CONFIG, SNOWFLAKE, connect, records

SOURCE = CONFIG["mirrors"]["production_database"]
DATABASE = CONFIG["mirrors"]["development"]["database"]
ENGINEER = SNOWFLAKE["engineer"]["role"]
PLATFORM_ROLE = SNOWFLAKE["pool"]["role"]
MODELLED_SCHEMA = "MAIN"
REPORT_SCHEMA = "ANALYTICS"

cur = connect().cursor()


def run(statement):
    started = time.time()
    cur.execute(statement)
    print(f"  {time.time() - started:6.1f}s  {statement[:140]}", flush=True)


run(f"create or replace database {DATABASE} clone {SOURCE} comment = 'Development clone of the retail warehouse for analytics engineering'")
# The platform's role resets this database (service/mirror.py), so it owns it; the grants below follow.
run(f"grant ownership on database {DATABASE} to role {PLATFORM_ROLE} revoke current grants")
schemas = [row["name"] for row in records(cur, f"show schemas in database {DATABASE}") if row["name"] not in ("INFORMATION_SCHEMA", "PUBLIC")]
# What the clone inherited belongs to production's roles; here it is handed to the roles of this database.
for schema in schemas:
    owner = ENGINEER if schema == MODELLED_SCHEMA else PLATFORM_ROLE
    run(f"grant ownership on schema {DATABASE}.{schema} to role {owner} revoke current grants")
    run(f"grant ownership on all tables in schema {DATABASE}.{schema} to role {owner} revoke current grants")
run(f"grant usage, create schema on database {DATABASE} to role {ENGINEER}")
run(f"grant usage on all schemas in database {DATABASE} to role {ENGINEER}")
run(f"grant select on all tables in database {DATABASE} to role {ENGINEER}")
run(f"grant create table, create view on schema {DATABASE}.{REPORT_SCHEMA} to role {ENGINEER}")

print("grants to the engineer role, by kind of object:")
counts = {}
for row in records(cur, f"show grants to role {ENGINEER}"):
    key = (row["privilege"], row["granted_on"])
    counts[key] = counts.get(key, 0) + 1
for (privilege, granted_on), count in sorted(counts.items()):
    print(f"  {privilege:<14} on {granted_on:<10} x {count}")
print("databases the role can use:", sorted({row["name"].split(".")[0] for row in records(cur, f"show grants to role {ENGINEER}") if row["granted_on"] == "DATABASE"}))
