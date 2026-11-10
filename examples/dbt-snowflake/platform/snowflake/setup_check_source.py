"""One-time preparation of the check source database (the warehouse as the benchmark's loader left it), run as
the account admin.

Why: a zero-copy clone of a database keeps who owns each schema and table inside it. The load leaves
them owned by the role that ran it, so the role a check container runs as cannot replace a model's table in its
clone ("must have OWNERSHIP"). After this script the schemas and tables of the source belong to the checks role,
every clone inherits that, and the role has no USAGE on the source database itself, so a check can rebuild
anything inside its own clone and nothing in the source. The platform's role (which creates and drops the
clones) inherits the checks role. PUBLIC loses the read access the loader granted on the source.

Idempotent. Usage: python setup_check_source.py
"""
import time

from admin import SNOWFLAKE, connect, records

SOURCE = SNOWFLAKE["source_database"]
CHECKS_ROLE = SNOWFLAKE["checks"]["role"]
PLATFORM_ROLE = SNOWFLAKE["pool"]["role"]

cur = connect(database=SOURCE).cursor()


def run(statement):
    started = time.time()
    cur.execute(statement)
    print(f"  {time.time() - started:6.1f}s  {statement[:150]}")


run(f"create role if not exists {CHECKS_ROLE} comment = 'Runs the ticket checks inside its own clone of the warehouse'")
run(f"grant role {CHECKS_ROLE} to role {PLATFORM_ROLE}")
run(f"grant usage on database {SOURCE} to role {PLATFORM_ROLE}")
run(f"grant ownership on all schemas in database {SOURCE} to role {CHECKS_ROLE} copy current grants")
run(f"grant ownership on all tables in database {SOURCE} to role {CHECKS_ROLE} copy current grants")
run(f"revoke select on all tables in database {SOURCE} from role PUBLIC")
run(f"revoke usage on all schemas in database {SOURCE} from role PUBLIC")
run(f"revoke usage on database {SOURCE} from role PUBLIC")

print("grants on the database:")
for row in records(cur, f"show grants on database {SOURCE}"):
    print(f"  {row['privilege']:<16} to {row['granted_to']} {row['grantee_name']}")
owners = records(cur, f"""
    select table_owner, count(*) as n from {SOURCE}.information_schema.tables
    where table_schema <> 'INFORMATION_SCHEMA' group by 1""")
print("table owners:", {row["TABLE_OWNER"]: row["N"] for row in owners})
schemas = records(cur, f"select schema_owner, count(*) as n from {SOURCE}.information_schema.schemata group by 1")
print("schema owners:", {row["SCHEMA_OWNER"]: row["N"] for row in schemas})
print("grants to PUBLIC that remain on the source:")
remaining = [row for row in records(cur, "show grants to role PUBLIC") if SOURCE in str(row["name"])]
print(f"  {len(remaining)}", [f"{row['privilege']} {row['name']}" for row in remaining[:8]])
