"""Create the service identities the platform and the build sessions use, with their warehouses, as the account
admin. Idempotent.

  checks     the identity a check container runs as: owns what is inside its own check database and has no
             access to the source database.
  pool       the platform's own identity: creates, resets and drops the check databases and the development
             and staging copies. Its role is an existing one (SYSADMIN in the example config).
  engineer   the identity the build sessions use: works in the development database only (created by
             setup_dev_database.py). Its private key is the one file of this setup a build machine receives.

Each is the user, role, warehouse and private key file that `config.json` names under `snowflake`; the release
identity is created by setup_production.py. A key pair is generated only when its private key file is absent,
and the public half is registered on the user each run. Nothing secret is printed.

Usage: python setup_principals.py
"""
import pathlib

from admin import SNOWFLAKE, connect, public_key_body, records

COMMENTS = {
    "checks": "Runs the ticket checks inside its own clone of the warehouse",
    "pool": "The data platform: the check databases, the development reset and the release rehearsal",
    "engineer": "Analytics engineering in the development database",
}
WAREHOUSES = SNOWFLAKE["warehouses"]

cur = connect().cursor()
for name, comment in COMMENTS.items():
    identity = SNOWFLAKE[name]
    user, role, warehouse = identity["user"], identity["role"], identity["warehouse"]
    print(f"{name}: user {user}, role {role}")
    cur.execute(f"create role if not exists {role} comment = '{comment}'")
    cur.execute(f"""create warehouse if not exists {warehouse} warehouse_size = {WAREHOUSES['size']}
                    auto_suspend = {WAREHOUSES['auto_suspend_seconds']} auto_resume = true initially_suspended = true
                    statement_timeout_in_seconds = {WAREHOUSES['statement_timeout_seconds']}
                    comment = '{comment}'""")
    cur.execute(f"grant usage on warehouse {warehouse} to role {role}")
    cur.execute(f"""create user if not exists {user} type = service default_role = {role}
                    default_warehouse = {warehouse} comment = '{comment}'""")
    cur.execute(f"alter user {user} set rsa_public_key = '{public_key_body(pathlib.Path(identity['private_key_file']))}'")
    cur.execute(f"grant role {role} to user {user}")
    grants = [f"{row['privilege']} on {row['granted_on']} {row['name']}" for row in records(cur, f"show grants to role {role}")
              if row["granted_on"] in ("WAREHOUSE", "DATABASE", "ACCOUNT", "ROLE")]
    print("  grants:", grants[:12])
# The platform's role resets what the checks role owns, so it inherits that role.
cur.execute(f"grant role {SNOWFLAKE['checks']['role']} to role {SNOWFLAKE['pool']['role']}")
print("users:", [row["name"] for row in records(cur, "show users")])
print("warehouses:", [(row["name"], row["size"], row["auto_suspend"]) for row in records(cur, "show warehouses")])
