"""Read or set the TIMEZONE parameter of the account or of one user, as the account admin.

The warehouse's snapshots are stamped in UTC by dbt while `current_date` follows the session's time zone. With
Snowflake's account default (America/Los_Angeles) the two disagree for eight hours a day, and one ticket's
reference fix fails its acceptance check between 00:00 and 08:00 UTC. Setting the zone to UTC removes the
dependence on the hour.

    python timezone.py show                       # the account's value and each service user's
    python timezone.py account UTC                # set the account's value
    python timezone.py user <USER> <ZONE>         # set one user's session default (to reproduce the fault)
    python timezone.py user <USER> unset          # back to the account's value
"""
import sys

from admin import SNOWFLAKE, connect, records

USERS = tuple(SNOWFLAKE[name]["user"] for name in ("checks", "engineer", "release", "pool"))

cur = connect().cursor()
command = sys.argv[1] if len(sys.argv) > 1 else "show"
if command == "user":
    user, zone = sys.argv[2], sys.argv[3]
    cur.execute(f"alter user {user} unset timezone" if zone == "unset" else f"alter user {user} set timezone = '{zone}'")
elif command == "account":
    cur.execute(f"alter account set timezone = '{sys.argv[2]}'")
elif command != "show":
    raise SystemExit(f"unknown command {command!r}")

account = records(cur, "show parameters like 'TIMEZONE' in account")
print(f"account: {account[0]['value']} (level {account[0]['level'] or 'default'})")
existing = {row["name"] for row in records(cur, "show users")}
for user in USERS:
    if user not in existing:
        print(f"{user}: no such user")
        continue
    row = records(cur, f"show parameters like 'TIMEZONE' in user {user}")[0]
    print(f"{user}: {row['value']} (level {row['level'] or 'default'})")
print("now:", records(cur, "select to_varchar(current_timestamp(), 'YYYY-MM-DD HH24:MI TZH:TZM') as now, to_varchar(sysdate(), 'YYYY-MM-DD HH24:MI') as utc")[0])
