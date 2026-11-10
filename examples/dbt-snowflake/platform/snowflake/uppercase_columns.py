"""Rename every column of the check source database that is stored in lower or mixed case to its upper-case name.

Why: the benchmark's loader creates each table with the column names quoted exactly as DuckDB has them, and DuckDB
has most modelled columns in lower case. Snowflake then stores "order_id" as a case-sensitive name, and a model
that says `order_id` (which Snowflake reads as ORDER_ID) fails with "invalid identifier". The benchmark's own
reference solutions fail that way on the database as loaded. Upper-case storage is what an account with
QUOTED_IDENTIFIERS_IGNORE_CASE on would have produced at load time, and what unquoted SQL needs.

A rename keeps the table, its rows, its owner and its grants. Run it after setup_check_source.py, as the account
admin. Usage:

    python uppercase_columns.py            # dry run: counts and collisions
    python uppercase_columns.py --apply    # rename, sixteen statements at a time
"""
import collections
import concurrent.futures
import sys
import threading
import time

from admin import SNOWFLAKE, connect

DATABASE = SNOWFLAKE["source_database"]
WORKERS = 16
local = threading.local()


def cursor():
    if not hasattr(local, "cursor"):
        local.cursor = connect().cursor()
    return local.cursor


def rename(item):
    schema, table, column = item
    cursor().execute(f'alter table {DATABASE}."{schema}"."{table}" rename column "{column}" to "{column.upper()}"')
    return item


main_cursor = cursor()
main_cursor.execute(f"""
    select table_schema, table_name, column_name from {DATABASE}.information_schema.columns
    where table_schema <> 'INFORMATION_SCHEMA' order by 1, 2, ordinal_position""")
columns = main_cursor.fetchall()
by_table = collections.defaultdict(list)
for schema, table, column in columns:
    by_table[(schema, table)].append(column)
to_rename, collisions = [], []
for (schema, table), names in by_table.items():
    upper_counts = collections.Counter(name.upper() for name in names)
    for name in names:
        if name != name.upper():
            if upper_counts[name.upper()] > 1:
                collisions.append((schema, table, name))
            else:
                to_rename.append((schema, table, name))
print(f"{len(columns)} columns in {len(by_table)} tables; {len(to_rename)} to rename; {len(collisions)} would collide")
for collision in collisions[:20]:
    print("  collision:", collision)
unusual = [item for item in to_rename if not item[2].replace("_", "").isalnum()]
print(f"{len(unusual)} names with characters beyond letters, digits and underscore:", unusual[:10])
if "--apply" not in sys.argv[1:]:
    print("dry run; pass --apply to rename")
    sys.exit(0)
started = time.time()
done = 0
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    for _ in executor.map(rename, to_rename):
        done += 1
        if done % 2000 == 0:
            print(f"  {done} renamed after {time.time() - started:.0f} s", flush=True)
print(f"{done} columns renamed in {time.time() - started:.0f} s")
main_cursor.execute(f"""
    select count(*) from {DATABASE}.information_schema.columns
    where table_schema <> 'INFORMATION_SCHEMA' and column_name <> upper(column_name)""")
print("columns still not upper case:", main_cursor.fetchone()[0])
