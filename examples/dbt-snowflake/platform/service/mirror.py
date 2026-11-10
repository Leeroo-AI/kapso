"""Put a working copy of the production warehouse back to production's state: the development database
before an attempt or a ticket, the release staging database before a release.

The same idea as the check databases' reset (pool.py), for databases that are not handed out per check:

  - every schema production does not have is dropped;
  - every table or view created or altered since the copy was last in production's state is restored from
    production, a table by a table-level zero-copy clone and a view by its own definition with the copy's
    database in production's place (a view cannot be cloned), or dropped when production has no such object,
    and handed to the role that owns that schema in the copy;
  - every production object the copy lost is restored;
  - then the whole inventory (schema, name, kind, rows, bytes) is compared with production's, read at that
    moment (production changes when a release is published).

The copy's comment says where it stands: "in the state of production since <time>" or "differs: <why>" (no
apostrophe: the text is written into a SQL literal). A reset
restores a handful of objects, so it takes seconds and costs a few cloud-services statements, not a clone of
the whole warehouse.

    python mirror.py status
    python mirror.py reset development|staging
"""
import concurrent.futures
import fcntl
import json
import pathlib
import re
import sys
import threading
import time

from pool import connect, load_config

IN_STATE = "in the state of production since "
DIFFERS = "differs: "
STAMP = "YYYY-MM-DD HH24:MI:SS.FF3 TZHTZM"


class Mirror:
    def __init__(self, config, which):
        self.snowflake = config["snowflake"]
        self.settings = config["mirrors"][which]
        self.which = which
        self.database = self.settings["database"]
        self.source = config["mirrors"]["production_database"]
        self.lock_file = pathlib.Path(config["mirrors"]["lock_dir"]) / f"{which}.lock"
        self.parallel = config["mirrors"]["restore_parallel"]
        self.connection = None

    def restore_objects(self, statements_per_object):
        """Run each object's statements in order, several objects at a time, every worker on a connection of
        its own: a build that rebuilt a hundred tables is undone in seconds, not minutes."""
        local, opened = threading.local(), []

        def restore(statements):
            if getattr(local, "connection", None) is None:
                local.connection = connect(self.snowflake, "pool")
                opened.append(local.connection)
            cursor = local.connection.cursor()
            for statement in statements:
                cursor.execute(statement)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel) as executor:
            list(executor.map(restore, statements_per_object))
        for connection in opened:
            connection.close()

    def quick(self, statement):
        if self.connection is None or self.connection.is_closed():
            self.connection = connect(self.snowflake, "pool")
        cursor = self.connection.cursor()
        cursor.execute(statement)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def inventory(self, database):
        rows = self.quick(f"""
            select table_schema, table_name, table_type, row_count, bytes, is_transient
            from {database}.information_schema.tables where table_schema <> 'INFORMATION_SCHEMA'""")
        return {(row["TABLE_SCHEMA"], row["TABLE_NAME"]): (row["TABLE_TYPE"], row["ROW_COUNT"], row["BYTES"], row["IS_TRANSIENT"]) for row in rows}

    def view_definition(self, schema, table):
        """Production's own `create or replace view` statement for a view, with the copy's database wherever
        production's is named: in the view's name and in every relation the definition reads. A view is not
        cloned (Snowflake refuses), and a definition left on production's name would read production."""
        ddl = self.quick(f"""select get_ddl('view', '{self.source}."{schema}"."{table}"', true) as ddl""")[0]["DDL"]
        renamed = re.sub(rf'(?<![\w"]){re.escape(self.source)}(?=\s*\.)', self.database, ddl, flags=re.IGNORECASE)
        renamed = re.sub(rf'"{re.escape(self.source)}"(?=\s*\.)', f'"{self.database}"', renamed, flags=re.IGNORECASE)
        return renamed.rstrip().rstrip(";")

    def comment(self):
        rows = self.quick(f"show databases like '{self.database}'")
        if not rows:
            raise RuntimeError(f"{self.database} does not exist")
        return rows[0]["comment"] or ""

    def owner_of(self, schema):
        return self.settings["schema_owners"].get(schema, self.settings["default_owner"])

    def reset(self):
        """Restore, verify, mark. Returns {"database", "clean", "restored", "seconds", "problems"?}."""
        started = time.time()
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            comment = self.comment()
            since = comment[len(IN_STATE):] if comment.startswith(IN_STATE) else None
            source = self.inventory(self.source)
            source_schemas = {row["name"] for row in self.quick(f"show schemas in database {self.source}")} - {"INFORMATION_SCHEMA", "PUBLIC"}
            dropped_schemas, copy_schemas = [], set()
            for row in self.quick(f"show schemas in database {self.database}"):
                if row["name"] in ("INFORMATION_SCHEMA", "PUBLIC"):
                    continue
                if row["name"] not in source_schemas:
                    self.quick(f'drop schema if exists {self.database}."{row["name"]}" cascade')
                    dropped_schemas.append(row["name"])
                else:
                    copy_schemas.add(row["name"])
            for schema in sorted(source_schemas - copy_schemas):
                # A schema a release created in production (the snapshot's, for one).
                self.quick(f'create schema {self.database}."{schema}"')
                self.quick(f'grant ownership on schema {self.database}."{schema}" to role {self.owner_of(schema)} revoke current grants')
                for reader in self.settings["readers"]:
                    self.quick(f'grant usage on schema {self.database}."{schema}" to role {reader}')
            # Without the time of the last known state (a copy this module has not reset yet) everything would
            # count as touched; then only what the inventories show as different is restored.
            touched = [] if since is None else self.quick(f"""
                select table_schema, table_name from {self.database}.information_schema.tables
                where table_schema <> 'INFORMATION_SCHEMA'
                  and (last_altered > to_timestamp_tz('{since}', '{STAMP}') or created > to_timestamp_tz('{since}', '{STAMP}'))""")
            present = self.inventory(self.database)
            restore = {(row["TABLE_SCHEMA"], row["TABLE_NAME"]) for row in touched}
            restore |= {key for key in source if key not in present}
            restore |= {key for key in present if key not in source}
            restore |= {key for key, value in source.items() if key in present and present[key] != value}
            work = []
            for schema, table in sorted(restore):
                kind = present.get((schema, table), (None,))[0]
                source_kind = source.get((schema, table), (None,))[0]
                target = f'{self.database}."{schema}"."{table}"'
                statements = []
                if kind == "VIEW":
                    statements.append(f"drop view if exists {target}")
                elif kind is not None and ((schema, table) not in source or source_kind == "VIEW"):
                    statements.append(f"drop table if exists {target}")
                if source_kind == "VIEW":
                    statements.append(self.view_definition(schema, table))
                    statements.append(f"grant ownership on view {target} to role {self.owner_of(schema)} revoke current grants")
                    statements += [f"grant select on view {target} to role {reader}" for reader in self.settings["readers"]]
                elif (schema, table) in source:
                    # A clone keeps the source's kind: a transient table (dbt's default) cannot be cloned into a permanent one.
                    transient = "transient " if source[(schema, table)][3] == "YES" else ""
                    statements.append(f'create or replace {transient}table {target} clone {self.source}."{schema}"."{table}"')
                    statements.append(f"grant ownership on table {target} to role {self.owner_of(schema)} revoke current grants")
                    statements += [f"grant select on table {target} to role {reader}" for reader in self.settings["readers"]]
                work.append(statements)
            self.restore_objects(work)
            found = self.inventory(self.database)
            problems = [f"extra {schema}.{table}" for (schema, table) in found if (schema, table) not in source]
            problems += [f"missing {schema}.{table}" for (schema, table) in source if (schema, table) not in found]
            problems += [f"differs {key[0]}.{key[1]}" for key, value in source.items() if key in found and found[key] != value]
            now = self.quick(f"select to_varchar(current_timestamp(), '{STAMP}') as now")[0]["NOW"]
            if problems:
                self.quick(f"alter database {self.database} set comment = '{DIFFERS}{len(problems)} objects after a reset, first {problems[0]}'")
            else:
                self.quick(f"alter database {self.database} set comment = '{IN_STATE}{now}'")
        outcome = {"database": self.database, "clean": not problems, "restored": len(restore), "dropped_schemas": dropped_schemas,
                   "seconds": round(time.time() - started, 1)}
        if since is None:
            outcome["note"] = "no record of the last known state: restored what the inventories show as different"
        if problems:
            outcome["problems"] = problems[:5]
        return outcome


if __name__ == "__main__":
    configuration = load_config()
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        for name in ("development", "staging"):
            mirror = Mirror(configuration, name)
            print(f"{name}: {mirror.database}  {mirror.comment() or '(no comment yet)'}")
    elif command == "reset":
        print(json.dumps(Mirror(configuration, sys.argv[2]).reset(), indent=1))
    else:
        raise SystemExit(f"unknown command {command!r}")
