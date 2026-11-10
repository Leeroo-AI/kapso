"""A small set of reusable check databases, each a zero-copy clone of the check source, reset after every check.

Why not a fresh clone per check, as the benchmark's runner does: one clone of this warehouse (3,492 tables)
takes four to five minutes and about 0.31 credits of cloud services, measured on 2026-11-05. A campaign's
checks would spend the account. So a check database is cloned once and reused, and after each check it is put
back to the source's state and verified:

  - every table or view created or altered since the database was last clean is restored from the source by a
    table-level clone (or dropped, when the source has no such object), and handed back to the checks role;
  - every schema the source does not have is dropped;
  - every source object the check dropped is restored;
  - then the whole inventory (schema, name, kind, rows, bytes of every object) is compared with the source's.
    Equal: the database is marked clean with Snowflake's own clock. Not equal: it is marked dirty, never handed
    out again, and replaced by a fresh clone.

A database's comment says where it stands: "clean since <time>", "in use by <label> since <time>",
"preparing since <time>", "dirty: <why>". The checks role owns what is inside a check database and may create
schemas in it, but does not own the database, so the benchmark's own cleanup step (which drops "its" clone)
fails harmlessly and the database survives its check. Acquiring is serialised across processes on this host by
a file lock. The quick statements share one connection, reopened when it is older than the configured age.

    python pool.py status                 # what exists
    python pool.py adopt                  # take existing unmarked clones as check databases (verify, mark clean)
    python pool.py fill [<target>]        # clone new check databases until the target exists, then exit
    python pool.py reset <database>       # reset and verify one database by hand
    python pool.py drain                  # drop every check database
"""
import fcntl
import json
import pathlib
import secrets
import sys
import threading
import time

import snowflake.connector
from cryptography.hazmat.primitives import serialization

# The platform's one config file, beside this directory: copy config.example.json to config.json and fill it in.
CONFIG_FILE = pathlib.Path(__file__).resolve().parent.parent / "config.json"
PREPARING = "preparing since "
CLEAN = "clean since "
IN_USE = "in use by "
DIRTY = "dirty: "


def load_config():
    return json.loads(CONFIG_FILE.read_text())


def connect(snowflake_config, identity):
    """A key-pair connection as one of the configured identities ("checks" or "pool")."""
    who = snowflake_config[identity]
    key = serialization.load_pem_private_key(pathlib.Path(who["private_key_file"]).read_bytes(), password=None)
    der = key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    return snowflake.connector.connect(account=snowflake_config["account"], user=who["user"], private_key=der,
                                       role=who["role"], warehouse=who.get("warehouse"), client_session_keep_alive=True)


class CheckDatabases:
    def __init__(self, config):
        self.snowflake = config["snowflake"]
        self.settings = config["pool"]
        self.prefix = self.settings["clone_prefix"]
        self.source = self.snowflake["source_database"]
        self.checks_role = self.snowflake["checks"]["role"]
        self.creating = 0
        self.creating_lock = threading.Lock()
        self.shared = None
        self.shared_opened = 0.0
        self.shared_lock = threading.Lock()
        self.source_inventory = None

    # ---- statements -------------------------------------------------------------------------------------------

    def quick(self, statement):
        """Run one statement on the shared connection; returns its rows as dictionaries."""
        with self.shared_lock:
            too_old = time.time() - self.shared_opened > self.settings["connection_max_age_seconds"]
            if self.shared is None or self.shared.is_closed() or too_old:
                if self.shared is not None and not self.shared.is_closed():
                    self.shared.close()
                self.shared = connect(self.snowflake, "pool")
                self.shared_opened = time.time()
            cursor = self.shared.cursor()
            cursor.execute(statement)
            columns = [column[0] for column in cursor.description] if cursor.description else []
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def now(self):
        """Snowflake's clock, as text a later comparison can read back."""
        return self.quick("select to_varchar(current_timestamp(), 'YYYY-MM-DD HH24:MI:SS.FF3 TZHTZM') as now")[0]["NOW"]

    def mark(self, name, comment):
        self.quick(f"alter database {name} set comment = '{comment}'")

    # ---- inventory --------------------------------------------------------------------------------------------

    def inventory(self, database):
        """Every table and view of a database outside its information schema: (schema, name) -> (kind, rows, bytes)."""
        rows = self.quick(f"""
            select table_schema, table_name, table_type, row_count, bytes
            from {database}.information_schema.tables where table_schema <> 'INFORMATION_SCHEMA'""")
        return {(row["TABLE_SCHEMA"], row["TABLE_NAME"]): (row["TABLE_TYPE"], row["ROW_COUNT"], row["BYTES"]) for row in rows}

    def source_objects(self):
        """The source's inventory, read once: the source does not change while the service runs."""
        if self.source_inventory is None:
            self.source_inventory = self.inventory(self.source)
        return self.source_inventory

    def differences(self, name):
        """How a check database's inventory differs from the source's; an empty list means equal."""
        source, found = self.source_objects(), self.inventory(name)
        problems = [f"extra {schema}.{table}" for (schema, table) in found if (schema, table) not in source]
        problems += [f"missing {schema}.{table}" for (schema, table) in source if (schema, table) not in found]
        problems += [f"differs {schema}.{table}" for key, value in source.items() if key in found and found[key] != value
                     for (schema, table) in [key]]
        return problems

    # ---- the set ----------------------------------------------------------------------------------------------

    def databases(self):
        """Every database with the prefix: name and comment."""
        return [{"name": row["name"], "comment": row["comment"] or ""} for row in self.quick(f"show databases like '{self.prefix}%'")]

    def prepare(self, name):
        """Give an existing clone the privilege shape of a check database, verify it against the source, mark it."""
        self.quick(f"grant ownership on database {name} to role {self.snowflake['pool']['role']} revoke current grants")
        self.quick(f"grant usage, create schema on database {name} to role {self.checks_role}")
        problems = self.differences(name)
        if problems:
            self.mark(name, f"{DIRTY}differs from the source at preparation ({len(problems)} objects, first {problems[0]})")
            return False
        self.mark(name, f"{CLEAN}{self.now()}")
        return True

    def create_one(self):
        """Clone the source into a new check database and prepare it. Returns its name."""
        name = f"{self.prefix}{secrets.token_hex(6).upper()}"
        connection = connect(self.snowflake, "pool")
        try:
            connection.cursor().execute(f"create database {name} clone {self.source} comment = '{PREPARING}{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}'")
            self.prepare(name)
        finally:
            with self.creating_lock:
                self.creating -= 1
            connection.close()
        return name

    def adopt(self):
        """Prepare every existing clone that carries no comment or an old pool comment. Returns the names taken."""
        taken = []
        for database in self.databases():
            if database["comment"].startswith((CLEAN, IN_USE, DIRTY)):
                continue
            if self.prepare(database["name"]):
                taken.append(database["name"])
        return taken

    def acquire(self, label):
        """Take one clean check database for one check. Waits when none is clean."""
        deadline = time.time() + self.settings["acquire_wait_seconds"]
        lock_file = pathlib.Path(self.settings["lock_file"])
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        while True:
            with open(lock_file, "w") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                clean = sorted(database["name"] for database in self.databases() if database["comment"].startswith(CLEAN))
                listing = self.databases()
                clean = sorted((database["name"], database["comment"]) for database in listing if database["comment"].startswith(CLEAN))
                if clean:
                    name, comment = clean[0]
                    # The comment keeps when the database was taken and the moment it was last clean, which
                    # the reset reads back.
                    self.mark(name, f"{IN_USE}{label} at {int(time.time())} since {comment[len(CLEAN):]}")
                    return name
            if time.time() > deadline:
                raise RuntimeError(f"no clean check database within {self.settings['acquire_wait_seconds']} s")
            time.sleep(self.settings["acquire_poll_seconds"])

    def release(self, name):
        """Put a check database back to the source's state after its check, verify it, and mark it."""
        comment = [database["comment"] for database in self.databases() if database["name"] == name]
        if not comment:
            raise RuntimeError(f"{name} no longer exists; a check must not be able to drop its database")
        if not comment[0].startswith(IN_USE) or " since " not in comment[0]:
            raise RuntimeError(f"{name} is not in use: {comment[0]!r}")
        return self.reset(name, comment[0].rsplit(" since ", 1)[1])

    def reset(self, name, clean_since):
        """Restore everything touched since `clean_since`, verify the inventory, mark clean or dirty."""
        source = self.source_objects()
        source_schemas = {schema for schema, _ in source}
        for row in self.quick(f"show schemas in database {name}"):
            if row["name"] not in source_schemas and row["name"] not in ("INFORMATION_SCHEMA", "PUBLIC"):
                self.quick(f'drop schema if exists {name}."{row["name"]}" cascade')
        touched = self.quick(f"""
            select table_schema, table_name, table_type from {name}.information_schema.tables
            where table_schema <> 'INFORMATION_SCHEMA'
              and (last_altered > to_timestamp_tz('{clean_since}', 'YYYY-MM-DD HH24:MI:SS.FF3 TZHTZM')
                   or created > to_timestamp_tz('{clean_since}', 'YYYY-MM-DD HH24:MI:SS.FF3 TZHTZM'))""")
        present = self.inventory(name)
        restore = {(row["TABLE_SCHEMA"], row["TABLE_NAME"]) for row in touched} | {key for key in source if key not in present}
        restore |= {key for key in present if key not in source}
        for schema, table in sorted(restore):
            kind = present.get((schema, table), (None,))[0]
            target = f'{name}."{schema}"."{table}"'
            if kind == "VIEW":
                self.quick(f"drop view if exists {target}")
            elif kind is not None and (schema, table) not in source:
                self.quick(f"drop table if exists {target}")
            if (schema, table) in source:
                self.quick(f'create or replace table {target} clone {self.source}."{schema}"."{table}"')
                self.quick(f"grant ownership on table {target} to role {self.checks_role} revoke current grants")
        problems = self.differences(name)
        if problems:
            self.mark(name, f"{DIRTY}after a reset {len(problems)} objects differ, first {problems[0]}")
            return {"database": name, "clean": False, "restored": len(restore), "problems": problems[:5]}
        self.mark(name, f"{CLEAN}{self.now()}")
        return {"database": name, "clean": True, "restored": len(restore)}

    def recover_stale(self):
        """Reset the databases a check took and never gave back (the check's process died). Returns their names."""
        recovered = []
        limit = time.time() - 60 * self.settings["stale_in_use_minutes"]
        for database in self.databases():
            comment = database["comment"]
            if not comment.startswith(IN_USE) or " at " not in comment or " since " not in comment:
                continue
            taken_at = comment.rsplit(" at ", 1)[1].split(" since ", 1)[0]
            if taken_at.isdigit() and int(taken_at) < limit:
                self.reset(database["name"], comment.rsplit(" since ", 1)[1])
                recovered.append(database["name"])
        return recovered

    def replace_dirty(self):
        """Drop the databases marked dirty; fill() then brings the set back to its target."""
        dropped = [database["name"] for database in self.databases() if database["comment"].startswith(DIRTY)]
        for name in dropped:
            self.quick(f"drop database if exists {name}")
        return dropped

    def clones_left_today(self):
        """How many new clones the daily allowance still permits: a clone costs credits, and a fault that kept
        dirtying databases would otherwise keep buying new ones."""
        ledger = pathlib.Path(self.settings["clone_ledger_file"])
        counts = json.loads(ledger.read_text()) if ledger.exists() else {}
        return self.settings["max_new_clones_per_day"] - counts.get(time.strftime("%Y-%m-%d", time.gmtime()), 0)

    def count_clone(self):
        ledger = pathlib.Path(self.settings["clone_ledger_file"])
        counts = json.loads(ledger.read_text()) if ledger.exists() else {}
        today = time.strftime("%Y-%m-%d", time.gmtime())
        counts[today] = counts.get(today, 0) + 1
        ledger.write_text(json.dumps(counts) + "\n")

    def fill(self, target=None, wait=True):
        """Clone new check databases until `target` exist or are on their way, within the daily allowance; with
        wait, return when they exist."""
        target = self.settings["databases"] if target is None else target
        while True:
            existing = [database for database in self.databases() if database["comment"].startswith((CLEAN, IN_USE, PREPARING))]
            with self.creating_lock:
                creating = self.creating
            preparing = sum(1 for database in existing if database["comment"].startswith(PREPARING))
            missing = target - len(existing) - max(0, creating - preparing)
            for _ in range(max(0, min(missing, self.clones_left_today()))):
                with self.creating_lock:
                    self.creating += 1
                self.count_clone()
                threading.Thread(target=self.create_one, daemon=True).start()
            if not wait or (missing <= 0 and creating == 0) or self.clones_left_today() <= 0:
                return len(existing)
            time.sleep(15)

    def drain(self):
        names = [database["name"] for database in self.databases()]
        for name in names:
            self.quick(f"drop database if exists {name}")
        return names


if __name__ == "__main__":
    pool = CheckDatabases(load_config())
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        listing = pool.databases()
        clean = [database for database in listing if database["comment"].startswith(CLEAN)]
        print(f"{len(listing)} check databases, {len(clean)} clean")
        for database in sorted(listing, key=lambda database: database["name"]):
            print(f"  {database['name']}  {database['comment'] or '(no comment yet)'}")
    elif command == "adopt":
        print("adopted:", pool.adopt())
    elif command == "fill":
        started = time.time()
        count = pool.fill(int(sys.argv[2]) if len(sys.argv) > 2 else None)
        print(f"{count} check databases after {time.time() - started:.0f} s")
    elif command == "reset":
        print(pool.reset(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "2000-01-01 00:00:00.000 +0000"))
    elif command == "drain":
        print("dropped:", pool.drain())
    else:
        raise SystemExit(f"unknown command {command!r}")
