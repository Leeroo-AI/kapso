"""Release what is merged on the team repository's base branch into the production warehouse.

Production accepts only code that is on the base branch of the repository on GitHub: a release names a commit,
this module fetches the base branch itself and refuses a commit that is not on it. Merging a pull request is
therefore the approval; nothing a build machine uploads is ever built in production, and the release identity's
key never leaves this machine.

A release has three steps, each asked for on its own:

  stage     The staging database is put back to production's state (mirror.py), then the models and snapshots
            the commit changed since the last release are built there with dbt, as the release identity.
            Production is untouched. Whoever asked can read staging and audits it before going on.
  publish   Allowed only after a stage in which every selected resource built. Every relation the release
            replaces is backed up first (a zero-copy clone of a table, the text of a view), then the same build
            runs on production.
  rollback  Every relation the release replaced is put back from its backup, what it created is dropped, and
            production's inventory is compared with the one recorded before the publish.

State: runs/production_state.json holds the commit production stands at and the list of releases; each release
has a directory under runs/releases/ with the exported project, dbt's output and its record.

    python release.py status
    python release.py local-stage <project dir> <name> [<name> ...]     # private rehearsal from a local tree
    python release.py local-publish <release id> | local-rollback <release id>
"""
import fcntl
import json
import pathlib
import secrets
import shutil
import subprocess
import sys
import time

from build_errors import classify
from mirror import Mirror
from pool import connect, load_config

PROJECTS = ("warehouse", "marketing")
BACKUP_SCHEMA_PREFIX = "RELEASE_BACKUP_"
INTERNAL_SCHEMAS = ("INFORMATION_SCHEMA", "PUBLIC")


class ReleaseRefused(Exception):
    """A request this module will not carry out; the message is for whoever asked."""


class Releases:
    def __init__(self, config):
        self.config = config
        self.settings = config["release"]
        self.snowflake = config["snowflake"]
        self.production = config["mirrors"]["production_database"]
        self.staging = config["mirrors"]["staging"]["database"]
        self.releases_dir = pathlib.Path(self.settings["releases_dir"])
        self.state_file = pathlib.Path(self.settings["state_file"])
        self.lock_file = pathlib.Path(config["mirrors"]["lock_dir"]) / "release.lock"
        self.connection = None

    # ---- small helpers ------------------------------------------------------------------------------------------

    def quick(self, statement):
        if self.connection is None or self.connection.is_closed():
            self.connection = connect(self.snowflake, "pool")
        cursor = self.connection.cursor()
        cursor.execute(statement)
        columns = [column[0] for column in cursor.description] if cursor.description else []
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def state(self):
        if not self.state_file.exists():
            raise ReleaseRefused(f"{self.state_file} does not exist: production has no recorded commit (setup writes it)")
        return json.loads(self.state_file.read_text())

    def save_state(self, state):
        self.state_file.write_text(json.dumps(state, indent=1) + "\n")

    def record_path(self, release_id):
        return self.releases_dir / release_id / "release.json"

    def load(self, release_id):
        path = self.record_path(release_id)
        if not path.is_file():
            raise ReleaseRefused(f"no release {release_id}")
        return json.loads(path.read_text())

    def save(self, release):
        path = self.record_path(release["release"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(release, indent=1) + "\n")

    def git(self, *arguments):
        cache = self.settings["repository_cache"]
        completed = subprocess.run(["git", "--git-dir", cache, *arguments], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"git {' '.join(arguments[:3])} failed: {completed.stderr.strip()[-300:]}")
        return completed.stdout

    # ---- the commit and what it changes ------------------------------------------------------------------------

    def fetch_base(self):
        """The base branch's tip as GitHub has it now."""
        cache = pathlib.Path(self.settings["repository_cache"])
        url = f"https://github.com/{self.settings['repository']}.git"
        if not cache.exists():
            subprocess.run(["git", "init", "--bare", "-q", str(cache)], check=True)
        branch = self.settings["base_branch"]
        self.git("-c", "credential.helper=!gh auth git-credential", "fetch", "-q", url, f"+refs/heads/{branch}:refs/heads/{branch}")
        return self.git("rev-parse", branch).strip()

    def selection(self, previous, commit):
        """What the commit changes since `previous`, by file: resources to build by name, and the rest named."""
        selected = {project: {"models": [], "snapshots": []} for project in PROJECTS}
        other, deleted = [], []
        for line in self.git("diff", "--name-status", "--no-renames", previous, commit, "--", *PROJECTS).splitlines():
            status, path = line.split("\t", 1)
            parts = pathlib.PurePosixPath(path)
            project, kind = parts.parts[0], parts.parts[1] if len(parts.parts) > 2 else ""
            if status == "D":
                deleted.append(path)
            elif parts.suffix == ".sql" and kind == "models":
                selected[project]["models"].append(parts.stem)
            elif parts.suffix == ".sql" and kind == "snapshots":
                selected[project]["snapshots"].append(parts.stem)
            else:
                other.append(path)
        return {"build": selected, "other_changed_files": other, "deleted_files": deleted}

    def export(self, commit, target):
        target.mkdir(parents=True, exist_ok=True)
        archive = subprocess.run(["git", "--git-dir", self.settings["repository_cache"], "archive", commit, *PROJECTS],
                                 stdout=subprocess.PIPE, check=True).stdout
        subprocess.run(["tar", "-x", "-C", str(target)], input=archive, check=True)

    # ---- dbt ----------------------------------------------------------------------------------------------------

    def write_profile(self, directory):
        identity = self.snowflake["release"]
        outputs = {}
        for target, database in (("staging", self.staging), ("production", self.production)):
            outputs[target] = {"type": "snowflake", "account": self.snowflake["account"], "user": identity["user"],
                               "role": identity["role"], "private_key_path": identity["private_key_file"],
                               "warehouse": identity["warehouse"], "database": database, "schema": "main",
                               "threads": self.settings["threads"]}
        (directory / "profiles.yml").write_text(json.dumps({self.settings["profile_name"]: {"target": "staging", "outputs": outputs}}, indent=1) + "\n")

    def dbt(self, project_dir, arguments, log_file, profiles_dir, target):
        command = [self.settings["dbt"], "--no-use-colors", *arguments, "--profiles-dir", str(profiles_dir), "--target", target]
        with open(log_file, "a") as log:
            log.write(f"\n$ dbt {' '.join(arguments)} --target {target}\n")
            log.flush()
            completed = subprocess.run(command, cwd=project_dir, stdout=log, stderr=subprocess.STDOUT,
                                       timeout=self.settings["build_timeout_seconds"])
        return completed.returncode

    def build(self, release, target):
        """Build the release's selection on one target. Returns the per-resource results and whether all built."""
        directory = self.releases_dir / release["release"]
        log_file = directory / f"dbt_{target}.log"
        results = []
        for project in PROJECTS:
            chosen = release["selection"]["build"][project]
            names = chosen["models"] + chosen["snapshots"]
            if not names:
                continue
            project_dir = directory / "project" / project
            if (project_dir / "packages.yml").is_file():
                self.dbt(project_dir, ["deps"], log_file, directory, target)
            run_results = project_dir / "target" / "run_results.json"
            if run_results.exists():
                run_results.unlink()
            code = self.dbt(project_dir, ["build", "--select", *names, "--exclude-resource-type", "test", "unit_test", "--full-refresh"],
                            log_file, directory, target)
            if not run_results.exists():
                results.append({"project": project, "resource": "(the whole build)", "status": "error", "dbt_exit": code,
                                "errors": classify(log_file.read_text(errors="replace"), project_dir)})
                continue
            manifest = json.loads((project_dir / "target" / "manifest.json").read_text())
            for item in json.loads(run_results.read_text())["results"]:
                node = manifest["nodes"].get(item["unique_id"], {})
                entry = {"project": project, "resource": node.get("name") or item["unique_id"], "kind": node.get("resource_type"),
                         "status": item["status"], "relation": (node.get("relation_name") or "").replace('"', ""),
                         "seconds": round(item.get("execution_time") or 0, 1)}
                if item["status"] != "success":
                    entry["errors"] = classify(str(item.get("message") or ""), project_dir)
                results.append(entry)
            built = {entry["resource"] for entry in results if entry["project"] == project}
            for name in names:
                if name not in built:
                    results.append({"project": project, "resource": name, "status": "not_found",
                                    "errors": [f"the project at this commit has no model or snapshot named {name}"]})
        return results, bool(results) and all(entry["status"] == "success" for entry in results)

    # ---- inventory, backup, restore ----------------------------------------------------------------------------

    def inventory(self, database):
        rows = self.quick(f"""
            select table_schema, table_name, table_type, row_count, bytes from {database}.information_schema.tables
            where table_schema <> 'INFORMATION_SCHEMA' and table_schema not like '{BACKUP_SCHEMA_PREFIX}%'""")
        return {f'{row["TABLE_SCHEMA"]}.{row["TABLE_NAME"]}': [row["TABLE_TYPE"], row["ROW_COUNT"], row["BYTES"]] for row in rows}

    def transient_tables(self, database):
        """The tables of a database that are transient (dbt's default): a clone of one must be transient too."""
        rows = self.quick(f"""
            select table_schema, table_name from {database}.information_schema.tables
            where table_schema <> 'INFORMATION_SCHEMA' and table_type = 'BASE TABLE' and is_transient = 'YES'""")
        return {f'{row["TABLE_SCHEMA"]}.{row["TABLE_NAME"]}' for row in rows}

    def schemas(self, database):
        return sorted(row["name"] for row in self.quick(f"show schemas in database {database}")
                      if row["name"] not in INTERNAL_SCHEMAS and not row["name"].startswith(BACKUP_SCHEMA_PREFIX))

    def backup(self, release):
        """Back up every production relation the staged build produced a counterpart for."""
        backup_schema = f"{BACKUP_SCHEMA_PREFIX}{release['release'].upper().replace('-', '_')}"
        self.quick(f'create schema if not exists {self.production}."{backup_schema}"')
        before = self.inventory(self.production)
        transient = self.transient_tables(self.production)
        backups = []
        for entry in release["staged"]:
            relation = entry.get("relation") or ""
            if not relation:
                continue
            _, schema, name = [part.upper() for part in relation.split(".")]
            kind = before.get(f"{schema}.{name}", [None])[0]
            item = {"schema": schema, "name": name, "kind": kind}
            source = f'{self.production}."{schema}"."{name}"'
            if kind == "BASE TABLE":
                item["clone"] = f'{self.production}."{backup_schema}"."{schema}__{name}"'
                item["transient"] = f"{schema}.{name}" in transient
                self.quick(f"create or replace {'transient ' if item['transient'] else ''}table {item['clone']} clone {source}")
            elif kind == "VIEW":
                item["ddl"] = self.quick(f"select get_ddl('view', '{self.production}.{schema}.{name}', true) as ddl")[0]["DDL"]
            backups.append(item)
        return {"schema": backup_schema, "relations": backups, "inventory_before": before, "schemas_before": self.schemas(self.production)}

    def restore(self, release):
        backup = release["backup"]
        owner_of = self.config["mirrors"]["staging"]["schema_owners"]
        present = self.inventory(self.production)
        for item in backup["relations"]:
            target = f'{self.production}."{item["schema"]}"."{item["name"]}"'
            # Snowflake refuses `drop view` on a table and the reverse, even with `if exists`.
            kind_now = present.get(f'{item["schema"]}.{item["name"]}', [None])[0]
            if kind_now == "VIEW":
                self.quick(f"drop view {target}")
            elif kind_now is not None:
                self.quick(f"drop table {target}")
            if item["kind"] == "BASE TABLE":
                self.quick(f"create {'transient ' if item.get('transient') else ''}table {target} clone {item['clone']}")
                owner = owner_of.get(item["schema"], self.config["mirrors"]["staging"]["default_owner"])
                self.quick(f"grant ownership on table {target} to role {owner} copy current grants")
            elif item["kind"] == "VIEW":
                self.quick(item["ddl"])
        for schema in self.schemas(self.production):
            if schema not in backup["schemas_before"]:
                self.quick(f'drop schema if exists {self.production}."{schema}" cascade')
        after = self.inventory(self.production)
        before = backup["inventory_before"]
        problems = [f"extra {key}" for key in after if key not in before] + [f"missing {key}" for key in before if key not in after]
        problems += [f"differs {key}" for key, value in before.items() if key in after and after[key] != value]
        if not problems:
            self.quick(f'drop schema if exists {self.production}."{backup["schema"]}" cascade')
        return problems

    # ---- the three steps ---------------------------------------------------------------------------------------

    def new_release(self, commit, selection, source):
        release_id = f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-{secrets.token_hex(3)}"
        release = {"release": release_id, "state": "staging", "commit": commit, "previous_commit": self.state()["released_commit"],
                   "source": source, "selection": selection, "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        self.save(release)
        return release

    def request(self, commit=None, also=None):
        """Validate a release request against GitHub and record it; the caller then runs stage()."""
        tip = self.fetch_base()
        commit = commit or tip
        full = self.git("rev-parse", "--verify", f"{commit}^{{commit}}").strip()
        if subprocess.run(["git", "--git-dir", self.settings["repository_cache"], "merge-base", "--is-ancestor", full, tip]).returncode != 0:
            raise ReleaseRefused(f"{commit} is not on {self.settings['base_branch']} of {self.settings['repository']}: production takes only merged code")
        state = self.state()
        if full == state["released_commit"]:
            raise ReleaseRefused(f"production already stands at {full[:12]}: nothing to release")
        selection = self.selection(state["released_commit"], full)
        for name in also or []:
            # A model whose own file did not change: `<model>` in the warehouse project, `<project>/<model>` otherwise.
            project, _, model = str(name).rpartition("/")
            project = project or "warehouse"
            if project not in PROJECTS or not model:
                raise ReleaseRefused(f"an extra model is named <model> (warehouse) or <project>/<model>; {name!r} names no project of this repository")
            if model not in selection["build"][project]["models"]:
                selection["build"][project]["models"].append(model)
        if not any(selection["build"][project][kind] for project in PROJECTS for kind in ("models", "snapshots")):
            raise ReleaseRefused(f"{full[:12]} changes no model or snapshot since {state['released_commit'][:12]}")
        release = self.new_release(full, selection, f"{self.settings['repository']}@{self.settings['base_branch']}")
        self.export(full, self.releases_dir / release["release"] / "project")
        return release

    def stage(self, release_id):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.lock_file, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            release = self.load(release_id)
            directory = self.releases_dir / release_id
            self.write_profile(directory)
            release["staging_reset"] = Mirror(self.config, "staging").reset()
            if not release["staging_reset"]["clean"]:
                release["state"] = "stage_failed"
                self.save(release)
                return release
            release["staged"], all_built = self.build(release, "staging")
            release["state"] = "staged" if all_built else "stage_failed"
            release["staged_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.save(release)
            return release

    def publish(self, release_id):
        with open(self.lock_file, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            release = self.load(release_id)
            if release["state"] != "staged":
                raise ReleaseRefused(f"release {release_id} is {release['state']}: only a staged release in which everything built is published")
            state = self.state()
            if state["released_commit"] != release["previous_commit"]:
                raise ReleaseRefused("production moved since this release was staged: stage it again")
            release["backup"] = self.backup(release)
            release["state"] = "publishing"
            self.save(release)
            release["published"], all_built = self.build(release, "production")
            release["state"] = "published" if all_built else "publish_failed"
            release["published_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.save(release)
            if all_built:
                state["released_commit"] = release["commit"]
                state["releases"].append({"release": release_id, "commit": release["commit"], "at": release["published_at"]})
                self.save_state(state)
            return release

    def rollback(self, release_id):
        with open(self.lock_file, "w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            release = self.load(release_id)
            # Whatever part of a publication reached production is undone from the backup taken before it: a
            # release that completed, one whose build failed, and one whose publication stopped midway.
            if "backup" not in release or ("rollback" in release and release["rollback"]["clean"]):
                raise ReleaseRefused(f"release {release_id} has nothing in production to roll back")
            state = self.state()
            if state["released_commit"] not in (release["commit"], release["previous_commit"]):
                raise ReleaseRefused("a later release was published on top of this one: roll that one back first")
            problems = self.restore(release)
            release["rollback"] = {"clean": not problems, "problems": problems[:8], "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            release["state"] = "rolled_back" if not problems else "rollback_incomplete"
            self.save(release)
            if state["released_commit"] == release["commit"]:
                state["released_commit"] = release["previous_commit"]
                state["releases"] = [item for item in state["releases"] if item["release"] != release_id]
                self.save_state(state)
            return release

    def public_view(self, release):
        """What whoever asked may read: no inventories, no view texts."""
        view = {key: release[key] for key in ("release", "state", "commit", "previous_commit", "source", "selection", "requested_at",
                                               "staging_reset", "staged", "staged_at", "published", "published_at", "rollback", "error") if key in release}
        if "backup" in release:
            view["backed_up"] = [f'{item["schema"]}.{item["name"]} ({item["kind"] or "absent"})' for item in release["backup"]["relations"]]
        return view


if __name__ == "__main__":
    releases = Releases(load_config())
    command = sys.argv[1] if len(sys.argv) > 1 else "status"
    if command == "status":
        print(json.dumps(releases.state(), indent=1))
    elif command == "local-stage":
        # A private rehearsal: the selection is named by hand and the code is a local tree, never a build machine's upload.
        source_dir, names = pathlib.Path(sys.argv[2]).resolve(), sys.argv[3:]
        chosen = {project: {"models": [], "snapshots": []} for project in PROJECTS}
        for name in names:
            project, kind, resource = name.split(":")
            chosen[project][kind].append(resource)
        rehearsal = releases.new_release("local", {"build": chosen, "other_changed_files": [], "deleted_files": []}, f"local:{source_dir}")
        for project_name in PROJECTS:
            shutil.copytree(source_dir / project_name, releases.releases_dir / rehearsal["release"] / "project" / project_name,
                            ignore=shutil.ignore_patterns("target", "dbt_packages", "logs"))
        print(json.dumps(releases.public_view(releases.stage(rehearsal["release"])), indent=1))
    elif command == "local-publish":
        print(json.dumps(releases.public_view(releases.publish(sys.argv[2])), indent=1))
    elif command == "local-rollback":
        print(json.dumps(releases.public_view(releases.rollback(sys.argv[2])), indent=1))
    else:
        raise SystemExit(f"unknown command {command!r}")
