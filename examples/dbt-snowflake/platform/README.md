# The data platform

What stands outside the build's reach in this example: one small HTTP service that runs the ticket checks,
puts the development database back to production's state and releases what is merged, and the scripts that
make a Snowflake account ready for it. The team repository's `checks/` clients talk to the service; nothing
else of this directory, and nothing of the benchmark's checkout, belongs on a build machine.

## What is here

| Path | What it is |
|---|---|
| `config.example.json` | the one config file: copy it to `config.json` beside it and fill it in (below) |
| `requirements.txt` | the packages, at the versions we ran |
| `service/server.py` | the service: `POST /runs`, `GET /runs/<id>`, `GET /health`, `POST /dev/reset`, `POST /releases`, `GET /releases/<id>`, `POST /releases/<id>/publish`, `POST /releases/<id>/rollback`, `GET /production` |
| `service/check_runner.py` | one ticket's check: a container from the benchmark's base image, the ticket's recipe, the task's own tests |
| `service/pool.py` | the check databases: zero-copy clones of the source, reused, reset and verified after every check |
| `service/mirror.py` | the development database and the release staging database put back to production's state |
| `service/release.py` | a release of what is merged: staged, published after a backup, rolled back on request |
| `service/build_errors.py` | what a failed build may tell its submitter: the kind of each error and the names involved |
| `service/recipes/` | one build recipe per ticket, and the steps they share (`lib.sh`) |
| `service/tickets.json` | each ticket's benchmark task |
| `service/set_production_commit.py` | records the commit production starts at |
| `snowflake/` | the account setup, run once as the account admin: identities, the two fixes to the loaded warehouse, the time zone, production, staging, development |

## How a check works

`checks/run_checks.py` in the team repository sends `warehouse/` and `marketing/` as they stand in the working
tree. For each ticket the service takes a clean check database, starts the benchmark's base image the way the
task's compose file starts it, assigns the database exactly as the image's own clone step would, runs the
ticket's recipe and then the task's own `tests/test.sh`, and reads the verdict. A ticket passes only when all of
its tests pass. Per ticket the service answers pass or fail, how the build went, and for a failed build the kind
of each error with the names involved: the build's log stays on the platform's machine, since a model could
print the check environment into it, and the build runs with the key removed from its environment
(`recipes/lib.sh`). The tests' code, output and expected values never leave the machine either.

A recipe is the scaffolding the benchmark's own runner gives a task (the connection profile, the schema steps,
the dbt commands) with the submitted project in place of any reference file. A recipe never writes a model:
what is built is what was submitted.

**Check databases are reused, not cloned per check.** The benchmark's runner clones the warehouse afresh for
every trial. On this warehouse (3,492 tables) one clone took four to five minutes and about 0.31 credits of
cloud services, measured by us on 2026-11-05: no data is copied, but the services layer writes metadata for
every table. At that price a hundred checks cost about thirty credits.

So the service keeps a small set of check databases. After each check, every table or view created or altered
since the database was last clean is restored from the source by a table-level clone or dropped, every schema
the source lacks is dropped, every object the check removed is restored, and the whole inventory (schema, name,
kind, rows and bytes of every object) is compared with the source's before the database is marked clean again.
One that does not compare equal is marked dirty, never handed out again, and replaced within a daily allowance
of new clones. The checks role owns what is inside a check database but not the database, so the image's own
cleanup step, which drops "its" clone, fails harmlessly. A ticket's check takes 40 to 120 seconds and a full
run about five minutes.

## The warehouse

| Database (example config) | What it is | Who writes |
|---|---|---|
| `DBT_BENCH_RETAIL` | the warehouse as loaded and repaired; the check databases are clones of it | nobody |
| `RETAIL_CLONE_*` | the check databases | the checks role, inside its own check |
| `RETAIL_DW` | production: a clone of the source; the release role owns schema `MAIN` | the release service only |
| `RETAIL_DW_STAGING` | where a release is rehearsed: put back to production's state at every request | the release service only |
| `RETAIL_DW_DEV` | development: put back to production's state by `POST /dev/reset` | the engineer role, which reads the other three and writes none of them |

Four service identities, each with a key pair: `checks` (a check container runs as it), `pool` (the platform
itself: it creates, resets and drops databases, and its role inherits the other three), `release` (builds in
staging and production) and `engineer` (the build sessions). Only the engineer's private key goes to a build
machine.

## The config

`config.json` holds every machine path, identity, database name and limit the service and the setup scripts
read. Paths are absolute. It holds no secret: keys and the admin's login are files it names.

| Key | What it is |
|---|---|
| `image` | the benchmark's base image, as its task compose files name it |
| `tasks_dir` | the `tasks/` directory of your data-eng-bench checkout: a check reads its task's `tests/` there |
| `tickets_file`, `recipes_dir` | `service/tickets.json` and `service/recipes` of this directory |
| `runs_dir` | where each run's submission, build logs and verdicts are kept; create it before the first command |
| `snowflake.account` | the account identifier, `ORGNAME-ACCOUNTNAME` |
| `snowflake.source_database`, `snowflake.schema` | the database the benchmark's loader fills and its modelled schema |
| `snowflake.admin` | for the setup scripts only: `login_file` (three lines: `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD` as a password or a programmatic access token, `SNOWFLAKE_ROLE`) and a `warehouse` the admin can use |
| `snowflake.warehouses` | size, auto-suspend and statement timeout of the warehouses the setup creates |
| `snowflake.checks`, `.pool`, `.release`, `.engineer` | each identity's `user`, `role`, `warehouse` and `private_key_file`; a key file that does not exist is generated by the setup. The pool's role is an existing one that may create databases (`SYSADMIN`), and the release identity uses a warehouse the setup of another identity creates (the checks' in the example) |
| `mirrors.production_database` | production |
| `mirrors.development`, `mirrors.staging` | each copy's `database`, the role that owns each schema in it (`schema_owners`, else `default_owner`) and the roles that read it |
| `mirrors.lock_dir`, `mirrors.restore_parallel` | where resets and releases take their file locks; how many objects a reset restores at a time |
| `release.repository`, `release.base_branch` | the team repository on GitHub, `OWNER/NAME`, and the branch production follows |
| `release.repository_cache`, `release.state_file`, `release.releases_dir` | the service's bare clone of the repository, the commit production stands at with the list of releases, and each release's directory |
| `release.dbt`, `release.profile_name`, `release.threads`, `release.build_timeout_seconds` | the dbt the release service runs, the profile name both projects use, and its limits |
| `pool.clone_prefix`, `pool.databases`, `pool.max_new_clones_per_day` | the check databases' name prefix, how many to keep, and how many new clones a day may cost |
| `pool.clone_ledger_file`, `pool.lock_file` | the day's clone count and the lock that serialises taking a check database |
| `pool.stale_in_use_minutes`, `pool.connection_max_age_seconds`, `pool.acquire_poll_seconds`, `pool.acquire_wait_seconds` | when a database a dead check never gave back is recovered, and the pool's waits |
| `container` | CPU and memory of a check container, and the build's and the tests' time limits |
| `max_parallel_checks` | checks running at once across all runs |
| `server` | the loopback address and port the service listens on, and the largest submission it accepts |

## Set it up

On the platform's machine: Python 3.12, Docker, the `gh` CLI logged in with read access to the team repository
(the release service fetches the base branch through it), and a checkout of
[data-eng-bench](https://github.com/Snowflake-Labs/data-eng-bench).

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
docker pull ghcr.io/snowflake-labs/data-eng-bench-base:1.0.0
cp config.example.json config.json      # then fill it in
```

`config.json` names three things you create by hand before the first command: `runs_dir`, the admin's login
file (three lines, in the table above) and, if you want the keys somewhere particular, their directory.
`release.dbt` is the `dbt` of the environment just made.

1. **Load the warehouse** with the benchmark's own loader, as its README describes under "Running
   (Snowflake)": extract `retail.duckdb` from the base image and run `base-image/migrate_duckdb.py` on it. It
   creates `DBT_BENCH_RETAIL`. Our load on 2026-10-26 uploaded 3,492 of 3,493 tables in 20 schemas
   (`MAIN.STG_SAP__GOODS_MOVEMENTS` failed on a DuckDB numeric overflow; no ticket reads it) and the day was
   billed 4.1 credits. With snowflake-connector-python 3.18 the loader needs `SNOWFLAKE_HOST` set
   (`<account>.snowflakecomputing.com`): it passes `host=None` otherwise.
2. **The identities and their warehouses**: `venv/bin/python snowflake/setup_principals.py`.
3. **The first fix, ownership**: `venv/bin/python snowflake/setup_check_source.py`. The loader leaves every
   table owned by the role that ran it, and a clone keeps that, so a check could not replace a model's table
   in its own clone. The source's schemas and tables now belong to the checks role, which has no access to the
   source itself.
4. **The second fix, column names**: `venv/bin/python snowflake/uppercase_columns.py` for the count, then
   again with `--apply`. The loader quotes column names as DuckDB has them, so they are stored in lower case
   and a model that says `order_id` fails with "invalid identifier"; 19,158 columns were renamed in our
   account. Before the two fixes the benchmark's own reference solutions passed 6 of its 19 fix tasks there.
5. **The time zone**: `venv/bin/python snowflake/timezone.py account UTC`. dbt stamps snapshots in UTC while
   `current_date` follows the session; on Snowflake's default zone one ticket's check depended on the hour.
6. **Production and staging**: `venv/bin/python snowflake/setup_production.py`.
7. **Development**: `venv/bin/python snowflake/setup_dev_database.py`.
8. **The check databases**: `venv/bin/python service/pool.py fill`, then `venv/bin/python service/pool.py status`.
   Each new clone is counted against `pool.max_new_clones_per_day`.
9. **Record the copies' state**: `venv/bin/python service/mirror.py reset development` and
   `venv/bin/python service/mirror.py reset staging`.
10. **Production's commit**, once the team repository is pushed (the example's README, "Set it up"):
    `venv/bin/python service/set_production_commit.py`.
11. **Start the service**: `venv/bin/python service/server.py` (it logs on stdout), then
    `curl http://127.0.0.1:8765/health`.

A build machine reaches the service on its own `localhost:8765` through a tunnel, for instance
`ssh -N -R 8765:localhost:8765 <build machine>` run on the platform's machine. The engineer's private key and a
dbt profile (`../profiles.example.yml`) are the only other things of this setup a build machine holds.

Day to day:

```bash
venv/bin/python service/pool.py status              # the check databases and where each stands
venv/bin/python service/mirror.py status            # the development and staging copies
venv/bin/python service/release.py status           # production's commit and the releases
venv/bin/python service/check_runner.py <project tree> <ticket> [<ticket> ...]   # a check by hand, without the service
```

Run one service on a pool: a second one would hand out the same check databases.

## What was run from this directory, and what was not

- The service modules are the code that ran behind our own builds, changed in two places: the config file
  moved from beside the modules to this directory, and the build-error classifier builds its check database
  pattern from `pool.clone_prefix` where it carried the prefix as a literal.
- On 2026-11-10 we ran the service from a copy of this directory against our account and sent two project
  trees through the published `checks/run_checks.py`: every ticket's reference fix, which passed all seventeen
  checks, and the repository as `assemble.py` builds it, which passed none and made the client exit 3, as it
  should. A development reset through `checks/dev_reset.py` restored 16 objects in 20 seconds, the shipped
  evaluator's `--fast` spot check passed its four tickets on the reference fixes, and the classifier returned
  the same lines as the earlier one on 123 stored failed builds.
- The setup scripts are the ones that prepared our account, with every name read from the config where the
  originals carried it as a constant, the platform's own identity created by `setup_principals.py` (it was
  created by a separate script before), and the key-pair helper shared. On 2026-11-10 `timezone.py show` and the
  dry run of `uppercase_columns.py` ran from this directory against that account. The scripts that create or
  replace identities and databases were not run again there, so a fresh account is their first full run in
  this form: read each before running it.
- The release paths were not exercised on 2026-11-10 (a release publishes to production). They last ran on
  2026-11-07.
