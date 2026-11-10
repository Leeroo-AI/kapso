# Brightmere retail data warehouse

The dbt code behind Brightmere's analytics warehouse on Snowflake.

- `warehouse/`: the main dbt project (`retail_dw_master`): sources, staging, intermediate and mart models
  over the retail systems (SAP, POS, GA, Salesforce, WMS, payments), about 2,350 models.
- `marketing/`: the marketing reports' own small dbt project (`marketing_marts`). Its models are tables in the
  `ANALYTICS` schema and read the GA models the warehouse project builds in `MAIN`.
- `tickets/`: the open backlog, one file per ticket.
- `checks/`: the data platform's clients (the ticket checks, the development reset, the release) and the list
  of tickets the checks know.
- `kapso-config.yaml`: the models and limits our Kapso builds run on.

## The backlog

Seventeen open tickets, one file each under `tickets/`, named by the key the checks use:

| Area | Tickets |
|---|---|
| Revenue | `daily-revenue`, `category-revenue`, `midnight-sales`, `return-reconciliation`, `repeat-purchase-cohorts` |
| Customer | `customer-dimension`, `customer-ltv`, `customer-snapshot`, `customer-activity`, `daily-cohorts` |
| Inventory | `inventory-metrics`, `inventory-balance`, `product-metrics` |
| Marketing | `email-attribution`, `multi-touch-attribution`, `paid-search-attribution`, `cac-payback` |

Some tickets fix a model that produces wrong numbers, some ask for a model that does not exist yet, and the
marketing ones rebuild a report that broke as a new `*_fixed` model. Each ticket names its model file, its
output table and what has to hold.

**The bar:** a ticket is done when its ticket check passes, and the backlog is done when every ticket's check
passes: `python checks/run_checks.py` prints `tickets_passing: 1.0000`. The ticket checks are what the team
calls its production tests: a model is ready for production when its ticket check passes.

## At a glance

How this team builds, checks, resets and releases; the sections below give the detail.

| What | Command or dbt target |
|---|---|
| Build | target `dev` (the default), database `RETAIL_DW_DEV` |
| Read what production holds | target `production`, read only |
| Read a rehearsed release | target `staging`, read only |
| Put development back to production's state | `python checks/dev_reset.py` |
| Check one ticket | `python checks/run_checks.py --ticket <ticket>`; its line reads `PASS  <ticket>` or `FAIL  <ticket>`; up to four at a time, each built from the working tree in a check database of its own, never the development database |
| Check the backlog | `python checks/run_checks.py`; the last line is `tickets_passing: <share>` |
| Deliver a fix | one pull request per ticket against `main` |
| Release what is merged | `python checks/release.py request`, then `publish <release>`; `rollback <release>` undoes it |

## The ticket checks

```bash
python checks/run_checks.py                          # the whole backlog; prints tickets_passing last
python checks/run_checks.py --ticket daily-revenue   # one ticket, a spot check
python checks/run_checks.py --fast                   # four tickets, one per area, a spot check
```

The checks belong to the data platform, not to this repository. The script sends `warehouse/` and
`marketing/` as they stand in the working tree to the check service. For each ticket the service takes a check
database of its own (a zero-copy clone of the warehouse as it stood when this backlog was filed, restored and
verified after every check), builds that ticket's models there from what was sent, and runs the ticket's
acceptance checks on the result. It answers pass or fail per ticket, and for a build that failed the kind of
each error with the names involved. The script prints one line per ticket, `PASS  <ticket>` or
`FAIL  <ticket>`, a failed build's errors beneath it, then the count and the share as the last line.

- The acceptance checks themselves are not in this repository and are not shown: a fix is written from the
  ticket and the data, and the check says whether it holds.
- A ticket passes only when all of its acceptance checks pass.
- Only the ticket's own models are built. Every other model is read as the warehouse held it when the backlog
  was filed, so a ticket's check never sees another ticket's fix, even of a model it reads.
- A failed build's log is not returned. The same build in the development database, which holds the same
  data, shows the full message.
- The check of a marketing ticket first rebuilds `stg_ga__sessions`, `stg_ga__events` and
  `int_sessions_events_joined` from the submitted warehouse project, then builds the report with the marketing
  project's shipped configuration (a table in `ANALYTICS`) and its macros; changes to
  `marketing/dbt_project.yml` are not carried into a check. The paid search report is built inside the
  warehouse project's directory and sees both projects' macros.
- A full run takes about five minutes and a single ticket one to two. Up to four single-ticket runs may be
  started at the same time (each takes a check database of its own); more than the service can run are queued.
  The service runs on Snowflake compute of its own; nothing is billed to the development warehouse.
- The service listens on `localhost:8765` on a build machine (`checks/service.json`). If the script says the
  service is not reachable, the tunnel to the check host is down: stop and say so; nothing can be scored
  without it.
- `checks/` and `tickets/` are not modified by a fix.

## Working on the warehouse

- **Connection.** dbt reads the profile `retail_dw_master` from `~/.dbt/profiles.yml` on the machine (not in
  this repository): Snowflake, key-pair authentication, role `BRIGHTMERE_ENGINEER`, warehouse
  `BRIGHTMERE_DEV_WH`, schema `main`. Both projects use it. It has three targets, and every session of the
  warehouse runs on UTC:

  | Target | Database | What the role may do |
  |---|---|---|
  | `dev` (the default) | `RETAIL_DW_DEV` | build: it owns the modelled layer (`MAIN`), can create tables in `ANALYTICS` and can create schemas; the source and raw schemas are read-only |
  | `production` | `RETAIL_DW` | read only: production as it stands |
  | `staging` | `RETAIL_DW_STAGING` | read only: where a release is rehearsed before it reaches production |

  The targets are named in full on purpose: the project's `generate_schema_name` macro sends a target called
  `prod` to the `ANALYTICS_*` schemas, so no target carries that name.

- **The development database.** `RETAIL_DW_DEV` is a zero-copy clone of production. A build replaces the
  table production's copy held, and every model that reads it then reads the replacement.
  `python checks/dev_reset.py` puts the database back to production's state in seconds (up to a minute after
  a build that replaced a hundred tables); run it before measuring anything that should start from what
  production holds.
- **Production is read, never built by hand.** `dbt show --target production` and any query against `RETAIL_DW`
  read what the business sees today. A fix reaches production through a pull request and a release (below).
- **Build with selection.** `cd warehouse && dbt deps` once, then `dbt run --select <model>` (and
  `dbt snapshot --select <snapshot>`). Never build the whole project: it is about 2,350 models.
- **Where models land.** Every model of the warehouse project lands in schema `MAIN`. The `models:` block of
  `warehouse/dbt_project.yml` is keyed to an old project name, so its folder-level settings do not apply; a
  model without a `config()` of its own is a view. Leave that block alone: changing it would move and
  re-materialise every model in the project, which no ticket asks for.
- **Staging models are shared.** No ticket is fixed by editing a staging model.
- **Marketing reports.** `cd marketing && dbt run --select <report>`. They cannot `ref()` warehouse models;
  they read `main.<model>` through `adapter.get_relation`, as the tickets describe.

## Pull requests

A fix ships as a pull request against `main`: one per ticket, changing only that ticket's files, so each can be
reviewed, merged and released on its own. A ticket whose model cannot build without another ticket's change is
stacked on that ticket's pull request: it targets that branch, and its diff shows the ticket alone.

Merging a stack: merge the pull request the others are stacked on first, with the merge button or
`gh pr merge <number> --merge`. This repository deletes a merged branch itself, and GitHub then moves the
stacked pull requests onto `main`. Never delete such a branch from the command line (`--delete-branch`,
`git push --delete`): that closes every pull request stacked on it.

## Releasing

Production takes only what is merged on `main`: the data platform fetches the commit from GitHub itself, so
merging is the approval and nothing in a working tree is released.

```bash
python checks/release.py request            # stage what main holds beyond production's commit, and wait
python checks/release.py publish <release>  # build the staged release in production
python checks/release.py rollback <release> # put production back to before that release
python checks/release.py production         # the commit production stands at
```

`request` puts `RETAIL_DW_STAGING` back to production's state and builds there every model and snapshot the
commit changed since the last release; production is untouched, and staging is readable for an audit
(`--target staging`). `publish` is accepted only when everything built in staging: the platform backs up what
the release replaces, then runs the same build on production. `rollback` undoes whatever part of a
publication reached production, also one that failed or stopped midway, and is refused once a later release
has been published on top.

A release builds the models and snapshots whose own files the commit changed. A model whose file did not
change is not rebuilt, even when a merged change alters what it produces (a macro it calls, a model it
reads); it follows at its next scheduled build, unless the request names it:
`python checks/release.py request --also <model> [<model> ...]` (`<project>/<model>` for a model outside the
warehouse project).

Every verb takes `--json` and then prints the release's record as one JSON object: `release` (its id),
`commit`, `state` (`staged`, `published` or `rolled_back` when the step held; otherwise the state it stopped
in, with `error`; `refused` when the platform would not start it), `staged` and `published` (one entry per
model or snapshot: `project`, which is the project's directory, `warehouse` or `marketing`; `resource`, its
name; `status`; `relation`; `errors`) and `backed_up`. A verb exits 0 only when its step held.
`production --json` prints `released_commit`, and `status <release>` prints the record of a release at any
time.

## Building with Kapso

Builds with Kapso load the harness at `../harnesses/data-engineering`. `kapso-config.yaml` sets the models,
turns the build sessions' web tools off (the warehouse's work stays on the warehouse and the repository), and
gives the harness's setup and verification sessions the time a backlog of this size needs.

A finished build is delivered and released through Kapso's deployments, which that harness provides, never
by branches cut by hand. `kapso deploy <campaign> --target pull-request` opens the pull requests, one per
ticket. Once they are merged, `kapso deploy <campaign> --target warehouse-release` releases them through
the platform's release client above.

## Provenance

The dbt projects, the data and the tickets derive from a public benchmark; see `NOTICE.md`. Its public
repository, with its reference solutions and tests, is not consulted while the backlog is worked, online or
anywhere else: the work is done from the tickets, this repository and the warehouse.
