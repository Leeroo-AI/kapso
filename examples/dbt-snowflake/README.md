# Data engineering: dbt models on Snowflake

A retail warehouse on Snowflake: a dbt project of about 2,350 models and an open backlog of seventeen tickets,
reports with wrong numbers to fix and models to build. A ticket is done when its ticket check passes, and the
checks belong to a data platform the build cannot read: it sends the project and gets pass or fail per ticket.
The build runs on the bundled [`data-engineering`](../../src/kapso/harness/bundled/data-engineering) harness, which closes each ticket
with measured evidence, delivers one pull request per ticket, and releases what was merged to production
through the platform. It is the largest example here and the one with the most to set up: a Snowflake account,
a second machine, and the public benchmark the warehouse comes from. This README is that setup: what to bring,
how to stand each piece up, and the lines that run the build, deliver it, release it and learn from it.

## What you get

- `repo-template/` and `assemble.py`: the team repository. The template holds its own files (the README a
  build reads, the seventeen tickets, the clients of the platform under `checks/`, the small marketing project,
  `kapso-config.yaml`); `assemble.py` adds the warehouse's dbt project from a checkout of the benchmark, with
  the seven starting files of its fix tasks in place, and writes `repo/`.
- `kapso-brief.md`: the brief the build is launched with.
- `evaluation/`: the evaluator, a copy of the check client that Kapso protects, so a change to the
  repository's `checks/` changes nothing that is scored. The last line it prints is
  `tickets_passing: <share>`.
- `platform/`: what stands outside the build's reach, with its own README: the service that runs the ticket
  checks, resets the development database and releases what is merged, and the scripts that prepare a
  Snowflake account for it.
- `profiles.example.yml`, the engineer's dbt profile with its three targets; `open_issues.py`, which opens one
  GitHub issue per ticket if you want the backlog there too; `run.py`, the build from Python.
- From a build: `campaign/`, the record; one pull request per ticket on your repository; a release in your
  production database.

What the example does not include: the warehouse's dbt project, its data and the tickets' acceptance tests.
All three come from your own checkout of the benchmark, and the tests never leave the platform's machine.

## What you need

- **A Snowflake account you administer.** A trial is enough. As a guide from our own account's metering:
  the load is billed about 4 credits, nine hours of building about 15 on two X-Small warehouses, and
  delivery, release and learn about 2 more. A setup makes thirteen zero-copy clones (ten check databases,
  production, staging, development), each about 0.3 credits of cloud services. Never clone the warehouse per
  check, as the benchmark's own runner does: that is where a trial's credits go, and it is why `platform/`
  reuses its check databases.
- **A checkout of [data-eng-bench](https://github.com/Snowflake-Labs/data-eng-bench)** (Snowflake Inc.,
  Apache 2.0), with Docker for its base image.
- **Two machines**, or two users of one. The platform's side holds the benchmark's checkout, the account
  admin's login and the platform's keys. The build's side holds the team repository, the harness, the
  engineer's key and a tunnel to the platform's port. The benchmark's checkout carries reference solutions:
  a build that can read it proves nothing.
- **A private GitHub repository** that deletes a merged branch itself ("Automatically delete head branches"),
  and the `gh` CLI logged in on both sides: stacked pull requests move onto `main` only when GitHub removes
  the branch under them.
- **On the build's side**: Python 3.12 with `dbt-core` 1.10.22 and `dbt-snowflake` 1.10.3
  (`repo-template/requirements.txt`), Claude Code and Codex logged in, and Kapso from this repository's
  checkout or a release after 0.9.0: the two delivery targets read `KAPSO_SOURCE_REMOTE` and
  `KAPSO_VERIFICATION_DIR`, which 0.9.0 does not set.

## Set it up

1. **The platform.** On its machine, with [`platform/README.md`](platform/README.md) open: it explains each
   step, the config's keys and what each script changes in the account.

   ```bash
   cd platform
   python -m venv venv && venv/bin/pip install -r requirements.txt
   docker pull ghcr.io/snowflake-labs/data-eng-bench-base:1.0.0
   cp config.example.json config.json                       # then fill it in
   # load DBT_BENCH_RETAIL with the benchmark's own migrate_duckdb.py (its README, "Running (Snowflake)")
   venv/bin/python snowflake/setup_principals.py            # the identities, their warehouses and keys
   venv/bin/python snowflake/setup_check_source.py          # first fix: the checks role owns the loaded tables
   venv/bin/python snowflake/uppercase_columns.py --apply   # second fix: column names in upper case
   venv/bin/python snowflake/timezone.py account UTC
   venv/bin/python snowflake/setup_production.py            # production, staging, the release identity
   venv/bin/python snowflake/setup_dev_database.py          # development
   venv/bin/python service/pool.py fill                     # the check databases
   venv/bin/python service/mirror.py reset development
   venv/bin/python service/mirror.py reset staging
   ```
2. **The repository.** Where the benchmark's checkout is, from this directory:

   ```bash
   python assemble.py /path/to/data-eng-bench repo
   cd repo
   git init -b main && git add -A && git commit -m "The retail warehouse's dbt projects, the open backlog and the ticket checks' client"
   gh repo create YOUR-ORG/retail-dw --private --source . --push
   gh repo edit YOUR-ORG/retail-dw --delete-branch-on-merge
   ```

   `python open_issues.py YOUR-ORG/retail-dw repo` opens the seventeen issues, if you want them.
3. **The platform, again.** Name the repository in `platform/config.json` (`release.repository`), record the
   commit production starts at, start the service and give the build machine its door to it:

   ```bash
   cd platform
   venv/bin/python service/set_production_commit.py
   venv/bin/python service/server.py                        # the service, on 127.0.0.1:8765; it logs on stdout
   ssh -N -R 8765:localhost:8765 <build machine>            # from another terminal: the build machine's localhost:8765
   ```
4. **The build machine.** Clone this repository and install Kapso from it (`pip install -e .` at its root); in
   this directory, clone the team repository as `repo/` (`git clone <your repository> repo`) and install its
   `requirements.txt`. Copy `profiles.example.yml` to `~/.dbt/profiles.yml` with your account and the path of
   the engineer's private key, the one file of the platform's setup this machine receives. With the tunnel of
   step 3 open, the platform's service answers here on `localhost:8765`. Then check each piece:

   ```bash
   (cd repo/warehouse && dbt deps && dbt debug)
   python repo/checks/dev_reset.py              # the development database, back to production's state
   python repo/checks/run_checks.py --fast      # four tickets: 0 of 4 as shipped
   kapso harness check data-engineering
   kapso doctor build --config repo/kapso-config.yaml --models
   ```

   As shipped the repository passes none of the seventeen checks and its client exits 3: one model of the
   backlog calls a macro nothing defines, so the warehouse project does not parse until that ticket is fixed.
   That is the starting point a correct setup shows.

## Run it

In Claude Code with the Kapso plugin, opened in `repo/`, the build starts from one line:

```text
/kapso:build some of our dbt models on Snowflake have data quality issues, and we need a few new ones. each has an open ticket. fix and build them, and make sure they pass our production tests
```

The plugin reads the repository, finds the tickets, the check client and the harness its README names, writes
a brief and an evaluation copy of its own (the ones in this directory were written that way), and asks once
before it launches. The repository's README names the `data-engineering` harness, which ships inside the
package and loads by name.

The same build from the command line, from this directory:

```bash
kapso build --brief kapso-brief.md --repo repo --evaluation evaluation --output campaign --config repo/kapso-config.yaml --detach
kapso watch campaign --follow
```

`python run.py` is that build from Python, into the same `campaign/`: run one or the other.
`repo/kapso-config.yaml` sets the models, turns the build sessions' web tools off (the benchmark is public,
and the work is done from the tickets, the repository and the warehouse), gives the harness's setup and
verification sessions the time a backlog of this size needs, and lets the judge's "achieved" vote end the
campaign. The brief's budget is twelve hours and 34 experiments.

## Deliver and release

A finished build ships through the harness's two delivery targets, never through branches cut by hand:

```bash
kapso deploy campaign --target pull-request --config repo/kapso-config.yaml
```

One pull request per closed ticket, each changing only that ticket's files, stacked on another's only when its
code cannot build without it. Every branch is checked alone before anything is pushed, and each title and body
is rendered from verification's own measurement of the delivered code, with every point still open named.
Merging is yours: merge the pull request the others are stacked on first, with the merge button or
`gh pr merge <number> --merge`, and let GitHub delete its branch. A branch deleted from the command line closes
every pull request stacked on it.

```bash
kapso deploy campaign --target warehouse-release --config repo/kapso-config.yaml
kapso deploy campaign --status
```

The release target asks the platform to rehearse what `main` holds beyond production in the staging database,
audits the rehearsal with every released ticket's checks, publishes, audits production the same way, and
rolls the release back by itself when the publication fails or production does not hold. Stopping a release
deployment (`kapso deploy campaign --stop --number <n>`) is a rollback too: the platform puts production back
to before that release.

A release runs for the length of its audits, past the thirty minutes the engine gives a deployment session and
the ten it gives a target script, so `repo/kapso-config.yaml` sets `deployment.timeout` and
`deployment.script_timeout` to an hour each. Keep both lines: without them the engine cuts a release while it
publishes.

## Learn

```bash
kapso learn campaign --detach --config repo/kapso-config.yaml
kapso experience list
```

Learn reads the campaign's experiments and writes lesson cards into your bank, each a rule with the evidence
that earned it; a later build of the same domain can be served from it.

## What to look at

- `campaign/results/backlog.json` and `campaign/results/proofs/<ticket>.json`: the harness's ledger and each
  ticket's proof, with the evidence under `campaign/results/<ticket>/`.
- `campaign/.kapso/verification/<branch>/verdict.json` and `evidence_check.json`: the verification session's
  verdict and its re-measurement of every ticket, which is what a pull request is published on.
- `campaign/.kapso/deployments/<n>/`: each deployment's record; under `workspace/deploy/pull_requests/`,
  `opened.json` holds each pull request's label and open points and `branch_checks.json` each branch's check.
- `kapso trace campaign`: every attempt with its sessions, what it read and what was voided.
- `repo/checks/results/<run>.json` on the build machine, and the run's directory on the platform's machine,
  where the build logs stay.

## Provenance

The warehouse's dbt project, its data and the tickets derive from data-eng-bench by Snowflake Inc., used under
the Apache License, Version 2.0. `NOTICE.md` says what in this directory derives from it and what was changed,
and `repo-template/NOTICE.md` is the notice the assembled repository carries; Brightmere is a fictional
retailer of ours. A ticket's check is its benchmark task's own test suite, run by the platform as the
benchmark's runner would run it. The benchmark's repository is public, with reference solutions in it, which
is why the build sessions have no web tools and the brief forbids consulting it; after a build, search the
campaign's transcripts for it before you quote a result.
