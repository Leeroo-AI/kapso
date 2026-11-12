---
name: dbt-workflow
description: The worked stack of this harness, with the dbt commands that keep an attempt fast and honest on a large project (selection, ad-hoc queries through dbt show, JSON listings), the project conventions in dbt terms (staging one-to-one with sources, the materialization ladder, tests, snapshots, contracts, compiled SQL as the debugging start), the branch and macro patterns for several engines, what the bundled scripts need, slim CI and its blind spots, and the runtime traps. Use before the first dbt command of a session on a dbt project and whenever a build takes longer than the dataset's lineage should.
---
# dbt workflow

## Selection first

- `dbt ls --select +<model>` lists the upstream closure, `--select <model>+` the downstream
  one; `--resource-type model` keeps sources and tests out; `--output json --output-keys
  name resource_type depends_on config schema original_file_path` prints the graph the
  lineage script reads.
- `dbt run --select +<model>` builds the model and what it needs: on a large project,
  minutes instead of hours; on a small one a full `dbt build` is a fair last check.
  `dbt build --select <model>` also runs
  the model's tests, and a failing upstream test stops the build.
- `dbt compile --select <model>` writes the rendered SQL under target/compiled/ when the
  Jinja is the suspect; `dbt parse` refreshes the manifest after a file change.

## Ad-hoc queries through dbt

```
dbt show --inline "select count(*) as n from {{ ref('model') }}" --output json --limit 1000
```

Runs any SQL against the project's warehouse with refs and sources resolved and prints
`{"show": [...]}` after the log lines. It prints only the first few rows unless `--limit`
is given, so set it. Jinja works
inside, so `{% if target.type == 'snowflake' %}` branches a dialect. No trailing
semicolon. The audit helpers of the project's packages (`compare_relations`, `equality`,
`equal_rowcount`) give the keyed comparison of two relations when installed.

## The bundled scripts

The invariants, lineage and fingerprint scripts talk to the warehouse through `dbt show`
and `dbt ls`, so they need dbt on PATH, the project's profile (`--profiles-dir` when it is
elsewhere) and, when the profile has several targets, `--target`. They resolve a model by
its unique name and read column metadata from the information schema of the connection's
current database; a model present in two packages or versions is refused rather than
guessed, and a relation whose columns the connection cannot list is recorded as absent
only when the adapter says no such relation exists, as unmeasured otherwise; the proof
then records the fully qualified relation and the query used instead. `evidence.py`
builds with `dbt build` and reads dbt's own record of that run where dbt writes it by
default, `<project>/target/run_results.json`. What counts as changed code follows the
repository's own ignore rules: dbt's output (`target/`, `dbt_packages/`, `logs/`) has to
be ignored there, or it reads as changed files no ticket owns. Row keys come back
lower-cased whatever the engine's case folding. They were run on Snowflake and DuckDB. Date arithmetic and the float type go
through dbt's cross-database macros; the two forms that have none (the test for a
non-finite float, the one-statement JSON fingerprint) are refused or replaced by plain
aggregates on another adapter rather than guessed. A column name a spec writes goes into
the SQL as written, so a quoted mixed-case column is quoted in the spec; a name the
scripts read from the engine's own metadata (a fingerprint's columns, the `auto` finite
check) is the stored name and is quoted the adapter's way.

## The canon in dbt terms

These are the conventions of the projects this skill was worked on. A project organised
another way keeps its own architecture, and a ticket is no reason to move it.

Staging one-to-one with sources, the only place `source()` appears, views, no joins or
aggregations; intermediate for re-graining and isolating joins; marts entity-grained and
wide. A `unique` and `not_null` test on every model's primary key; assumptions about
sources tested on the sources; unit tests for logic (date math, window functions, long
`case` chains, previously reported bugs), run in development and CI; data tests after the
build with failures stored so the failing rows can be read. The materialization ladder
(view, table, incremental, in that order and only under pressure); ephemeral models
cannot be queried for troubleshooting; incremental models with a lookback for late data,
a non-null unique key, and a full refresh of the model and its downstream whenever the
incremental logic changes; snapshots with a unique key and a hard-delete policy, in their
own schema, never rebuilt. Contracts define shape (every column's name and type, enforced
before the build); versions exist only for breaking changes. Debugging starts from the
compiled SQL under target/ run in the warehouse, one model at a time, materializing a
chain as tables when the engine hides which step fails.

## Several engines

```sql
{% if target.type == 'snowflake' %}
    listagg(method, ',') within group (order by method)
{% else %}
    string_agg(method, ',' order by method)
{% endif %}
```

Prefer the adapter-dispatched cross-database
macros in the `dbt` namespace (`dbt.dateadd`, `dbt.datediff`, `dbt.date_trunc`,
`dbt.safe_cast`, `dbt.split_part`), and a macro with `adapter.dispatch` when the same
expression is needed in several models; verify the installed adapter's behaviour, since a
macro named `safe_cast` does not guarantee error-to-null on every engine. Compile the
second project when it exists, run it when its engine is reachable, and record the
evidence class per engine in the proof (engine-portability).

## CI, and its blind spots

Slim CI selects `state:modified+` against a production manifest with `--defer --state` in
an isolated schema. A change only to a `var` or `env_var` value may escape
`state:modified`, so select the affected resources explicitly when runtime configuration
changes; seeds of 1 MiB or more are compared by path only; tests with several parents may
run across environments; `--state` and `--target-path` on the same path is non-idempotent.
Build-then-test ships corrupted data before the test runs, so the audit-then-swap pattern
(a cloned schema built and tested under a separate target, then swapped) is the reliable
form. A reviewer, a CI job or an evaluator rebuilds the model from the committed files;
a model materialized by hand, a table left in the warehouse, or a test added to the
project proves nothing about the SQL file, and a changed materialization is a structural
change the rebuild sees.

## Runtime traps

- `dbt deps` once per session when `dbt_packages/` is missing; pinned versions stay pinned.
- The profile comes from the task (`DBT_PROFILES_DIR` or profiles.yml in the project); the
  scripts take `--profiles-dir` when it is elsewhere.
- A target that reads production (the reference target of prepared/environment.json)
  is for `dbt show` and the evidence scripts, never for `run`, `build` or `snapshot`.
  Before trusting it, list one model through it (`dbt ls --select <model> --output json
  --output-keys database schema --target <it>`) and see that the schema is the one the
  dataset lives in: a project's own `generate_schema_name` often keys on the target's
  name and sends a target called `prod` to other schemas than a development target.
- Every dbt start on a large project costs seconds before the first query. Put several
  questions in one `dbt show --inline` (a union of scalar subqueries) rather than one
  call each; the evidence scripts already do.
- A view's upstream is re-read at every query: time one full-lineage query before looping.
- Metadata freshness treats views as always fresh; the partial-parse cache goes stale after
  a rename (`dbt parse` fixes it); deprecation warnings print after the JSON block.
- `--full-refresh` on a production target and anything that drops what others read need
  the approval the task statement names; a failing test is a question about the code, the
  data or the environment, answered with evidence before it is changed.
