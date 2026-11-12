# Setup

Pin the assumptions, the measurement context and the state of the data before anything
changes: the cheap steps first, each written down as it finishes.
1. Read the task statement, its tickets and the repository's own documentation (README,
   runbooks, CI files) for how this team builds, tests and ships, as far as it says. Write
   the domain statement: the stack (engine, transformation tool, orchestrator), the
   billed quantity (warehouse-cost), the project directory with its profile and target,
   the project's test command, the task's acceptance check with what one run costs, one
   ticket or a backlog, and the kind of change each ticket asks for (validation-report).
2. Prove the connection with one query through the project's interface (dbt-workflow on
   a dbt project) and pin what every later measurement must share: the code revision,
   the target, the input snapshot or source versions when they can be frozen, an output
   namespace per attempt when the environment offers one, the session time zone. What
   cannot be frozen or isolated is written down as a limit on every comparison.
3. On a dbt project, write prepared/environment.json from those sources, each key only
   when they give it (most repositories give some): a key left out is a measurement no
   attempt can take, and the label says so. `build_target` and `reference_target` (the
   dbt targets a session builds on and reads production through); `reset_command` (puts
   the build namespace back to the reference state); `acceptance_command` when the
   repository has a check a session may run (`{ticket}` in it for a ticket's key), with
   `acceptance_pass_pattern` and `acceptance_fail_pattern` when its output, not its exit
   status, carries the verdict, and `acceptance_at_a_time` only as validation-report
   says; `release_command` and `rehearsal_target` when the repository documents a
   release. Then `evidence.py environment --ticket <one ticket>` must exit 0. On another
   stack, record the same facts in the domain statement: the scripts read none of them.
4. A backlog: write the inventory (ticket-backlog) to prepared/backlog/inventory.json
   from the tickets and the project's graph, and prepared/backlog/summary.json
   (`backend`, `tickets`, `context`); return prepared/backlog and stop. Each ticket's
   baseline is taken when the ticket is worked, and nothing is built in setup.
5. One ticket: declare the grain of its dataset (grain-and-modelling); for a fix,
   reproduce the reported number exactly (same filter, period, time zone, grain), or record
   that it cannot be reproduced, which is the first finding; for every kind, resolve the
   producers and consumers (lineage-bisection, `--counts`). Write the baseline invariants spec,
   prepared/baseline/spec_<model>.json, each check justified by a contract or a source
   definition and never by a number copied from the ticket, and
   prepared/baseline/summary.json (`backend`, `model`, `grain`, `change_kind`,
   `downstream_models`, `spec`, `context`); return prepared/baseline. An attempt copies
   that spec to results/change/, adds its own checks, and takes the baseline before it
   changes a file (`evidence.py baseline --ticket change`). Nothing is fixed in setup.
