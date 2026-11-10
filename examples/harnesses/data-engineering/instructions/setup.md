# Setup

Pin the assumptions, the measurement context and the state of the data before anything
changes: the cheap steps first, each written down as it finishes.
1. Read the task statement, its tickets and the repository's own documentation (README,
   runbooks): how this team builds, checks, resets and releases is written there. Write
   the domain statement: the stack (engine, transformation tool, orchestrator), the
   billed quantity (warehouse-cost), the project directory with its profile and target,
   the project's test command, the task's acceptance check with what one run costs, one
   ticket or a backlog, and the kind of change each ticket asks for (validation-report).
2. Prove the connection with one query through the project's interface (dbt-workflow on
   a dbt project) and pin what every later measurement must share: the code revision,
   the target, the input snapshot or source versions when they can be frozen, an output
   namespace per attempt when the environment offers one, the session time zone. What
   cannot be frozen or isolated is written down as a limit on every comparison.
3. Write prepared/environment.json from those sources, a key only when they give it: a
   key left out is a measurement no attempt can take. `build_target` and
   `reference_target` (where a session builds; what reads production); `reset_command`
   (puts the build namespace back to the reference state); `acceptance_command` with
   `{ticket}` for the ticket's key, `acceptance_pass_pattern` and
   `acceptance_fail_pattern` (regular expressions with `{ticket}` matching its output
   for a pass and for a fail); `acceptance_at_a_time` (how many may run at once) only
   when a check builds its own state away from this checkout, reading neither the build
   namespace nor this tree's dbt artifacts; `release_command` and `rehearsal_target` when
   the platform releases merged changes. Then `evidence.py environment --ticket <one
   ticket>` must exit 0.
4. A backlog: write the inventory (ticket-backlog) to prepared/backlog/inventory.json
   from the tickets and the project's graph, and prepared/backlog/summary.json
   (`backend`, `tickets`, `context`); return prepared/backlog and stop. Each ticket's
   baseline is taken when the ticket is worked, and nothing is built in setup.
5. One ticket: declare the grain of the named dataset (grain-and-modelling); reproduce
   the ticket's number exactly (same filter, period, time zone, grain), or record that it
   cannot be reproduced, which is the first finding; resolve its producers and consumers
   (lineage-bisection, `--counts`). Write the baseline invariants spec,
   prepared/baseline/spec_<model>.json, each check justified by a contract or a source
   definition and never by a number copied from the ticket, and
   prepared/baseline/summary.json (`backend`, `model`, `grain`, `change_kind`,
   `downstream_models`, `spec`, `context`); return prepared/baseline. An attempt copies
   that spec to results/change/, adds its own checks, and takes the baseline before it
   changes a file (`evidence.py baseline --ticket change`). Nothing is fixed in setup.
