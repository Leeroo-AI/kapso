Objective    Work the open backlog under tickets/ to its end, so that every one of the seventeen
             ticket checks passes, which the team calls its production tests: fix the models and
             the snapshot that produce wrong numbers, build the new models, and rebuild the broken
             reports as *_fixed models, each at the file, the output table and the rules its
             ticket names, in dbt on the Snowflake warehouse.
Measured by  evaluation/evaluate.py, a protected copy of the team's checks/run_checks.py: it sends
             warehouse/ and marketing/ to the check service, which builds each ticket's models in
             a clone of its own and runs the ticket's acceptance checks; tickets_passing, the
             share of the 17 checks that pass, is the last line. A full run takes about five
             minutes; --fast is a four-ticket spot check whose figure is not a score.
Baseline     0 of 17, and the client exits 3 ("the project does not build, so this is not a
             score"): int_sales__orders_enriched.sql calls parse_timestamp, which nothing defines,
             so the warehouse project does not parse and every ticket's build fails until that
             model, the midnight-sales ticket's, compiles; the ten new-model tickets fail first on
             the missing file.
Target       tickets_passing 1.0000: every ticket check passes
Rules
  - staging models, checks/, tickets/, evaluation/ and both projects' dbt_project.yml are never
    modified
  - a fixed model or snapshot keeps its materialization and every existing output column
  - a ticket's change stays within the files its ticket names, plus a macro or schema file that
    ticket needs, so that each ticket ships as a pull request of its own, stacked on another's
    only when it cannot build without it
  - build with --select, never the whole warehouse project; production is read, never built
  - the public benchmark the backlog derives from is not consulted, online or anywhere else
Budget       minutes=720 experiments=34
Harness      ../harnesses/data-engineering
