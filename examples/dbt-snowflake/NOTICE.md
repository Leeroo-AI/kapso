# Notice

Parts of this example derive from [data-eng-bench](https://github.com/Snowflake-Labs/data-eng-bench),
Copyright 2026 Snowflake Inc., used under the Apache License, Version 2.0; a copy of that licence is in
`LICENSE-APACHE-2.0` beside this file. The benchmark describes its tasks, dbt models, verifiers and `retail`
dataset as original works authored by Snowflake; the dataset is a synthetic retail data warehouse and contains
no real customer data.

What derives from it here, and what was changed:

- `repo-template/tickets/`: the instructions of seventeen of its tasks, rewritten as tickets for one warehouse
  on Snowflake. `platform/service/tickets.json` names each ticket's task.
- `platform/service/recipes/`: the build steps the benchmark's scaffolding runs for those tasks (the connection
  profile, the schema steps, the dbt commands), with the submitted project in place of any reference file.
- `assemble.py` places the benchmark's Snowflake dbt project and the starting files of seven of its tasks into
  the repository it writes. They are read from your own checkout and are not part of this example.

The benchmark's dbt project, its data, its acceptance tests and its reference solutions are not included.
Everything else in this directory is part of Kapso, under Kapso's own licence. Brightmere is a fictional
retailer of ours.
