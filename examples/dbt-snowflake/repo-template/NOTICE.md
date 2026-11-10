# Notice

The dbt projects under `warehouse/` and `marketing/`, the retail dataset loaded in the warehouse, and the
tickets under `tickets/` derive from **data-eng-bench** by Snowflake Inc. (Copyright 2026 Snowflake Inc.),
used under the Apache License, Version 2.0. The benchmark describes its tasks, dbt models and `retail` dataset
as original works authored by Snowflake; the dataset is a synthetic retail data warehouse and contains no real
customer data.

Changes made here: the Snowflake variant of the benchmark's dbt project is used as `warehouse/`, with the
models several of its tasks start from already in place; a small second project, `marketing/`, holds the
marketing report models; the task instructions were rewritten as tickets for one warehouse on Snowflake; the
tasks' acceptance tests are run by a separate check service and are not part of this repository. Brightmere is
a fictional retailer of ours.

A copy of the Apache License, Version 2.0 is at https://www.apache.org/licenses/LICENSE-2.0.
