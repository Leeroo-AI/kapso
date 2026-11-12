# Bundled harnesses

A harness in this directory loads by name: `kapso build --harness <name>`,
`Kapso().load_harness("<name>")`, the brief's `Harness` line; `kapso harness
list` prints each one under `bundled` with what this machine lacks for it.
Each is a complete harness in the package's own grammar and holds no task.
Two were written from the benchmarks the engine was measured on
(docs/research/harness-ports.md, the ports of
docs/plans/redesign/wp8-harness-ports.md): the benchmark runners under
`benchmarks/` supply the task material and launch a build with the harness.
Two were written from published research and practice in their domains and
revised after independent reviews; each has an example task under `examples/`.

| Harness | Domain | Written from |
|---|---|---|
| `predictive-modelling` | structured prediction: a test set that never trains, cross-validation discipline, leakage as the failure that voids, features before architectures, ensembling last; relational databases with a temporal cutoff as the situation its skills name | RelBench (the measured practices, the temporal protocol, the leakage audit), then MLE-bench |
| `competitive-programming` | a scored contest under a clock: validity before score, an insurance submission early, the scorer's measurement over every claim, a submission ledger and a quota spent on purpose; heuristic-optimisation rounds and olympiad-style Kaggle tasks as the modalities its skills name | ALE-Bench (the scorer discipline, the heuristic toolbox), then the IOAI campaigns (the contest economics, the Kaggle flow, the olympiad playbook) |
| `data-engineering` | changes to a data pipeline delivered with measured evidence: the grain declared before the model, a cause traced through the lineage to its mechanism, invariants and a row comparison as the proof, a label computed from the measurements and never chosen; dbt on a SQL warehouse as the stack its scripts and its two delivery targets drive | published practice on data modelling, data quality, pipeline reliability and warehouse cost; the example is `examples/dbt-snowflake/` |
| `agent-engineering` | the harness around a fixed model, so that it does a job a task runner scores: the done-check, the tool layer, the turn cycle and its bounds, the context policy and the guards, each component a dated hypothesis measured by paired runs and a triage of the failed episodes; customer conversations and locally served models as the situations two of its skills name | published research on agent harnesses, each component with its measured effect and its conditions; the example is `examples/support-agent-harness/` |

The gate runs `check` over every directory here at the packaged page budget
and fails on a warning (`tests/test_bundled_harnesses.py`), and scans each
for a benchmark's names and for a number pinned beside a quota, a deadline or
hardware (`tests/test_bundled_abstraction.py`). A number beside a score word
is read in the two harnesses written from benchmarks, where it would be a
benchmark's result, and not in the two written from research, whose skills
cite their studies' results as evidence. The worked example for writing one
lives under `examples/harnesses/trading/`.
