# Claude Code Baseline

How far does a frontier coding agent get on RelBench without any scaffolding? This baseline
runs Claude Code (Fable-5) on the same sanitized task data Kapso sees, with
one short generic prompt ([`PROMPT.md`](PROMPT.md)), the same hardware, and the same 4-hour
budget per task. Predictions are scored one way by the official RelBench evaluator
([`score_baseline.py`](score_baseline.py)); the agent never sees test labels.

![RelBench against a frontier coding agent, three panels: outcome prediction, KumoRFM-v2 70.7 AUROC, Claude Code 73.2, Kapso 76.9; forecasting, 10.4% error, 8.0%, 7.6%, lower being better; recommendations, best reported 16.1 MAP, Claude Code 12.5, Kapso 18.8](../../../docs/images/relbench-vs-claude-code.png)

Six headline tasks, two per family. Kapso wins every settled cell, and in recommendations the
raw agent stays below the best reported bar while Kapso clears it.

| Family | Metric | Bar to beat | Claude Code | Kapso |
|---|---|---|---|---|
| Outcome prediction | AUROC, higher is better | 70.7 (KumoRFM-v2) | 73.2 (+3.6%) | **76.9** (+8.8%) |
| Forecasting | error, lower is better | 10.4% (KumoRFM-v2) | 8.0% (−22%) | **7.6%** (−26%) |
| Recommendations | MAP, higher is better | 16.1 (best reported) | 12.5 (−22%) | **18.8** (+16%) |

The chart's panels are drawn from truncated axes — 68 AUROC, 12% error, 10 MAP, each printed
under its own panel title — so that margins of a point or two are legible; the table is the
same result without the axis. These six tasks are a different, smaller set from the
[headline RelBench figure](../README.md#results), so the two are not comparable.

Per-task scores, the drop protocol, and run archives: [`RESULTS.md`](RESULTS.md). The harness
(`run_baseline.sh`) provisions a box, stages the sanitized cache, runs the agent under a hard
timeout, and archives every session.
