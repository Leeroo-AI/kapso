# RelBench Integration

Rendered guide: [docs.leeroo.com/docs/benchmarks/relbench](https://docs.leeroo.com/docs/benchmarks/relbench) · Kapso by Leeroo: [github.com/Leeroo-AI/kapso](https://github.com/Leeroo-AI/kapso) · `pip install leeroo-kapso`

Every company's most valuable data has the same shape: customers, orders, events, and records
spread across linked tables. [RelBench](https://relbench.stanford.edu) (Stanford STAR,
[v2 paper](https://arxiv.org/abs/2602.12606)) turns that shape into a benchmark: 11 real
relational databases (e-commerce reviews, classified ads, fashion retail, online Q&A, clinical
trials, and in v2 also SAP sales orders and ICU records) and 66 prediction tasks: who churns,
what sells, which trial succeeds, what to recommend next. The official leaderboard scores 31
of those tasks, all from the seven v1 databases.

Kapso attacks each task the way a research team would: a campaign of
**ideation → implementation → judged feedback**, every iteration scored on the official
validation metric, every lesson compounding into the next round. Ideas are grounded in two
knowledge sources: a **bank of lessons** of measured lessons from its own past campaigns,
and **Leeroopedia**, Leeroo's curated ML knowledge base.

## Results

The [official RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/)
has three boards, classification, regression and recommendations, and Kapso holds first place
on all three. The two strongest other entries are Stanford STAR's own
[RT-J](https://github.com/stanford-star/relational-transformer) (fine-tuned) and
[KumoRFM-v2](https://docs.nvidia.com/sdgm/rfm/overview)
([paper](https://arxiv.org/abs/2604.12596); on the board as KumoRelational), the strongest
foundation model for relational data. Neither has a recommendation entry, so there the bar is
the best other entry: ID-GNN, a RelBench baseline, at 9.3 mean MAP over the same ten tasks.
Board read 2026-10-11:

![RelBench, three panels: classification, Kapso 81.2 AUROC against KumoRFM-v2's 79.3 and RT-J's 79.0; regression, Kapso 0.2476 NMAE against RT-J's 0.2711 and KumoRFM-v2's 0.2847, lower being better; recommendations, Kapso 18.4 MAP against the official leaderboard's best other entry, ID-GNN, at 9.3. The first two panels are drawn from a truncated axis, the third from zero](../../docs/images/relbench.png)

| Board | Metric | Best other entry | Kapso | Margin |
|---|---|---|---|---|
| Classification · 12 tasks | AUROC, higher is better | 79.3 (KumoRFM-v2; RT-J 79.0) | **81.2** | +2.4% |
| Regression · 9 tasks | NMAE, lower is better | 0.2711 (RT-J; KumoRFM-v2 0.2847) | **0.2476** | 8.7% less error |
| Recommendations · 10 tasks | MAP, higher is better | 9.3 (ID-GNN) | **18.4** | +97% |

The first two panels of the chart are drawn from a truncated axis, its origin at 76 AUROC and 0.20 NMAE, because
both metrics live in a narrow band far from zero, and a zero-based bar renders a two-point
AUROC margin as no margin at all. The recommendations panel starts at zero: a near-doubling
needs no help, and a floor just under 9.3 would draw it as an eightfold gap. The table is the
same result without the axis.

All Kapso scores come from the official RelBench evaluator on the official task sets; the
comparison numbers are the methods' own leaderboard entries. The same solutions also beat a frontier
coding agent (Claude Code, Fable-5) under an identical budget: see
[`claude_code_baseline/`](claude_code_baseline/). Published entries and the standing bars per
task are on the [official RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard/).

## How it stays honest

1. **Sanitized copy**: the runner builds a per-task database copy with the
   predictive-modelling harness's sanitizer (its `sanitized-copy` skill), everything
   test-derivable physically removed; the agent-visible test table carries only
   (entity, seed-time) rows, and the copy is reused across campaigns under its marker.
2. **Search**: `build()` with the bundled `predictive-modelling` harness (or the one
   `--harness` names) iterates ideation → implementation → verification → judged feedback
   against official *validation* scores through the provided immutable grader, whose
   `task.json` names the task, the copy and the caps.
3. **Selection**: the archive's registered finals are rescored by the head evaluator; the
   winner is scored **once** on test by the runner, the harness's leakage scan runs over
   its code, and `final_report.json` records both beside what the build delivered.

Temporal-regime details, including the rolling per-tick harness for windowed tasks, live in
[`EVALUATION_PROTOCOL.md`](EVALUATION_PROTOCOL.md).

## Quickstart

```bash
# from the repository root
pip install -e .
pip install -r benchmarks/relbench/requirements.txt
```

Run one task:

```bash
expert-relbench -s rel-f1 -t driver-position --time-budget-hours 4
```

The default harness is the bundled `predictive-modelling`; `--harness REF` swaps it,
`--experience REFS` serves banks or an index, and `--workspace DIR --resume` continues a
campaign. The task's archive root (`tmp/relbench/<dataset>--<task>/`, the `benchmark.work_root`
key of `config.yaml`) holds the sanitized copy, the staged evaluation, the shared cache, the
runs and `final_report.json`.

Multi-task campaign (task queue, hardware gating, budget threading):

```bash
PYTHONPATH=src:. python -m benchmarks.relbench.campaign \
    --hours-per-task 10 --hardware gpu
```

## Layout

| path | role |
|---|---|
| `runner.py` / `campaign.py` | single-task CLI on `build()` / multi-task campaign driver |
| `scoring.py` | the runner's own scoring: the sanitized copy through the harness's script, the staged evaluation, the final report over the archive |
| `context.py` / `task_specs.py` | the Brief from the task's facts / per-task metadata |
| `config.yaml` | the override file over the packaged config (`kapso doctor --show-config --config benchmarks/relbench/config.yaml`); the runner writes a runtime copy with the task's shared cache; the `benchmark:` block holds the runner's own keys |
| `data/` | agent-facing starter kit, provided evaluator, seed baseline |
| `claude_code_baseline/` | the frontier-coding-agent baseline: prompt, harness, results |
