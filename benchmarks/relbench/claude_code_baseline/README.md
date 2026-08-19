# Claude Code baseline

A raw-agent baseline for the RelBench campaign: one headless **Claude Code**
session (model `claude-fable-5`, `--effort xhigh`) per task, a fixed wall-clock
budget on a **1×A100** GCP box, the same masked-test sandbox and output
contract Kapso's agents get, scored the same way afterwards. The only
variable between this and the Kapso row in `RESULTS.md` is the system.

## What stays identical to Kapso

- **Data**: the sanitized cache from `benchmarks.relbench.sandbox` — train/val
  labeled, test reduced to entity/time seed rows, read-only.
- **Contract**: the starter kit (`data/starter_kit` → `kapso_datasets/`):
  `load_task()`, `save_predictions()`, `CONTRACT.md` shapes and dtypes.
- **Scoring**: one-way, after the session ends, with relbench's own
  `task.evaluate` against the canonical cache (`score_baseline.py`).

## What differs

- No Kapso machinery: no task-context document, no tree search, no ideation
  ensemble, no evaluator maintainer, no knowledge base — just `PROMPT.md`
  (generic, templated with dataset/task/budget/hardware) piped to
  `claude -p`.
- 1×A100 instead of 4×A100; the time budget is the hard `timeout`, so the
  session is killed at the deadline and whatever is on disk is scored.

## Running

```
bash run_baseline.sh box  relbench-cc-1 asia-southeast1-c          # 1xA100 from a lane snapshot
bash run_baseline.sh run  relbench-cc-1 asia-southeast1-c rel-trial/study-outcome 4
bash run_baseline.sh stop relbench-cc-1 asia-southeast1-c
```

`run` is blocking and sequential by design: it builds the sanitized cache
(once per task), stages a fresh workdir with the starter kit, launches the
session in tmux, waits for the deadline, pulls the run dir to
`tmp/claude_code_baseline/<task>/<stamp>/`, scores it, appends nothing to the
Kapso ledgers, and archives one `.tgz` per run to
`gs://leeroo-kapso-relbench-artifacts/baselines/claude_code/<task>/`.
Auth is the `CLAUDE_CODE_OAUTH_TOKEN3` line of the worktree `.env`.

Results go in `RESULTS.md` next to this file, never in the Kapso tables.
