# Cross-task learning harvester (IOAI local tasks)

A mechanism to run kapso on **related** IOAI tasks and distill transferable
lessons into the **Night Watch** (audio class-incremental) shared-cache seed.

## Pipeline
```
1. RUN        python -m benchmarks.ioai_tasks.runner --root <run> --hours 2 \
                 --node-expansion 2 [--shared-cache-dir <seed>]
              (LocalTaskHandler + generic runner; scores submission/solution.py
               on a PRIVATE held-out split via the task's evaluate.py)
2. EXTRACT    one learning-extraction agent per finished run, briefed by
              harvest/extract_prompt.md → a structured learning JSON
              (what worked / failed+why / validation lessons /
               transfer_to_night_watch tagged NEW|CONFIRMS / novelty)
3. AGGREGATE  one agent briefed by harvest/aggregate_prompt.md merges the
              per-task JSONs + the current Night Watch LEARNINGS.md → an
              updated seed (keeps MISSION, adds only NEW bullets, notes
              confirmations, ends with a HARVEST VERDICT)
4. INSTALL    drop the updated LEARNINGS.md into the Night Watch t1_seed_cache;
              (optional) re-run Night Watch to measure if it moved 0.86382
```

## Adding a task
1. Write `data/<task>_statement.md` (statement + solution.py contract) and
   `data/<task>_evaluate.py` (prints `<metric_name>: <float>`; fail loud).
2. Write `data/prepare_<task>.py`: build `<root>/task/dataset/` (agent-visible
   train + dev + statement + evaluate.py), keep the test split under
   `<root>/private/`, and write `<root>/task/task_meta.json`
   ({task_name, metric_name, self_check_command, final_eval_command}).
3. The generic handler/runner/config need no changes.

## Feasibility notes (2024/2025 tasks)
- **Help_BOBAI (2024)** — fully self-contained in the repo (frozen
  `base_classifier.pth`, labeled train + test); CPU-only; implemented here.
- **Weather (2025)** — data lives on the Bohrium contest platform
  (`/bohr/train-ma50/v2/`), not in the repo; needs that data to harness.
- **Chicken_Counting (2025)** — images + frozen backbone; harness-able if the
  dataset is obtained.
