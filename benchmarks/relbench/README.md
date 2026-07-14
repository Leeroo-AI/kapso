# RelBench Integration

This module integrates Kapso with [RelBench](https://relbench.stanford.edu) (Stanford/Kumo's
relational deep learning benchmark, [v2 paper](https://arxiv.org/abs/2602.12606)): **11
databases, 66 tasks** across four families — entity classification (AUROC), entity
regression (MAE/NMAE, some R²), recommendation (MAP@K), and autocomplete (new in v2).

A model receives a temporal relational database plus seed rows (entity id, seed time) and
must predict future outcomes, missing attributes, or ranked future links — with every
feature and sampled neighborhood censored at each row's seed time.

## Why RelBench is winnable (state of play, July 2026)

- The [official leaderboard](https://huggingface.co/spaces/relbench/leaderboard) covers
  only 31 of 66 tasks and is curated from papers (a redesigned board with submissions is
  announced). **All 23 autocomplete tasks and most new v2 tasks have no leaderboard** —
  the de facto SOTA is the v2 paper's own GraphSAGE baseline, several of which are at or
  below random (negative R², ~50 AUROC).
- The recommendation board is stale: published ContextGNN/KumoRFM numbers beat it 2-3×
  on several tasks but were never added.
- The closest competitor archetype, RelAgent (GPT-5.2 SQL+GBDT agent,
  [arXiv:2605.07840](https://arxiv.org/abs/2605.07840)), holds board bests on 2 regression
  tasks using only 5 rollouts and tree models — no GNNs, no ensembling.

Kapso's edge: many more search iterations, GNN+GBDT+heuristic diversity in one loop,
official-metric scoring every run, and a per-task SOTA bar injected into ideation
(`data/sota.json`).

## How the integration works

```
benchmarks/relbench/
├── runner.py        # CLI entry: one task per run
├── handler.py       # RelBenchHandler: executes candidates, scores VAL, hides TEST
├── context.py       # problem-context builder (schema, task SQL, contract, playbooks)
├── task_specs.py    # task families, primary metrics (incl. R²/NMAE routing), timeouts
├── sandbox.py       # sanitized read-only data cache builder (leak-proofing)
├── config.yaml      # search modes (RELBENCH_CONFIGS / HEAVY_EXPERIMENTATION / MINIMAL)
├── BASELINES.md     # verified per-task numbers + protocols for RelAgent & KumoRFM-ft
├── RESULTS.md       # THE reference: agent results vs baselines, hardware, status (auto-generated)
├── scorecard.py     # category tables (clf/reg/rec) + RESULTS.md generator (--reference)
└── data/
    ├── sota.json    # per-task best published numbers (the bar shown to ideation)
    ├── baselines.json # machine-readable verified baseline numbers (see BASELINES.md)
    ├── leaderboard_snapshot.json # full official board (all methods x tasks, 2026-07-14)
    └── starter_kit/ # -> workspace kapso_datasets/: contract, self-check, helpers,
                     #    vendored official RelBench examples
```

Per experiment, the tree search has a coding agent implement a solution on a git branch,
then the handler runs it (`python main.py --debug`, then full) and scores it:

- The candidate's process sees **only a sanitized, read-only RelBench cache**
  (`RELBENCH_CACHE_DIR`): the database is physically truncated at the test cutoff
  (forecasting/recommendation) or has post-cutoff target values blanked (autocomplete);
  the test task table carries only entity ids + timestamps. Test labels cannot be read,
  even deliberately — this also fixes two upstream relbench leaks (the masked
  recommendation test table retains ground-truth destination lists; the on-disk DB cache
  retains post-test rows).
- The candidate writes `val_predictions.npy` + `test_predictions.npy` to
  `$KAPSO_RUN_DATA_DIR`. The handler validates shapes/dtypes, computes **official
  validation metrics** (search score = the task's primary metric), and computes test
  metrics **privately** (never shown to the agent or the search).
- `final_evaluate` selects the best-by-validation run, replays a static anti-leakage
  audit over its code, and writes a leaderboard-ready `final_report.json` with val + test
  metrics — i.e. tune-on-val / report-test-once, exactly the RelBench protocol.
- `$KAPSO_SHARED_CACHE_DIR` persists across experiments for text embeddings,
  materialized graphs, and per-model predictions (cheap late-stage ensembling).

## Prerequisites

1. **Core Kapso** (repo root): `pip install -r requirements.txt`
2. **Evaluation layer**: `pip install relbench duckdb pooch pyarrow scikit-learn`
3. **Modeling stack** (GPU machine; generated code shares the interpreter):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   pip install "relbench[full]" sentence-transformers lightgbm xgboost catboost tqdm
   ```
4. **API keys** in `.env`: `OPENAI_API_KEY` (aider + ideation), optionally
   `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` for other coding agents.

Datasets download automatically into `~/.cache/relbench` on first use (rel-amazon and
rel-hm are multi-GB). `rel-mimic` requires credentialed PhysioNet + BigQuery access.

## Usage

```bash
# List all native tasks
PYTHONPATH=. python -m benchmarks.relbench.runner --list

# Smoke test on the smallest database
PYTHONPATH=. python -m benchmarks.relbench.runner -s rel-f1 -t driver-position -i 3 -m MINIMAL

# Full run on one task
PYTHONPATH=. python -m benchmarks.relbench.runner -s rel-hm -t user-item-purchase -i 25

# Heavy mode with Claude Code as the coding agent
PYTHONPATH=. python -m benchmarks.relbench.runner \
    -s rel-trial -t site-sponsor-run -i 30 -m HEAVY_EXPERIMENTATION -d claude_code

# Stop early once a validation bar is reached
PYTHONPATH=. python -m benchmarks.relbench.runner -s rel-salt -t sales-payterms --target-val 0.60
```

## CLI options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --dataset` / `-t, --task` | RelBench dataset/task | required |
| `-i, --iterations` | Max search iterations | 20 |
| `-m, --mode` | Config mode | `RELBENCH_CONFIGS` |
| `-d, --coding-agent` | aider, gemini, claude_code, openhands | from config |
| `--target-val` | Early-stop validation bar | none |
| `--workspace` / `--resume` | Reuse/resume a workspace | fresh |
| `--rebuild-cache` | Rebuild the sanitized cache | cached |
| `--knowledge-file` | Extra context markdown | none |
| `--no-kg` | Disable knowledge graph | enabled |

## Campaign plan to rank first

Priority order (impact ÷ compute), given the July 2026 landscape:

1. **Autocomplete sweep (23 tasks, no leaderboard)** — many near-random baselines
   (`rel-salt sales-group` 15.8 acc, `sales-payterms` 37.5, `rel-amazon review-rating`
   R² < 0, `rel-event event_interest-*` ≈ random). Most are join-lookup/GBDT problems;
   small databases → cheap iterations.
2. **Uncontested v2 tasks** — `rel-ratebeer` (3 rec + 4 entity), `rel-arxiv` (4 tasks):
   only v2 baselines exist.
3. **Recommendation board** — reproduce-and-tune the hybrid pairwise/two-tower recipe +
   candidate-generation+GBDT rankers; the board is stale even against published numbers.
4. **Soft classification/regression cells** — `rel-avito ad-ctr` & `rel-trial
   study-adverse` (agent-held records), `site-success` (widest spread), `user-clicks`,
   `driver-position`, `user-attendance` (zero-inflation trick ≈ 10× NMAE).
5. **Foundation-model strongholds last** (`KumoRFM-ft` 81.1 mean AUROC, `RT-ft` 0.2328
   mean NMAE) — needed only for mean-level #1, not per-task wins.

Leaderboard entry is currently by team curation: publish results (val+test metrics per
task, seeds, code, `final_report.json` audit) and contact the maintainers via the
[RelBench Google Group](https://groups.google.com/forum/#!forum/relbench) /
[GitHub](https://github.com/stanford-star/relbench) ahead of the redesigned submission
flow.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | ideation + aider | required |
| `CUDA_DEVICE` | GPU id for generated code | `0` |
| `RELBENCH_FULL_TIMEOUT` | full-run cap (s) | per-dataset tier (2h/4h/8h) |
| `RELBENCH_DEBUG_TIMEOUT` | debug-run cap (s) | tier (15/20/30 min) |
| `RELBENCH_PRISTINE_CACHE_DIR` | source cache for sanitizer | `~/.cache/relbench` |

## Protocol & integrity guarantees

- Search optimizes official **validation** metrics only; test metrics are computed once
  per run and quarantined in `tmp/relbench/<task>/runs/*/private/` until `final_evaluate`.
- Candidates physically cannot read test labels (sanitized cache is complete for
  train/val work and read-only; `download=True` fails loudly).
- `final_report.json` includes a static audit of the winning code for forbidden access
  patterns — reproduce the winner cleanly before publishing numbers.
