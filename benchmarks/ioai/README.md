# IOAI × Kapso

## 0. First live probe: IOAI 2026 Home Task 3 (built)

Before the 2025 contest integration below, `runner.py`/`handler.py`/
`config.yaml` implement a single live task as a pipe-cleaner: **IOAI 2026
Home Task 3 "Animal Deduction"** ([notebook](https://github.com/IOAI-official/IOAI-2026/blob/main/Home%20Task/Home-Task-3.ipynb))
— 20-questions against a local Qwen2.5-3B oracle (1,471 animals, 558
questions, 15-query budget, `score = solved − 0.02·queries`). Being a 2026
task it is post-cutoff for current models: no contamination caveat. The agent
iterates on `dev.csv` (150 rows); `test1.csv` (500 rows) is held out and
scored by the runner with a pristine harness + static source audit.

```bash
python -m benchmarks.ioai.data.prepare_animal_deduction --root ~/ioai_run
python -m benchmarks.ioai.runner --root ~/ioai_run --hours 2
python -m benchmarks.ioai.runner --root ~/ioai_run --final-eval-only  # rescore
```

Needs one ≥16 GB GPU (the oracle is a 3B model; contest reference is a free
Colab T4), `CLAUDE_CODE_OAUTH_TOKEN` + `OPENAI_API_KEY` in the env.

---

# IOAI 2025 × Kapso (contest integration design)

**Status: design — nothing built yet.** Run Kapso as a "virtual contestant" on the
[IOAI 2025](https://ioai-official.org/) Individual Contest (2nd International Olympiad in
AI, Beijing, Aug 2025): six applied-ML tasks, solved under contest constraints, scored
with the official metrics and normalized onto the official 284-contestant scoreboard.
Same integration shape as `benchmarks/posttrain/` (runner drives
`OrchestratorAgent.solve` under a wall-clock budget, generic sequential search) and
`benchmarks/relbench/` (sanitized data cache, val-driven search, test scored privately).

## 1. The benchmark, precisely

Sources: [task repo](https://github.com/IOAI-official/IOAI-2025) (CC-BY-4.0; statements,
data, official solutions, `metrics.py` per task),
[HF dataset](https://huggingface.co/datasets/IOAI-official/IOAI2025) (~2.2 GB heavy data),
[contest rules v2.2](https://ioai-official.org/wp-content/uploads/2025/07/Contest-Rules-for-IOAI-2025-version-2.2.pdf),
[2025 results](https://ioai-official.org/china-2025/results-2025/).

- **Six tasks**, two on-site days, **6 h per day for 3 tasks**, one contestant = one
  machine with an identical GPU (**≥24 GB** VRAM class, web Jupyter on Bohrium):

| # | Task | Day | Domain | Submission artifact |
|---|------|-----|--------|---------------------|
| 1 | Radar | 1 | signal processing / detection | predictions CSV |
| 2 | Chicken_Counting | 1 | CV counting (provided `base.pth`) | predictions |
| 3 | Concepts | 1 | NLP with LLM assist + **LLM judge API** | `clues_a/b.jsonl` |
| 4 | Restroom | 2 | icon matching | `answer_{a,b}.npy` |
| 5 | Antique | 2 | painting authentication | `submission{A,B}.csv` |
| 6 | Pixel | 2 | pixel-efficiency masks | `submission.jsonl` |

- **Split convention**: every task has `training_set` / `validation_set` (public, "a") /
  `test set` (private, "b"). Official `metrics.py` scores both and writes `score.json`
  (`public_a`, `private_b`). Contestants got provisional feedback per submission;
  final ranking uses the private split.
- **Score normalization** (per task, 100 pts, subtasks sum):
  `Norm = (Sub − Min) / (Max − Min) × 100`, `Min` = provided baseline solution's score,
  `Max = max(0.9 × SC_Solution, best contestant submission)`.
- **2025 scoreboard anchors** (284 contestants, 600 pts max): winner 542.05;
  cutoffs ≈ gold 380.4, silver 269.5, bronze 112.3.
- **Environment rules**: fixed pinned env (repo `requirements.txt`, Python 3.12.7),
  **no extra installs**, no TensorFlow/Keras; internet restricted to
  sklearn/pytorch/HF/numpy/python/pypi docs + a filtered search engine; **pretrained
  models and external data forbidden unless the task statement allows them**; LLM APIs
  forbidden except Concepts (official proxy, $10 credits + 12,500 judge calls, 1 GB
  upload cap; judge API is train/validate-time only, never at inference).

## 2. Integration design

One task = one Kapso campaign (`expert-ioai`), generic sequential search,
`parent_policy: best`, iterating on the **validation** score; the private split never
enters the workspace.

```
benchmarks/ioai/
├── runner.py      # expert-ioai: one task per campaign; wall-clock budget into
│                  # OrchestratorAgent.solve; finally-block guarantees a valid
│                  # best-so-far submission artifact (posttrain pattern)
├── handler.py     # IOAIHandler: contest rules + env facts + per-task submission
│                  # contract; runs candidate, scores VAL via official metrics.py,
│                  # scores TEST privately (relbench pattern)
├── config.yaml    # IOAI mode: generic sequential, parent=best, xhigh,
│                  # web tools disabled (see §3), ~2h implementation timeout
├── data/
│   ├── prepare.py     # GitHub repo + HF dataset -> per-task sanitized agent bundle
│   │                  # + private scoring bundle; leak-scan asserts no test labels,
│   │                  # solutions, or Scoring/ dirs in the agent bundle
│   └── anchors.json   # per-task Min (baseline) + SC_Solution scores, measured once
│                      # by executing the official notebooks (cross-checked against
│                      # released score.json / scoreboard bests)
└── scorecard.py   # raw private_b -> normalized IOAI points -> total -> virtual
                   # rank/medal vs the 2025 scoreboard; RESULTS.md generator
```

**Sanitization (the crux).** The public repo mixes solutions and test ground truth next
to the data: `*_Solution.ipynb`, `Solution/`, `Scoring/` (`ground_truth_*.csv`,
`answer_*.npy`, `testans.json`, `Y_test.npz`), `score.json`. The agent bundle contains
ONLY: statement notebook, `training_set`, `validation_set` (with labels — contestants
had them), test **inputs** label-free, and a val-only scoring entrypoint. Test labels +
official `metrics.py` live outside the workspace; the handler scores test privately per
iteration for our records (never fed back), and the final report is
select-on-val / report-test-once.

**Deliverable guarantee.** Handler mandates an atomically-updated best-so-far submission
under `task/artifacts/` (out of git) + `best_score.log`; runner's finally-block restores
it on any exit, so a hard kill still leaves a scorable submission.

## 3. Fidelity decisions (defaults, revisit on green-light)

- **Time budget: 2 h/task** (= 6 h/day ÷ 3 tasks). Cold-start note: day-1 tasks extended
  the at-home round, so real contestants arrived with weeks of prior work on task 1–3
  variants — 2 h cold is if anything harsher than parity. A generous 6 h/task mode is a
  config knob, reported separately.
- **GPU parity: one 24 GB-class GPU** — GCP `g2-standard-8` (L4 24 GB, ~$0.85/h
  on-demand, ~$0.25 spot). Whole 6-task contest ≈ 15 GPU-h ≈ **$5–15** compute (+
  Anthropic spend for ~12 h of sessions). Dev iteration may use the existing H100 setup
  but parity runs are L4.
- **Agent web access: OFF** (`WebSearch`/`WebFetch` disallowed in adapter sessions).
  Contestants had docs-only internet, and — decisive — the official solutions are
  public on GitHub: any web-enabled run can trivially fetch the answer key. Solution
  code runs offline (`HF_HUB_OFFLINE` in the sandboxed child env; pre-populated HF cache
  holds only task-allowed models), which also blocks pulling the HF dataset's labeled
  test splits.
- **Env parity**: conda env from the official `requirements.txt` (py 3.12.7); handler
  forbids new installs; audit diffs `pip freeze` after the run.
- **Contamination caveat (honest framing)**: solutions and community writeups have been
  public since ~Sep 2025 — inside every current model's training window. Offline mode
  blocks lookup but not memorization; results are therefore an upper bound. Mitigation:
  post-run audit compares final code against the official solution notebooks and flags
  structural plagiarism (posttrain's judge-audit pattern).
- **Concepts is special-cased to a fast-follow**: it needs a live LLM-judge endpoint
  (repo ships `judge_api.py` for OpenRouter) plus proxy-budget emulation ($10 /
  12,500 calls). v1 ships the other 5 tasks; Concepts lands once the judge plumbing has
  its own smoke test. Virtual totals over 5 tasks are still rankable (scoreboard is
  per-task).

## 4. Build plan

1. `data/prepare.py` + leak-scan; sanitized bundles for 5 tasks (Concepts deferred).
2. Anchor measurement on the GPU box: run baseline + official solution notebooks once →
   `data/anchors.json` (validates env, data, metrics plumbing before any agent runs).
3. `runner.py` + `handler.py` + `config.yaml` (crib posttrain runner skeleton, relbench
   handler scoring shape; smoke with a stub task first — no GPU needed).
4. `scorecard.py` + scoreboard snapshot (per-task bests for the `Max_Submission` term).
5. One real 2 h run (Restroom or Antique — smallest data) on L4; review trace; then the
   remaining tasks; then Concepts.
6. Optional extensions: at-home tasks (Chameleon/Weather/Radar-home) as free dev/smoke
   fixtures — they never touch official-task experiment memory; GAITE tasks; a "day
   mode" (one 6 h campaign per 3-task day) if we later want strict-format parity.

Open before build: confirm 2 h/task default, L4 parity hardware, and Concepts deferral.
