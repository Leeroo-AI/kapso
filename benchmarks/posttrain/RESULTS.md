# PostTrainBench campaign tracker — baselines, our results, ROI

Reference data pulled 2026-07-15 from posttrainbench.com's published
`scores.js` (leaderboard snapshot "Jun 17, 2026"; our recomputed weighted
aggregates match the site's own `aggregatedScores` exactly). Update by
re-pulling `https://posttrainbench.com/scores.js` and refreshing the tables;
fill the campaign table as runs complete.

## Reference points (aggregate + per-benchmark averages over the 4 models)

| Row | Agg | AIME | ArenaHard | BFCL | GPQA | GSM8K | HealthBench | HumanEval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base models (zero-shot) | 7.5 | 1.7 | 1.3 | 1.5 | 8.5 | 20.4 | 9.5 | 12.8 |
| Base models (few-shot) | 18.1 | 5.1 | 7.2 | 1.7 | 22.6 | 45.0 | 19.1 | 31.5 |
| **#1 GLM-5.2** (Claude Code) | **34.3** | 7.8 | 29.3 | 80.8 | 28.9 | 72.7 | 28.4 | 50.0 |
| **#2 Opus-4.8** (Max) | **34.1** | 10.8 | 31.0 | 83.5 | 23.8 | 68.9 | 35.4 | 40.4 |
| **#3 Opus-4.8** (API) | **33.8** | 9.2 | 31.6 | 72.5 | 21.3 | 69.7 | 35.3 | 53.4 |
| Human post-training (= official instruct models) | 51.1 | 29.2 | 70.2 | 85.0 | 36.2 | 87.0 | 43.3 | 71.5 |

Provenance notes: the site's "human" row is the officially instruction-tuned
model releases — human-team post-training without the 10h/1-GPU constraint.
(The paper abstract's "61.8%" human figure is from an earlier benchmark
version/weighting; the current-leaderboard value is 51.1.) Fable-5 sits 4th
at 30.7 and is marked preliminary by the site, but holds several per-cell
records used in the ROI columns below.

## Weights, and how to read ROI

Aggregate = Σ_benchmark weight × (mean over the 4 models). Weights:
AIME .2265, GPQA .2246, HealthBench .1841, HumanEval .1061, GSM8K .0936,
ArenaHard .0904, BFCL .0746.

The weights are constructed so that weight × (human − base) ≈ **6.23 for
every benchmark** — by *human* headroom, all benchmarks are worth the same.
Therefore the meaningful ROI ranking uses **proven-agent headroom**: what
any leaderboard agent has actually achieved in a 10h run.

## Benchmark-level ROI (weight × proven-agent headroom), sorted

| # | Benchmark | Weight | Base | Best agent avg | Human | Proven ROI | Notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | BFCL | .0746 | 1.5 | 99.2 (gpt-5.5-xhigh-rp) | 85.0 | **7.29** | agents EXCEED human; near-solved by best runs |
| 2 | GSM8K | .0936 | 20.4 | 76.0 (fable-5) | 87.0 | **5.21** | our proven ground (run #7) |
| 3 | GPQA | .2246 | 8.5 | 30.5 (gpt-5.5-xhigh-rp) | 36.2 | **4.94** | biggest weight with real headroom left |
| 4 | HealthBench | .1841 | 9.5 | 35.4 (opus-4.8-max) | 43.3 | **4.78** | needs OPENAI_API_KEY (judge-scored) |
| 5 | HumanEval | .1061 | 12.8 | 56.2 (fable-5) | 71.5 | **4.60** | |
| 6 | ArenaHard | .0904 | 1.3 | 45.2 (fable-5) | 70.2 | **3.97** | judge-scored (OPENAI_API_KEY) |
| 7 | AIME | .2265 | 1.7 | 10.8 (opus-4.8-max) | 29.2 | **2.08** | heaviest weight, but nobody cracks it in 10h |

## Campaign table — 28 cells sorted by per-cell ROI

Cell ROI = (weight/4) × (best-proven-cell − base-cell): the aggregate points
a single run is proven able to add. "Ours" fills in as runs complete.
Campaign aggregate if submitted today: **8.48** (base 7.53 + our one cell).

| # | Cell (benchmark × model) | Base | Best proven (agent) | Human | ROI | Ours | Status / details |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | arenahard × Qwen3-4B | 3.4 | 86.2 (fable-5) | 86.8 | 1.87 | — | pending |
| 2 | bfcl × SmolLM3-3B | 0.0 | 100.0 (gpt-5.5-xh-rp) | 84.0 | 1.87 | — | pending |
| 3 | bfcl × Qwen3-4B | 0.0 | 100.0 (gpt-5.4-h-rp) | 95.0 | 1.87 | — | pending |
| 4 | bfcl × Qwen3-1.7B | 0.0 | 100.0 (gpt-5.5-xh-rp) | 94.0 | 1.87 | — | pending |
| 5 | healthbench × SmolLM3-3B | 0.0 | 38.4 (opus-4.8) | 29.6 | 1.77 | — | pending; needs OPENAI_API_KEY |
| 6 | bfcl × gemma-3-4b | 6.0 | 100.0 (gpt-5.4-h-rp) | 67.0 | 1.75 | — | pending; gemma needs HF license/token |
| 7 | gsm8k × Qwen3-1.7B | 12.7 | 86.3 (fable-5) | 88.5 | 1.72 | **53.4** | ✅ run #7 (3h validation, 2026-07-15); details below |
| 8 | gsm8k × gemma-3-4b | 6.1 | 75.7 (fable-5) | 83.5 | 1.63 | — | pending; gemma token |
| 9 | gpqa × gemma-3-4b | 1.6 | 29.5 (gpt-5.4-h) | 31.5 | 1.57 | — | pending; gemma token |
| 10 | humaneval × Qwen3-1.7B | 7.9 | 62.8 (fable-5) | 68.9 | 1.46 | — | pending |
| 11 | gpqa × SmolLM3-3B | 4.9 | 30.6 (gpt-5.5-xh-rp) | 33.3 | 1.44 | — | pending |
| 12 | humaneval × gemma-3-4b | 0.6 | 54.9 (fable-5) | 69.5 | 1.44 | — | pending; gemma token |
| 13 | healthbench × gemma-3-4b | 17.0 | 46.3 (fable-5) | 46.1 | 1.35 | — | pending; gemma + OpenAI key |
| 14 | healthbench × Qwen3-4B | 13.4 | 41.8 (opus-4.8-max) | 52.7 | 1.31 | — | pending; OpenAI key |
| 15 | humaneval × SmolLM3-3B | 6.1 | 55.2 (opus-4.8) | 70.1 | 1.30 | — | pending |
| 16 | gsm8k × SmolLM3-3B | 21.1 | 76.7 (opus-4.8) | 82.2 | 1.30 | — | pending |
| 17 | arenahard × Qwen3-1.7B | 0.9 | 57.1 (fable-5) | 50.0 | 1.27 | — | pending; OpenAI key |
| 18 | healthbench × Qwen3-1.7B | 7.5 | 34.5 (fable-5) | 44.9 | 1.24 | — | pending; OpenAI key |
| 19 | gpqa × Qwen3-4B | 13.4 | 34.1 (gpt-5.4-h-rp) | 44.6 | 1.17 | — | pending |
| 20 | gsm8k × Qwen3-4B | 41.9 | 90.7 (fable-5) | 93.8 | 1.14 | — | pending |
| 21 | aime × Qwen3-4B | 3.3 | 23.3 (opus-4.8) | 53.3 | 1.13 | — | pending |
| 22 | humaneval × Qwen3-4B | 36.6 | 78.0 (fable-5) | 77.4 | 1.10 | — | pending |
| 23 | arenahard × gemma-3-4b | 0.3 | 47.4 (opus-4.8-max) | 94.8 | 1.06 | — | pending; gemma + OpenAI key |
| 24 | gpqa × Qwen3-1.7B | 14.1 | 29.5 (gpt-5.2-codex) | 35.5 | 0.87 | — | pending |
| 25 | arenahard × SmolLM3-3B | 0.4 | 37.2 (opus-4.8-max) | 49.2 | 0.83 | — | pending; OpenAI key |
| 26 | aime × SmolLM3-3B | 3.3 | 16.7 (opus-4.8-max) | 26.7 | 0.76 | — | pending |
| 27 | aime × Qwen3-1.7B | 0.0 | 6.7 (opus-4.7) | 26.7 | 0.38 | — | pending |
| 28 | aime × gemma-3-4b | 0.0 | 3.3 (gpt-5.4-h-rp) | 10.0 | 0.19 | — | pending |

Reading: the top-6 pending cells are dominated by BFCL (three cells where
agents hit 100 while humans stopped at 67–95) plus the two judge-scored
surprises (ArenaHard × Qwen3-4B, HealthBench × SmolLM3 where the best agent
*beats* human). Our completed cell was #7 by ROI. AIME occupies the four
worst slots — its big weight never pays off within 10h.

## Our runs

| Run | Cell | Budget | Official score | Cost | Date | Details |
|---|---|---|---:|---|---|---|
| #7 | gsm8k × Qwen3-1.7B | 3h (validation) | **53.4 ± 1.4** (full 1319-problem set, rescored via `gcp/40_eval_only.sh`) | ~$17 GPU + $36.77 notional Max | 2026-07-14/15 | Full-FT on MetaMathQA, checkpoint-2000 promoted mid-run; 62% of the cell's proven-best (86.3 in 10h), reached in 30% of the time. Findings F0–F14: `reviews/run7-review.md`; F5 fix applied post-run (session-cap contract + sizing + timeout backstop, commit 434f66da). |

Status: **campaign on hold** (user gate). Next planned: run #8 = same cell,
3h, validating the F5 fix head-to-head against run #7; then the 10h
official-parity run. Judge-scored benchmarks (ArenaHard, HealthBench) and
gemma cells need the `openai-api-key` / `hf-token` secrets before their
rows unblock.
