# PostTrainBench campaign tracker (v1.1) — baselines, our results, ROI

**Clean-slate restart on the v1.1 leaderboard.** The prior campaign (17
old-judge-clean cells, agg ≈20.97, built on the "Jun 17, 2026" snapshot) is
archived verbatim under `_archive/` (its `RESULTS.md`, `reviews/`,
`knowledge/` seed docs, `arenahard_board.png`, and fetched traces). Those
runs were scored under the OLD judge; their recorded traces would trip the
v1.1 lookup judge, so this campaign starts fresh.

Reference data pulled **2026-08-09** from `https://posttrainbench.com/scores.js`
(the v1.1 re-scored leaderboard). Base and human per-benchmark means are
unchanged from the old snapshot; what v1.1 changed is the AGENT rows —
re-scored after the audit removed contaminated / hosted-API-teacher /
model-substitution / PTB-lookup runs. Update by re-pulling `scores.js`.

## Reference points (aggregate + per-benchmark averages over the 4 models)

Aggregate = Σ_benchmark weight × (mean over the 4 base models). Base/human
aggregates are computed (the site stores only per-agent aggregates); agent
aggregates are the site's stored `aggregatedScores` values.

| Row | Agg | AIME | ArenaHard | BFCL | GPQA | GSM8K | HealthBench | HumanEval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base models (zero-shot) | 7.54 | 1.65 | 1.27 | 1.5 | 8.49 | 20.56 | 9.49 | 12.81 |
| **#1 fable-5** | **41.79** | 13.33 | 61.46 | 73.0 | 28.35 | 83.53 | 39.62 | 58.40 |
| **#2 gpt-5.6-sol** | **36.23** | 7.5 | 26.40 | 94.38 | 30.39 | 69.48 | 27.39 | 63.95 |
| **#3 opus-5** | **35.04** | 9.17 | 53.45 | 1.5† | 31.36 | 79.84 | 38.65 | 59.76 |
| opus-4.8 | 33.84 | 9.17 | 45.81 | 58.0 | 20.68 | 69.69 | 32.37 | 57.83 |
| opus-4.8-max | 32.90 | — | — | — | — | — | — | — |
| kimi-k3 | 31.96 | — | — | — | — | — | — | — |
| glm-5.2 | 31.70 | 7.53 | 27.03 | 32.92 | 29.65 | 73.81 | 25.55 | 52.43 |
| opus-4.7 | 28.56 | — | — | — | — | — | — | — |
| gpt-5.5-xhigh | 27.23 | — | — | — | — | — | — | — |
| grok-4.5-high | 23.45 | — | — | — | — | — | — | — |
| gemini-3.1-pro | 21.99 | — | — | — | — | — | — | — |
| gpt-5.4-high | 19.00 | — | — | — | — | — | — | — |
| **Human (official instruct)** | **51.19** | 29.17 | 70.23 | 85.0 | 36.47 | 87.0 | 43.33 | 71.49 |

† opus-5 shows BFCL 1.5 = base (no BFCL run on record for it), which drags
its aggregate. Per-benchmark means for rows marked "—" not yet pulled;
pull them per-benchmark when that benchmark's campaign starts.

Notes: there is NO few-shot base row in v1.1 (only zero-shot). The v1.1
human aggregate computes to 51.19 from the (unchanged) per-benchmark human
means — the abstract's ~61.8 figure is a different/earlier weighting.

## What changed from the pre-v1.1 (Jun-17) snapshot

- **New #1: fable-5 41.79** (was preliminary ~30.7). New rows scored:
  gpt-5.6-sol 36.23, opus-5 35.04, kimi-k3 31.96, opus-4.7 28.56,
  grok-4.5-high 23.45, gemini-3.1-pro 21.99.
- **Droppers** (contaminated/hosted-teacher runs removed): glm-5.2
  34.3 → 31.70; opus-4.8-max 34.1 → 32.90. opus-4.8 ~unchanged (33.8 → 33.84).
- Base (7.54) and human (51.19) reference rows unchanged. Weights unchanged.

## Weights, and how to read ROI

Weights: AIME .2265, GPQA .2246, HealthBench .1841, HumanEval .1061,
GSM8K .0936, ArenaHard .0904, BFCL .0746. Constructed so weight ×
(human − base) ≈ constant per benchmark — by human headroom all benchmarks
are worth about the same; prioritize by PROVEN-agent headroom (what a top
v1.1 agent has actually reached).

Best proven-agent average per benchmark (among top rows pulled 2026-08-09;
re-verify per-cell when starting each benchmark):
AIME fable-5 13.33 · ArenaHard fable-5 61.46 · BFCL gpt-5.6-sol 94.38 ·
GPQA opus-5 31.36 · GSM8K fable-5 83.53 · HealthBench fable-5 39.62 ·
HumanEval gpt-5.6-sol 63.95.

## Evaluation (v1.1)

Runs are judged by the v1.1 four-judge agent-as-judge system (data
contamination, API usage, PTB lookup, general). See `README.md` / the
harness for the pinned PTB version and judge invocation.

## Campaign table — per-cell (base / top-3 proven / human / ours)

Per-cell top-3 proven agents are pulled per benchmark as its campaign
starts (kept lean here to avoid stale numbers). Fill `Ours` + `Status` as
runs complete.

### AIME 2025 (weight .2265) — base 1.65 avg, best agent fable-5 13.33, human 29.17
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | — | — | — | pending |
| Qwen3-4B | 3.33 | 53.33 | — | pending |
| SmolLM3-3B | 3.33 | — | — | pending |
| gemma-3-4b | — | — | — | pending [G] |

### Arena Hard Writing (weight .0904) — base 1.27, best fable-5 61.46, human 70.23
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | — | — | — | pending [J] |
| Qwen3-4B | 3.42 | 86.84 | — | pending [J] |
| SmolLM3-3B | 0.42 | — | — | pending [J] |
| gemma-3-4b | — | 94.8 | — | pending [J][G] |

### BFCL (weight .0746) — base 1.5, best gpt-5.6-sol 94.38, human 85.0
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 0.0 | 94.0 | — | pending |
| Qwen3-4B | 0.0 | 95.0 | — | pending |
| SmolLM3-3B | 0.0 | 84.0 | — | pending |
| gemma-3-4b | 6.0 | 67.0 | — | pending [G] |

### GPQA Main (weight .2246) — base 8.49, best opus-5 31.36, human 36.47
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 14.06 | 35.5 | — | pending |
| Qwen3-4B | 13.39 | 44.64 | — | pending |
| SmolLM3-3B | 4.91 | 33.3 | — | pending |
| gemma-3-4b | 1.6 | 31.5 | — | pending [G] |

### GSM8K (weight .0936) — base 20.56, best fable-5 83.53, human 87.0
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 12.7 | 88.5 | — | pending |
| Qwen3-4B | 41.85 | 93.78 | — | pending |
| SmolLM3-3B | 21.08 | 82.2 | — | pending |
| gemma-3-4b | 6.1 | 83.5 | — | pending [G] |

### HealthBench (weight .1841) — base 9.49, best fable-5 39.62, human 43.33
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 7.5 | 44.9 | — | pending [J] |
| Qwen3-4B | 13.38 | 52.72 | — | pending [J] |
| SmolLM3-3B | 0.0 | 29.6 | — | pending [J] |
| gemma-3-4b | 17.04 | 46.1 | — | pending [J][G] |

### HumanEval (weight .1061) — base 12.81, best gpt-5.6-sol 63.95, human 71.49
| Model | Base | Human | Ours | Status |
|---|---:|---:|---:|---|
| Qwen3-1.7B | 7.9 | 68.9 | — | pending |
| Qwen3-4B | 36.59 | 77.44 | — | pending |
| SmolLM3-3B | 6.1 | 70.1 | — | pending |
| gemma-3-4b | 0.6 | 69.5 | — | pending [G] |

[J] = judge-scored, needs `openai-api-key`; [G] = gated model, needs HF
`hf-token` with the Gemma license accepted. Per-cell base/human values
marked "—" not yet re-pulled at the per-model level (only the 4-model means
above are confirmed); pull them when the benchmark's campaign starts.

## Our runs

_(clean slate — no v1.1 runs yet)_

| Run | Cell | Budget | Official score | Cost | Date | Details |
|---|---|---|---:|---|---|---|
| — | — | — | — | — | — | — |
