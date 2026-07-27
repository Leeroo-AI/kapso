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

## Campaign table — per-cell reference vs ours

For every cell (benchmark × base model): its base score, the top-3 proven
agent results from the leaderboard, human (official instruct), then ours and
status. Campaign aggregate if submitted today: **20.97** (base 7.53 + gsm8k net
+0.95 + bfcl net +1.80 + arena net +4.16 + aime net +2.08 + gpqa net +4.44).
**GPQA table COMPLETE** at 32.1/22.3/29.7/29.0 (4B/1.7B/SmolLM3/gemma) — our
4-model GPQA average 28.29 vs base 8.5 is the campaign's biggest single
weighted gain (.2246 × +19.79 = +4.44); three of the four sit in the proven
band (4B ~1σ under #1; only 1.7B trails, on subset-bias). AIME table complete
at 6.67/20.0/16.67/0.0. Two full benchmark tables (AIME, GPQA — the two
heaviest weights) plus Arena + BFCL now clean.
Cells needing unlocks are marked: [J] = judge-scored,
needs `openai-api-key` secret; [G] = gated model, needs HF `hf-token` with
the Gemma license accepted.

### AIME 2025 (weight .2265)

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 0.0 | opus-4.7 · 6.7 | **ours · 6.7** | opus-4.6-1m · 5.6 | 26.7 | **6.7 ✓clean** | ✅ run #25 (10h official, 2026-07-25): **6.67 ±4.6 via rescore — TIES proven #1**; on-VM final eval failed 9× (vLLM env issue, artifact intact), rescored officially on a fresh VM; post-hoc judges clean. `aime2025-qwen3-1-7b-base-07242238` |
| Qwen3-4B | 3.3 | opus-4.8-max · 23.3 | opus-4.8 · 23.3 | **ours · 20.0** | 53.3 | **20.0 ✓clean** | ✅ run #26 (10h official, 2026-07-25): **20.0 ±7.4 via rescore, judges clean (integrated)** — ties fable-5's #3 row, 1 question shy of proven #1; decode-forensics recipe (rp 1.1 sweet spot). on-VM final eval failed (qwen-AIME pattern, 2/2), rescored fresh. `aime2025-qwen3-4b-base-07250201` |
| SmolLM3-3B | 3.3 | opus-4.8-max · 16.7 | **ours · 16.7** | fable-5 · 16.7 | 26.7 | **16.7 ✓clean** | ✅ run #27 (10h official, 2026-07-25): **16.67 ±6.9, judges clean (integrated re-pin)** — TIES proven #1; in-run 23.33 banked (n=30 noise both ways); conciseness-anneal recipe. `aime2025-smollm3-3b-base-07250208` |
| gemma-3-4b | 0.0 | gpt-5.4-h-rp · 3.3 | opus-4.7 · 1.1 | gpt-5.3-codex · 1.1 | 10.0 | **0.0 ✓clean** | ✅ run #28 (10h official, 2026-07-25): **0.0 via rescore, judges clean (integrated)** — no lift over base (proven #1 is only 3.3; hardest AIME cell); in-run official never left 0/30 (dev MATH500 hit 0.41 but didn't transfer). SHARED-LEARNING RE-RUN (run #36, 2026-07-27): seeded the other 3 AIME cells' learnings — recipe fully adopted, still 0/30 across 3 iters → confirmed a capability wall, not a recipe deficit (`reviews/run36-review.md`). `aime2025-gemma-3-4b-pt-07250817` |

### Arena Hard Writing (weight .0904)

Figure: `arenahard_board.png` (aggregate + all four cells, site-styled).

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 0.9 | **ours · 67.8** | fable-5 · 57.1 | opus-4.8 · 45.0 | 50.0 | **67.8 ✓clean** | ✅ run #18 (10h official, 2026-07-19): **42.40 ±1.6, both judges clean** — 4th-tier row despite a 3h mid-run hang (dead ScheduleWakeup timer, reviews/run18-review.md); teacher-distill v1 17.06 → v2 34.6 → 42.4. `arenahardwriting-qwen3-1-7b-base-0718134` |
| Qwen3-4B | 3.4 | **ours · 89.6** | fable-5 · 86.2 | glm-5.2 · 54.2 | 86.8 | **89.6 ✓clean** | ✅ run #17 (10h official, 2026-07-19): **89.64 ±0.9, both judges clean — CELL RECORD, +3.4 over fable-5's 86.24 and +2.8 ABOVE the human/instruct row.** Relaxed-rules recipe: Qwen3-30B-A3B-Instruct-2507 teacher distillation (v3 18.2k ex → 88.64 in-run; v4 best-of-3 teacher-BoN data), baked decoding, DPO tried+rejected in 54min. Replaces run #16's 49.67. `arenahardwriting-qwen3-4b-base-07181341` |
| SmolLM3-3B | 0.4 | **ours · 56.6** | opus-4.8-max · 37.2 | glm-5.2 · 22.7 | 49.2 | **56.6 ✓clean** | ✅ run #19 (10h official, 2026-07-19): **56.58 ±1.6, both judges clean — top proven row, +7.4 ABOVE human** (best-known unlisted single trace remains 73.95; our es/fr-only multilingual slice left the zh/ru/ja axis unfixed, reviews/run19-review.md). `arenahardwriting-smollm3-3b-base-0718134` |
| gemma-3-4b | 0.3 | opus-4.8-max · 47.4 | **ours · 37.3** | opus-4.7 · 30.9 | 94.8 | **37.3 ✓clean** | ✅ run #24 (10h official, 2026-07-25): **37.28 ±1.5, post-hoc judges clean** — #2 proven; 27B-distill self-correction recipe. Gap study vs the 58.7 best trace (mean 47.4 hides 58.7+36.0) + the 94.8 human ceiling: `reviews/arena-gemma-3-4b-postmortem.md`. `arenahardwriting-gemma-3-4b-pt-07241547` |

Cell gap analysis (why 49.7 and not 86; ideation-priors / cross-run-memory /
feedback-judge diagnosis): `reviews/arena-qwen3-4b-postmortem.md`. Key
leaderboard nuance from scores.js: glm-5.2's 54.2 is a 3-run mean with std
16.5; fable-5's 86.2 is a single run 0.6 pts under the official-instruct row;
naive-scaffold opus-4.8 scores 41.6 ±3.1 (kapso +8 on the same model).
Best-known official traces per arena cell (from the public
`aisa-group/PostTrainBench-Trajectories` HF dataset; per-run bests, means
hid them — 1.7B 74.9, SmolLM3 74.0, both opus-4.8, both beat human):
recipes + review watchlists in `reviews/arena-best-baseline-traces.md`.

### BFCL (weight .0746)

Task definition (verified against the harness): the `exec_simple` subset
of gorilla-llm/Berkeley-Function-Calling-Leaderboard, pinned revision,
**exactly 100 samples** — the complete official PostTrainBench test set
(1 sample = 1 point; finals run all 100/100). All rows below, including
human and proven agents, are scored on this same task; it is NOT the full
multi-category public BFCL suite. Leaderboard cells are MEANS over ~3
repeated runs (scores.js stores e.g. glm-5.2 = 95.33 with a separate std
table), while each single run scores an integer /100 — so our single-run
96.0/93.0 sit atop measured 95-96/92-93 bands; claim "at or above" the
95.3 record, and run 2-3 repeats via `gcp/40_eval_only.sh` before any
strict parity claim to maintainers.

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 0.0 | gpt-5.5-xh-rp · 100.0 | glm-5.2 · 95.3 | opus-4.5-oc · 92.0 | 94.0 | **96.0 ✓clean** | ✅ run #9 (10h official, 2026-07-16, done in 5.6h): **96.0 with 'no contamination detected'** — retires run #8's caveat, beats GLM-5.2's record, 2nd all-time. Judge stopped at practical ceiling after iteration 1. Run #8 (also 96.0, caveated) superseded. |
| Qwen3-4B | 0.0 | gpt-5.4-h-rp · 100.0 | fable-5 · 100.0 | opus-4.6-1m · 97.3 | 95.0 | **95.0 ✓clean** | ✅ run #14 (10h official, 2026-07-18): **95.0, 'no contamination detected'** — TIES human exactly; deterministic (greedy artifact config), official run reproduced it precisely. `bfcl-qwen3-4b-base-07171548` |
| SmolLM3-3B | 0.0 | gpt-5.5-xh-rp · 100.0 | opus-4.8 · 97.0 | opus-4.6 · 86.7 | 84.0 | **93.0 ✓clean** | ✅ run #10 (10h official, 2026-07-16): **93.0, 'no contamination detected'** — 3rd all-time on the cell, +9 over human. 4 iterations, 3 promotions (SFT 92 → soup 93 → soup-tie 93), strict promote gates held |
| gemma-3-4b | 6.0 | gpt-5.5-xh-rp · 100.0 | gpt-5.4-h-rp · 100.0 | fable-5 · 100.0 | 67.0 | **93.0 ✓clean** | ✅ run #15 (10h official, 2026-07-18): **93.0, 'no contamination detected'** — +26 over human, 4th all-time. SFT 89 → GRPO 91 (campaign-first RL stage, legal reward) → greedy fix 92 → official 93. `bfcl-gemma-3-4b-pt-07171548` |

### GPQA Main (weight .2246)

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 14.1 | gpt-5.2-codex · 29.5 | gpt-5.5-xh-rp · 29.5 | gpt-5.4-high · 29.4 | 35.5 | **22.3 ✓clean** | ✅ run #33 (10h official, 2026-07-26): **22.32 ±2.0 via rescore, judges clean** — +8.2 over base, below the 29.5 proven band; in-run champion read 28.67 @150 but the first-150 subset proved a BIASED estimator of the full 448 (see review); 529-killed iter-1 recovered in 4 min; endgame champion-defense exemplary. `gpqamain-qwen3-1-7b-base-07252105` |
| Qwen3-4B | 13.4 | gpt-5.4-h-rp · 34.1 | gpt-5.5-xhigh · 34.0 | gpt-5.5-xh-rp · 33.9 | 44.6 | **32.1 ✓clean** | ✅ run #32 (10h official, 2026-07-26): **32.14 ±2.2 via rescore, judges clean** — 4th-tier row 1.8 under proven #1 (within ~1σ of the 33.9-34.1 band), +18.7 over base; teacher distillation + rep_penalty 1.1 (the AIME decode lever generalized); agent's own full-448 confirm read 33.48. `gpqamain-qwen3-4b-base-07252103` |
| SmolLM3-3B | 4.9 | gpt-5.5-xh-rp · 30.6 | glm-5.2 · 29.8 | gpt-5.4-high · 29.0 | 33.3 | **29.69 ✓clean** | ✅ run #34 (10h official, 2026-07-26): **29.69 ±2.2, judges clean** — mid-proven-band (between #2/#3), +24.8 over base 4.9, 3.6 under human; big recovery from iter-1's 16.67 parse-wall floor via R1-0528 science-MCQ SFT with anchored `ANSWER: X` tails. Score is the in-run full-448 official `evaluate.py` on the shipped final_model; a fresh-VM rescore couldn't serve it (vLLM-0.11 CUDA-graph illegal-memory crash on SmolLM3's NoPE arch — the model serves fine in the agent's own container, gemma/qwen rescore fine). `gpqamain-smollm3-3b-base-07260818` |
| gemma-3-4b | 1.6 | gpt-5.4-high · 29.5 | opus-4.7 · 28.7 | opus-4.8 · 28.7 | 31.5 | **29.0 ✓clean** | ✅ run #35 (10h official, 2026-07-26): **29.02 ±2.1 via rescore, judges clean** — mid-proven-band (ties #2/#3, 0.5 under #1), +27.4 over base 1.6, 2.5 under human, on the WEAKEST base of the table; monotone 26.34→27.72→30.36 in-run, final lever a zero-train model soup (0.5·StageB + 0.5·StageC, md5-verified). `gpqamain-gemma-3-4b-pt-07260822` |

### GSM8K (weight .0936)

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 12.7 | fable-5 · 86.3 | glm-5.2 · 79.6 | gpt-5.4-h-rp · 73.9 | 88.5 | **53.4** | ✅ run #7, 3h validation (2026-07-15) |
| Qwen3-4B | 41.9 | fable-5 · 90.7 | opus-4.8-max · 89.9 | glm-5.2 · 85.3 | 93.8 | — | pending |
| SmolLM3-3B | 21.1 | opus-4.8 · 76.7 | gpt-5.4-h-rp · 73.3 | gpt-5.5-xh-rp · 72.4 | 82.2 | — | pending |
| gemma-3-4b | 6.1 | fable-5 · 75.7 | opus-4.8-max · 69.7 | glm-5.2 · 61.9 | 83.5 | — | pending [G] |

### HealthBench (weight .1841)

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 7.5 | fable-5 · 34.5 | opus-4.8 · 31.2 | gpt-5.4-h-rp · 29.4 | 44.9 | — | pending [J] |
| Qwen3-4B | 13.4 | opus-4.8-max · 41.8 | opus-4.8 · 41.6 | gpt-5.5-xh-rp · 33.7 | 52.7 | — | pending [J] |
| SmolLM3-3B | 0.0 | opus-4.8 · 38.4 | opus-4.8-max · 32.6 | fable-5 · 32.6 | 29.6 | — | pending [J] |
| gemma-3-4b | 17.0 | fable-5 · 46.3 | opus-4.8-max · 42.0 | opus-4.8 · 29.9 | 46.1 | — | pending [J][G] |

### HumanEval (weight .1061)

| Model | Base | #1 proven | #2 proven | #3 proven | Human | Ours | Status |
|---|---:|---|---|---|---:|---:|---|
| Qwen3-1.7B | 7.9 | fable-5 · 62.8 | gpt-5.5-xh-rp · 61.0 | glm-5.2 · 55.3 | 68.9 | — | pending |
| Qwen3-4B | 36.6 | fable-5 · 78.0 | glm-5.2 · 76.0 | opus-4.8 · 71.3 | 77.4 | — | pending |
| SmolLM3-3B | 6.1 | opus-4.8 · 55.2 | gpt-5.5-xh-rp · 43.9 | gpt-5.4-h-rp · 42.7 | 70.1 | — | pending |
| gemma-3-4b | 0.6 | fable-5 · 54.9 | opus-4.8 · 48.2 | gemini-3-pro-oc · 45.7 | 69.5 | — | pending [G] |

Abbreviations: h-rp / xh-rp = high/xhigh reprompted variants; oc = OpenCode
scaffold. The benchmark-level ROI table above remains the prioritization
guide; this table is the per-cell scoreboard.

## Our runs

| Run | Cell | Budget | Official score | Cost | Date | Details |
|---|---|---|---:|---|---|---|
| #10 | bfcl × SmolLM3-3B | 10h (official length) | **93.0 (clean)** | ~$55 GPU + notional Max | 2026-07-16 | Full 10h, 4 iterations, 3 promotions: full-FT SFT on xLAM (92.0) → cross-dist soup (93.0) → arg-fidelity soup tie (93.0, higher internal EM); exp3/exp6 correctly NOT promoted (tie/below). Ideation PREVENTED the special-token trap run #9 hit. Judges: clean. 3rd all-time on cell, +9 over human. Findings: `reviews/run10-review.md`. |
| #9 | bfcl × Qwen3-1.7B | 10h (official length) | in flight | — | 2026-07-16 | Re-run of the #8 cell with the full fix stack live: gpt-5.6-luna xhigh memory loop (first run with cross-iteration insights), env_strip containment, prompt-via-stdin, feedback invariants + session_end_facts, ensemble forensics + retry, rotated Max OAuth token. Goal: clean ≥94 to retire run #8's contamination caveat. Findings: `reviews/run9-review.md`. |
| #8 | bfcl × Qwen3-1.7B | 10h (official length) | **96.0** (full set, first-attempt eval) | ~$70 GPU + $39.65+ notional Max | 2026-07-15/16 | First full-stack run: ensemble ideation (codex+fable-5, xhigh) + opus-4.8 xhigh implementation + F5 contract. 4 iterations: 0→93 (SFT 44k) →94 (self-mined DPO) →94 (soup) →**96** (convention-patch SFT). Beats human/instruct (94.0) and GLM-5.2's cell record (95.3); trails only gpt-5.5-xh-rp (100). Iteration-1 self-kill footgun (R8-F8) recovered by feedback+parent-ladder. Judge: R8-F17 RESOLVED 2026-07-16 (openai-api-key secret; gpt-5.1-codex verified via CODEX_API_KEY; agent phase keeps subscription auth). Judge-scored cells [J] unblocked. Findings: `reviews/run8-review.md`. |
| #7 | gsm8k × Qwen3-1.7B | 3h (validation) | **53.4 ± 1.4** (full 1319-problem set, rescored via `gcp/40_eval_only.sh`) | ~$17 GPU + $36.77 notional Max | 2026-07-14/15 | Full-FT on MetaMathQA, checkpoint-2000 promoted mid-run; 62% of the cell's proven-best (86.3 in 10h), reached in 30% of the time. Findings F0–F14: `reviews/run7-review.md`; F5 fix applied post-run (session-cap contract + sizing + timeout backstop, commit 434f66da). |

Latest: **runs #9 AND #10 COMPLETE, both official-clean** (2026-07-16):
bfcl × Qwen3-1.7B **96.0** (5.6h, judge-stopped at ceiling) and bfcl ×
SmolLM3-3B **93.0** (full 10h, 4 iterations). Aggregate-if-submitted
≈**12.0**. Known bounded issue in both runs' containers: gpt-5.6-luna
memory-layer calls 400 (litellm version skew nulls reasoning_effort) and
fail soft — memory dead, runs unaffected; fix landed on main (a43fe829: litellm==1.75.0 pin + no-null hardening),
ships with the post-run-10 rebuild. Gemma unblocked 2026-07-16: license
accepted + `hf-token` secret live; the same rebuild bakes gemma into the
cache snapshot. **Runs #11 + #12 LAUNCHED in parallel 2026-07-17 ~08:25 UTC** from
`46e6390f` (first runs with: live luna memory layer, judge per-sample
leak invariant, 1800s iteration-admission floor, gemma in warm cache):
`bfcl-qwen3-4b-base-07170824` and `bfcl-gemma-3-4b-pt-07170825`.
**Runs #11-13 stopped at ~t+5h (2026-07-17 13:5x UTC) by user decision**
to land framework upgrades before continuing (interim bests archived +
rescorable). **Runs #14-16 launched fresh 2026-07-17 ~15:48 UTC** on the
upgraded stack (0985fa1f: technical_difficulties capture chain +
fallback, per-session stream forensics, insight machinery removed —
difficulties IS the lesson artifact, full-content memory renders,
litellm pin, rotated OAuth token): `bfcl-qwen3-4b-base-07171548`,
`bfcl-gemma-3-4b-pt-07171548`, `arenahardwriting-qwen3-4b-base-07171548`
— same three cells, three parallel H100s.

## Run artifact index (GCS)

Everything needed for maintainer submission and offline analysis streams
to `gs://trans-density-437811-p2-posttrainbench/results/<run_id>/` (5-min
rsync from the VM; complete once `RUN_DONE` appears). Fetch a whole run
with `gcp/20_fetch_results.sh <run_id>`. Layout per run:

```
<root>/ptb-run.log                      VM startup + harness orchestration log
<root>/RUN_DONE                         completion marker (upload finished)
<root>/results/kapso_<agent-config>_<N>h/<eval>_<model>_<run_id>/
    solve_out.txt                       FULL timestamped agent stream (the trace reviews read this)
    output.log / error.log              runner stdout / stderr
    prompt.txt                          official task prompt as received
    final_eval_*.txt                    official eval attempts (the score of record)
    judge_output.txt / .json            contamination judge verdict (absent on run #7: no judge key yet)
    time_taken.txt                      wall-clock accounting
    final_model/                        submitted weights (config.json + safetensors + tokenizer)
    task/                               agent task dir: PLAN.md, best_score.log, artifacts/, training logs
```

| Run | Cell | Score | Run id / GCS root suffix | Review doc |
|---|---|---:|---|---|
| #16 | arenahard × Qwen3-4B 10h | **49.67** | `arenahardwriting-qwen3-4b-base-07171548` | `reviews/run16-review.md` |
| #17 | arenahard × Qwen3-4B 10h | **89.64** | `arenahardwriting-qwen3-4b-base-07181341` | `reviews/run17-review.md` — CELL RECORD, above human; first relaxed-rules run |
| #18 | arenahard × Qwen3-1.7B 10h | **42.40** | `arenahardwriting-qwen3-1-7b-base-0718134` | `reviews/run18-review.md` — clean despite 3h dead-timer hang (R18-P2-1) |
| #19 | arenahard × SmolLM3 10h | **56.58** | `arenahardwriting-smollm3-3b-base-0718134` | `reviews/run19-review.md` — top proven row, above human; zh/ru/ja mix unfixed |
| #20 | arenahard × Qwen3-1.7B 10h | ABORTED @t+2h | `arenahardwriting-qwen3-1-7b-base-0724124` | killed 2026-07-24 14:2xZ to re-point oauth token (wrong .env source); P1 review captured the new-stack debut (clean) + R20-P1-1 language-axis miss — `reviews/run20-review.md` |
| #21 | arenahard × Qwen3-1.7B 10h | ABORTED @t+50m | `arenahardwriting-qwen3-1-7b-base-0724142` | killed 15:0xZ — superseded by the negative-space coverage build (816500d1) |
| #22 | arenahard × gemma-3-4b 10h | ABORTED @t+25m | `arenahardwriting-gemma-3-4b-pt-07241440` | killed 15:0xZ (same); its first attempt 07241426 died on the transient host-python preflight, boot now hardened (2703dc24) |
| #23 | arenahard × Qwen3-1.7B 10h | **67.82 ✓clean** | `arenahardwriting-qwen3-1-7b-base-0724154` | official 0.6782 ±0.015 (9h29) — **CELL RECORD: +10.7 over fable-5's proven 57.1, +17.8 over human**; post-hoc judges clean ('no contamination detected' / 'only allowed use detected', gpt-5.6-sol replicating the exact harness stage after upstream's gpt-5.1-codex deprecation; verdicts in postjudge/); `reviews/run23-review.md` |
| #26 | aime2025 × Qwen3-4B 10h | **20.0 ✓clean** | `aime2025-qwen3-4b-base-07250201` | official via rescore (on-VM final eval failed 9× — qwen-AIME serving pattern 2/2; judges clean integrated) — 1 shy of proven #1; in-run pooled ≈24; `reviews/run26-review.md` |
| #36 | aime2025 × gemma-3-4b 10h (shared-learning experiment) | **0.0 ✓clean** | `aime2025-gemma-3-4b-pt-07262239` | cross-run-knowledge test: seeded `knowledge/aime2025.md` (other 3 cells' learnings); gemma READ+adopted the recipe fully but stayed 0/30 across 3 iters — capability wall, not knowledge deficit; official via rescore (clean 30/30); `reviews/run36-review.md` |
| #35 | gpqamain × gemma-3-4b 10h | **29.02 ✓clean** | `gpqamain-gemma-3-4b-pt-07260822` | official via rescore (7/7 serving-fail; judges clean) — mid-proven-band on the weakest base; model-soup final lever; `reviews/run35-review.md` |
| #34 | gpqamain × SmolLM3 10h | **29.69 ✓clean** | `gpqamain-smollm3-3b-base-07260818` | in-run full-448 official (mid-band, recovered from 16.67 parse floor); fresh-VM rescore blocked by SmolLM3 vLLM CUDA-graph crash (arch-specific); `reviews/run34-review.md` |
| #33 | gpqamain × Qwen3-1.7B 10h | **22.32 ✓clean** | `gpqamain-qwen3-1-7b-base-07252105` | official via rescore (final-eval fail 6/6 + first rescore attempt died on GATED gpqa dataset — hf-token fix 88fe3914); @150-subset bias exposed (28.67 in-run vs 22.32 full); 529-kill recovered in 4 min; `reviews/run33-review.md` |
| #32 | gpqamain × Qwen3-4B 10h | **32.14 ✓clean** | `gpqamain-qwen3-4b-base-07252103` | official via rescore — 1.8 under proven #1 34.1 (within ~1σ); iter-1 16.0-below-random recovered to 33.48 full-448 in-run (teacher distill + rp 1.1); failover-stack debut, zero limit events; `reviews/run32-review.md` |
| #31 | gpqamain × SmolLM3 10h | STOPPED @t+~35m | `gpqamain-smollm3-3b-base-07251809` | GPQA queue #3 (launched 18:09Z); stopped 2026-07-25 ~18:4xZ with #29/#30 — user call: implement OAuth failover first; trace archived (GCS + scratchpad stopped_runs_archive) for later study |
| #30 | gpqamain × Qwen3-1.7B 10h | STOPPED @t+~6.5h | `gpqamain-qwen3-1-7b-base-07251202` | GPQA queue #2 (launched 12:02Z); starved by the shared-token session limit ~18:20→20:10Z, stopped ~18:4xZ for the failover build; trace archived for later study |
| #27 | aime2025 × SmolLM3 10h | **16.67 ✓clean** | `aime2025-smollm3-3b-base-07250208` | official (9h18) TIES proven #1; first fully-integrated judge re-pin test PASSED; `reviews/run27-review.md` |
| #29 | gpqamain × Qwen3-4B 10h | STOPPED @t+~6.8h | `gpqamain-qwen3-4b-base-07251146` | GPQA queue #1 (launched 11:46Z); starved by the shared-token session limit ~18:20→20:10Z, stopped ~18:4xZ for the failover build; trace archived; relaunch heads the post-failover queue |
| #28 | aime2025 × gemma-3-4b 10h | **0.0 ✓clean** | `aime2025-gemma-3-4b-pt-07250817` | official via rescore (on-VM final eval failed 9× — 4th occurrence, now proven model-agnostic; judges clean integrated) — no lift over base 0.0 on the hardest AIME cell (proven #1 only 3.3); completes the AIME table; `reviews/run28-review.md` |
| #24 | arenahard × gemma-3-4b 10h | **37.28 ✓clean** | `arenahardwriting-gemma-3-4b-pt-07241547` | official 0.3728 ±0.015 (9h11) — first kapso row on this cell (base 0.3; proven 47.4 stands); post-hoc judges clean (verdicts in postjudge/); `reviews/run24-review.md` |
| #25 | aime2025 × Qwen3-1.7B 10h | **6.67 ✓clean** | `aime2025-qwen3-1-7b-base-07242238` | official via rescore (on-VM final eval failed 9× — vLLM env issue, artifact intact; judges clean post-hoc) — TIES proven #1 6.7; in-run best 3.33; `reviews/run25-review.md` |
| #15 | bfcl × gemma-3-4b 10h | **93.0** | `bfcl-gemma-3-4b-pt-07171548` | `reviews/run15-review.md` |
| #14 | bfcl × Qwen3-4B 10h | **95.0** | `bfcl-qwen3-4b-base-07171548` | `reviews/run14-review.md` |
| #10 | bfcl × SmolLM3-3B 10h | **93.0** | `bfcl-smollm3-3b-base-07161232` | `reviews/run10-review.md` |
| #9 | bfcl × Qwen3-1.7B 10h | **96.0** | `bfcl-qwen3-1-7b-base-07160950` | `reviews/run9-review.md` |
| #8 | bfcl × Qwen3-1.7B 10h | **96.0** | `bfcl-qwen3-1-7b-base-07152141` | `reviews/run8-review.md` |
| #7 | gsm8k × Qwen3-1.7B 3h | **53.4** | `gsm8k-qwen3-1-7b-base-07142047` | `reviews/run7-review.md` |

Debug/infra runs #1–6 (no scores; boot + harness shakeout, 2026-07-14)
share the same root pattern: `gsm8k-qwen3-1-7b-base-0714{0857,1016,1035,1233,1403,1424}`.

Not uploaded (by design, lives only on the ephemeral VM): the kapso
workspace git repo with `.kapso/ideation/` member artifacts — its
text content is fully embedded in `solve_out.txt`; treat that file as
the forensic source of truth after the VM self-deletes.
