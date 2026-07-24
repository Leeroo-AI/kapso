# Run #23 review — arenahardwriting × Qwen3-1.7B-Base (0724154)

Debut of the negative-space coverage build (816500d1) — the R20-P1-1
counterfactual run. Re-run of the cell scored 42.40 by #18; best-known
trace 74.85 (opus-4.8). Launched 2026-07-24 15:45Z. Dual-mandate reviews
per `arena-best-baseline-traces.md` (Cell 2 watchlist).

## P1 (t+0 → ~t+88min)

Headline: **the negative-space contract changed behavior end-to-end** —
candidates closed every family with Not-measured lines; the language mix
was MEASURED at ideation ("≥33% non-Latin: zh 35, ru 26, ar 2, ko 2,
other 17 — script-detection scan"); the selector substantively exercised
the audit ("I verified the contested claims directly against evaluate.py
and templates/qwen3.jinja", demoted two candidates for weak groundedness,
and MINTED A NEW Not-measured line during synthesis — no_robots CC-BY-NC
licensing + auto-replace remedy); and the implementor's recon explicitly
consumed the targets: "Let me settle two cheap 'not measured' sub-axes
now" → Latin-script breakdown (en 140 = 56%, vi 14, es 8, fr 5…) settling
the ASSUMED claim, plus a register split — exceeding the
settle-one-per-iteration rule. Strongest first 90 minutes of any run on
this cell: teacher (Qwen3-30B-A3B-Instruct-2507-FP8) live at t+21m, pool
built AND rebalanced against the measured mix (agent flagged "record the
coverage discrepancy" when the initial pool came out 85% Latin), SFT
airborne at t+85m, #18's traps all pre-empted (length guidance 400–1100
tok vs #18's 367 miss; generation_config {eos [151645,151643],
t0.7/p0.8/k20/rep1.05} pinned into the first-artifact spec; masking
CPU-validated).

- **R23-P1-1 — P2 (recipe), 17:12:21.** Multilingual drift AT BUILD TIME:
  gated SFT set 4,588 ex = 22.9% non-Latin vs the measured 32.8% target;
  zh survival through judge-gating only 62% (471/754) vs 88% overall. The
  agent noted judge generosity but not the zh attrition. The residual
  class the contract doesn't yet reach: coverage is audited at CLAIM time
  but not re-checked against the ARTIFACT actually built. Per-stratum
  eval exists — P2 must check it catches this empirically.
- **R23-P1-2 — P2 (framework/observability), 16:16:24.** eval_profile.md
  content unverifiable in-trace: Write bodies aren't streamed (only "File
  created"). Recon settled two sub-axes 30s before the write so content
  is likely compliant — verify the artifact in P2. Framework gap:
  contract-bearing Write bodies are invisible to reviewers.
- **R23-P1-3 — OBS, 16:03:17.** Selector audit depth: strong (see
  headline), though the punitive branch (missing Not-measured line →
  ASSUMED demotion) went untested — all 4 candidates carried complete
  Coverage on the first try.
- **R23-P1-4 — OBS, 15:59:41.** Codex member output not streamed
  (recurring, 4th run).
- **R23-P1-5 — OBS, 16:07:00.** core.gotchas rendered header-only —
  plausibly empty on a fresh campaign; adjacent to R16-P2-2, watch.
- **R23-P1-6 — OBS, 16:25:29.** Self-caught near-miss: killed a
  just-started 5,200-prompt generation ("max_model_len=5120… can exceed
  context and crash the judge phase ~40 min in"), guarded, restarted at
  8192 (~5 min cost). Plus a trivial PYTHONPATH fix. Agent-recovered.

Clean checks: ScheduleWakeup zero; all GPU-bound waits used
tracked-completion + dead-man's alarms, zero idle-GPU gaps; boot clean;
lens planner healthy (~2 min, both lenses flagged multilinguality —
independent second door for the language axis); 2/2+2/2 candidates.

SELECTED PLAN: (1) judge-gated best-of-2 distillation of
Qwen3-30B-A3B-2507-FP8 into 1.7B (5,200 language-balanced prompts, exact
qwen3.jinja empty-think render, early promote); (2) on-policy DPO vs the
exact rubric (">>" pairs ×2 mirroring the ×3 weighting) with automatic
RAFT fallback, ~45-min cap; (3) temp sweep baked into generation_config;
optional round-2 widening (best trace used 3 rungs — watch). SFT 2ep/1e-5
vs best-trace 3ep/2e-5. Baseline 0.0 @50 confirmed; first student eval
expected ~t+2.5–3h.

Verdict: **continue** — every known trap pre-empted, contract validated at
depth; P2 watches: zh attrition (does per-stratum eval catch it?), first
eval score, DPO time-box, single-rung vs 3-rung widening decision.

## P2 (17:13 → 20:33Z)

Score ladder verified: 0.0 base → v1 0.0317 (17:47, broken, promoted
17:56 as `exp1-sft-bf16-INSURANCE`) → v2 0.4072 (18:43, promoted 18:45)
→ v3 0.625 (20:25, promoted 20:25, `exp1-sft-fp32-v3-9k-neftune`).
At cutoff: limit-100 confirmation eval running (launched 20:27, ETA
~20:40); run clock 5:22 remaining, session ends 21:03.

- **R23-P2-1 — RECIPE/agent (resolved), 17:47:57.** v1 root cause:
  **pure-bf16 full-FT with lr 1e-5 on tied embeddings** — updates below
  bf16 rounding corrupted the shared embedding/LM-head ("garbage
  rare-token prefixes (`𬜯`, `JSGlobalScope`)… even under greedy"; loss
  plateaued 1.42). Agent bug, not framework. Diagnosis was exemplary:
  read stored answers → controlled greedy-vs-baked A/B (diag_gen.py, 5
  fixed probes) isolating model-vs-sampling → checked
  `tie_word_embeddings: True` → fp32 master weights + bf16 autocast →
  loss confirmation (0.98 vs 1.42) → greedy re-probe clean BEFORE
  spending an eval. ~73 min lost end-to-end (incl. the step-0 OOM,
  17:17, batch 8→4×16+grad-ckpt, ~6 min). Lesson persisted to agent
  memory (`qwen3-fulltft-fp32-master-weights.md`).
- **R23-P2-2 — RECIPE (OPEN), 19:27:41.** R23-P1-1 drift **worsened**:
  batch B only 13.3% non-Latin (latin 3938/4544; zh survival again 61%
  = 460/754, ru pool exhausted at 155 prompts); combined 9k set =
  18.2% non-Latin vs 32.8% target (was 22.9% in batch A). Agent noticed
  ("ru is thinner now") but waved it off: "combined coverage is fine" —
  no re-audit against the measured mix. Per-stratum eval RAN on all
  three evals and did catch v1's zh=0.0 catastrophically, but at n_q~3
  zh it cannot resolve drift-sized effects. Risk is real: the FULL eval
  set is 32.8% non-Latin vs ~17% in the limit-50 subset, so 0.625 is
  latin-weighted.
- **R23-P2-3 — RECIPE (watch), 20:25.** DPO stage (plan step 2, ~45-min
  box) and temp sweep NOT started — the v1 debacle consumed that
  window; agent banked limit-100 confirmation instead with ~37 min of
  session left. Correct triage; DPO + third rung must come from the
  remaining 5:22 run clock. v2→v3 delta: data 4.6k→9.1k (fresh batch B,
  exclude-list correct), NEFTune α=5 back on, epochs 3→2 — this IS the
  second rung, and re-adding NEFTune at fp32 retro-cleared it of v1
  blame.
- **R23-P2-4 — OBS.** Judged-question counts drifted across identical
  `--limit 50` evals: 63 (v1) / 48 (v2) / 46 (v3) — judge drop-outs mean
  rungs compare on slightly different subsets (v1's 63 > 50 unexplained).
- **R23-P2-5 — OBS.** Two more self-recovered nits: eval wrapper cwd
  crash on relative `question.jsonl` path (17:38, fixed in ~1 min —
  caught instantly because the tracked wait returned early); promote
  script copied training checkpoints into final_model (fixed 17:56).

Framework hygiene clean: dead-man alarms on every wait, max silent gap
16 min (18:43→18:59, alarm-covered), zero >20 min; ScheduleWakeup zero;
harness BLOCKED a foreground `sleep 195` chain (17:19) and the agent
complied with tracked probes; 1 rate-limit event, no impact; promotion
discipline strong (insurance promote, verify step, best_score.log,
post-promote vLLM integrity smoke). R23-P1-2 persists: eval_profile.md
edited 20:27:57, body still invisible in-trace. No session boundary in
window (R15-P2-1/R16-P2-2 n/a).

LADDER: 0.0317 (bf16-broken) → 0.4072 (fp32, t+3:00) → 0.625 (9k+NEFTune,
t+4:40) vs best-known 0.47 → 0.66 → 0.72 — on-pattern, half a rung behind
at comparable clock, third rung (DPO) still unspent.

SOTA OUTLOOK: 0.625 at n=46 carries ±7pt stderr and a latin-heavy subset
bias (full set is ~2× more non-Latin), so the honest read is 0.50–0.70
with a central estimate near 0.60; that makes beating proven 57.1 roughly
a coin-flip-plus (~75%) on SFT alone, with the limit-100 run about to
halve the error bars. Reaching 74.85 requires the best trace's third-rung
gain (+0.10–0.12) from DPO/widening in the remaining 5:22 — credible
given rung-2 landed on-pattern, but only if the multilingual strata
(R23-P2-2) don't bleed on the full set.

VERDICT: **continue** — a broken rung was diagnosed at textbook depth and
converted into two clean promotions plus a banked lesson; the one open
threat (multilingual build-time drift) needs a data-mix correction before
any rung-3 generation, not an intervention now.
