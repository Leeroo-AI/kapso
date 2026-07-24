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
