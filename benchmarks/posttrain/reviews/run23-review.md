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

## P3 (20:33Z → end 01:15:56Z) + closing

Delta arc: limit-100 confirm 0.6528 (20:37, session 1 closed clean at
20:39 with `<score>0.6528` + 5-entry difficulties, $33.07/184 tools) →
iteration-2 boundary (judge verify + 2×2 ensemble ideation + selector,
20:41–20:58) → iteration 2 (opus, 20:58–00:28, $31.60/214 tools): three
teacher-INTERVENTION levers all regressed on the real judge, pivot at
22:25 to more teacher-WRITTEN data → **v4 from-base retrain 0.68@100
(00:01, promoted) → 0.6855@250 confirm (00:26)** → iteration-3 boundary
(<stop>false, 1:11 left) → iteration 3 (00:50–01:09, $2.29): insurance-
first endgame + one α=0.7 v3/v4 weight soup, 0.6145@100, discarded, floor
intact → final judge <stop>true → "Stopping: goal achieved" 01:15:56,
t+9h30, ~31 min unused. Total agent cost **$74.92** (~9.5 H100-hours).

- **R23-P3-1 — FRAMEWORK (alarm14 root-caused; low sev, real exposure).**
  The 56-min CLI silence 22:51→23:49 is fully explained in-trace: alarm14
  was launched at 22:49:55 as `nohup bash -c 'sleep 900 && echo
  alarm14-tick' > alarm14.log` INSIDE the same tracked background task
  (`bhiuzyeit`) as the training-completion waiter — a detached, log-only
  process structurally incapable of notifying the CLI. It DID run on
  schedule (alarm14.log mtime 23:04, 13 bytes); no notification was ever
  lost because none could exist; the one notification (bhiuzyeit) fired
  exactly at train-done 23:49:59. Systematic: ALL session-2 "alarms"
  (3,4,6,7,8,10,11,13,16) used this dud idiom — every session-2 wake was
  waiter-driven — whereas session 1's alarms were standalone tracked
  tasks that really fired (20:36 "Alarm fired"). The dead-man redundancy
  ran one-legged for 7.5h and only the 61-min v4 train made it visible.
  Zero material cost here (GPU busy 100% of the silence; wake landed at
  the exact useful moment), but a hung waiter would have stranded the
  session until the cap. Propose (no unilateral patch): session-prompt
  rule that an alarm must be its OWN `run_in_background` task, and/or a
  harness lint flagging nohup-detached "alarm" processes.
- **R23-P3-2 — FRAMEWORK (upgraded-stack boundary, strong).** Both
  boundaries ran the full new machinery end-to-end. Implementor
  self-authored `<technical_difficulties>` are genuinely lesson-grade
  (iter 1: fp32-master/tied-embeddings with mechanism; EngineCore orphan
  kill; iter 2: teacher-proxy inversion, cwd-race, no-partial-save
  refactor). Extraction pulled all five tags incl. evaluation_script_path
  (R15-P2-1 exercised: judge verified evaluate.py "mtime unchanged
  15:47", and at iter 2 that the official script itself printed
  0.6855345911949685 at log line 514). Feedback judges (opus, $0.86–1.42,
  7–18 tools) verified rather than trusted: md5 final_model==winner,
  orphaned-value audit, batchC drawn=5087/generated=2222 accounting, and
  the final judge even caught the implementor's overclaim ("both
  result.json copies consistent" — only the task-root copy existed).
  ExperimentHistoryStore persisted experiments 0/1/2; ideation members
  held the new MCP tools and USED them — get_top/get_recent{k:5} at iter
  2, and at iter 3 `search_similar_experiments("model soup weight
  averaging…")` before proposing the soup; full solution/feedback/
  difficulties content demonstrably reached later candidates (soup plan
  cites the exp0 tied-embeddings lesson verbatim). Ensemble 2 members ×
  2 candidates pooled 4/4 at both boundaries (codex gpt-5.6-sol 410s/
  406s, 0 dropped); fable-5 selectors verified decisive facts with their
  own tool calls (timer arithmetic killed clock-infeasible candidates)
  and synthesized across candidates both times.
- **R23-P3-3 — RECIPE (the pivot; exemplary empiricism, humbling for
  selection).** The selected iter-2 plan (edit-RAFT primary, DPO
  opportunistic) went 0-for-3 on gpt-5-mini: greedy 0.05 (the 30B
  teacher-proxy sweep had ranked greedy BEST at 0.522 — teacher-signal
  inversion caught only by real-judge confirm before a config bake),
  edit-RAFT 0.5078, DPO 0.5278; the DPO result (reward margins ~0.02,
  model barely moved, yet −0.10) exposed **limit-50 noise ≈ ±0.10**. At
  22:25 the implementor pivoted on the correct abstraction — teacher
  AUTHORSHIP transfers, teacher INTERVENTION doesn't — drew 5,087
  weak-strata (short/medium) prompts from the 7,932 unused pool (ru
  exhausted), time-cut generation at 2,222, and retrained FROM BASE on
  11,354 ex with v3's exact recipe (fp32 master + bf16 autocast, NEFTune
  5, 2 ep, 504 steps, 61 min, loss ~1.12) → 0.68@100 vs incumbent
  0.6528, promoted with sanity-gate + md5 + monotonic best_score.log,
  confirmed 0.6855@250. The iter-2 judge graded the inversion honestly:
  the selector's primary lever lost, the "lowest expected marginal gain"
  axis it had demoted won (+0.033). DPO is no longer "unspent" — it is a
  measured settled-negative for this cell, alongside greedy, edit-RAFT,
  and the α=0.7 soup (0.6145, below BOTH parents: not mode-connected).
- **R23-P3-4 — RECIPE (multilingual drift CLOSED, sign inverted).**
  R23-P2-2's feared bleed inverted: full-250 strata are latin 0.6557
  (n≈175) / zh 0.7946 / ru 0.7083 / ja 0.8438 — non-Latin is the
  STRONGEST region despite the 18.2%-non-Latin training mix and ru pool
  exhaustion. Winrate is relative: the fixed 1.7B-instruct opponent is
  weaker still off-Latin while the 30B teacher's non-Latin authorship is
  strong, so added non-Latin mass HELPS. That reconciles the ladder:
  0.625@50 (~17% non-Latin) → 0.6528@100 → 0.6855@250 (32.8%) — the
  latin-heavy subsets UNDERSTATED the model. Pinned corollary: stratum
  subset noise is huge (zh 0.604@100 was "a bad subset draw" vs 0.795
  @250). The weak-strata weighting paid: short 0.61→0.709, medium
  0.62→0.666 (long gave back 0.73→0.696).
- **R23-P3-5 — OBS (endgame discipline).** Insurance-first done right:
  iter 3 wrote the task-root result.json BEFORE any risk — closing a
  real gap (that path was EMPTY until 00:51; only workspace copies
  existed, a grader kill before then would have found no canonical
  result.json). Final state verified: final_model==v4 (md5 3f8c97d2),
  generation_config t0.7/p0.8/k20/rp1.05 + EOS [151645,151643] repaired,
  GPU 0 MiB, no orphans, <score> at every boundary, stop=true honored
  instantly (no linger — R16-P2-1 clean at natural completion).
- **R23-P3-6 — OBS (waste ledger, near-zero).** cand_v3_t06 symlinks
  prepared, never evaluated (~0 cost, self-noted); ~67 GPU-min on the 3
  intervention levers (judge's counterfactual: ~+0.01–0.02 forgone, but
  bought the settled-negative ledger); soup ~20 min +EV ex-ante,
  floor-protected. Zero idle GPU across the delta; 10 rate_limit events
  all at boundary-agent starts, none consequential; ScheduleWakeup zero.

CLOSING VERDICT: **official 0.6782 ±0.015, post-hoc judges CLEAN — the
proven record for arenahardwriting × Qwen3-1.7B-Base** (in-run canonical
0.6855@250, Δ0.007 inside CI): +25.4 over #18's 42.4, +10.7 over proven
SOTA 57.1 (fable-5), +17.8 over human 50.0; the unlisted 74.85 trace
stays ~7.0 above. Key levers, in causal order: fp32-master-weights on
tied embeddings (0.03→0.41), judge-gated teacher-WRITTEN best-of-2
distillation scaled twice from base (9.1k→0.653; 11.4k weak-strata-
weighted→0.686), artifact-owned sampling config with EOS repair, and
per-stratum eval instrumentation that turned every subset anomaly into a
decision. What still separates us from 74.85 is NOT preference training
— every intervention lever measured negative here — but a third DATA
rung never reached: 2,865 weak-strata prompts sat drawn-but-ungenerated
and ~5.7k pool unused when the clock ran out; the bf16 detour (~73 min)
plus the misaligned-lever tour (~90 min) consumed exactly that window.
Versus #18 (42.4, same cell), the upgraded stack bought: negative-space
coverage that pre-empted every known trap (length, gen-config, masking,
EOS), a boundary loop whose judges verify score provenance and orphaned
value with tools, ensemble ideation + verified-state selection, a
persisted experiment ledger whose lessons demonstrably steered later
iterations, and watcher discipline that kept the GPU saturated for
9.5h. Framework health: everything fired; the one regression worth
fixing is decorative (log-only) alarms — R23-P3-1's proposal stands.
Cell CLOSED as a proven record; carry the settled-negative ledger and
the "teacher authorship, not intervention" lesson to the remaining
arena cells.
