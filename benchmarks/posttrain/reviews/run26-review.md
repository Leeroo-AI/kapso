# Run #26 review — aime2025 × Qwen3-4B-Base (us-east4-a, launched 2026-07-25 02:01Z)

Richest AIME cell: base 3.3 | best proven 23.3 opus-4.8-max | human 53.3;
weight .2265. Same upgraded stack as run #25 (negative-space coverage,
lens planner, max reasoning).

## P1 (t+0 → ~t+85min)

Headline: **run #25's entire discovery stack was independently re-derived
in 15 min of ideation wall-clock (~$2.2 tracked) — and deepened.** Beyond
the eos [151645,151643] lever, scorer-source read, and n=1/maj@k-illegal
finding (all re-found with file:line cites), #26 added: the `endswith`
false-positive quirk ("1204" matches "204", `_common.py:83`),
qwen3.jinja lacking `{% generation %}` markers → TRL `assistant_only_loss`
unusable → hand-masking required, chat-template injection at
`evaluate.py:123-137`, and vLLM-serves-temp-1.0-without-genconfig (which
explains the garbage baseline). Re-derivation cost is real but negligible
vs a 10h budget; cross-run knowledge would mostly buy polish here.

Boot clean (~8 min provision, H100 + cuda-tensor write pass); zero
us-east4 anomalies (OpenR1 94k split downloaded+filtered in ~6 min). Lens
plan (fable, 150s, web): L1 curated public long-CoT SFT / L2
measurement-attack + teacher self-distillation — orthogonal, math-native,
and the lens text itself already carried scorer/eos/16k awareness.
Members 2/2 + 2/2, dropped 0. All four candidates shipped 5 coverage
families with substantive Not-measured closers, e.g. metric mechanics:
"accuracy + stderr over 30 binary items — MEASURED (`aime2025.py:65`);
±~8 pts binomial noise… Not measured: run-to-run variance at temp 0.6
(needs repeated full evals — deferred; greedy A/B partially bounds it)."
ScheduleWakeup 0; layered alarms everywhere (fg sleep N−5/−10 under a bg
alarm), zero dud fires (no R23-P3-1 recurrence).

- **R26-P1-1 — OBS (positive).** See headline: full re-derivation plus
  four new mechanics findings, all pre-training.
- **R26-P1-2 — OBS.** Gate design weaker than #25's 90-problem n=4
  held-out: promotions ride the official 30-set every time, <7-pt deltas
  = ties broken on an AIME-2024 proxy. 7 pts < 1σ (~8) — the ratchet and
  the epoch-1-vs-2 pick can ride noise. P2 watch.
- **R26-P1-3 — OBS (echo R25-P1-3).** Codex member again never read the
  evaluator (ASSUMED harness claims); selector again compensated — read
  scorer/template/evaluate.py itself, caught C1's ~2×-optimistic epoch
  estimate and C2's unproven weight-8 answer-span loss. Member tooling
  asymmetry now 2-for-2 on AIME.
- **R26-P1-4 — OBS.** Minor churn: probe-1 tok/s counter read 0
  (collator counter lived in a dataloader worker) → probe-2 relaunch,
  ~3.5 min; four filler micro-turns 03:13–03:15 while a bounded poll task
  finished. Self-recovered.
- **R26-P1-5 — OBS (positive).** Baseline 0.000 (t+29min) accepted only
  after reading sample completions (garbage confirmed) — R25-P1-2
  behavior remediated. 0.0 vs leaderboard 3.3 = same floor within
  one-problem noise. View-only eval profiling done without
  self-censoring (relaxed-rules semantics correct).

LENS PLAN: L1 curated-corpora SFT excellence / L2 harness-fidelity +
verified self-distillation (+GRPO top-up only if hours remain).

SELECTED PLAN: SFT e1 = full-FT bf16 on 15,000 OpenR1-Math-220k verified
traces (≤11.5k tok, ≤2/problem, 8-gram+Jaccard deconned; ~83M tok at
measured 8.1k tok/s → ~2.9h), lr 1e-5 cosine, bs 1×16, maxlen 12288,
hand-masked assistant span, byte-exact jinja + `ANSWER: N`. Ship
generation_config {temp 0.6/top_p 0.95/top_k 20, eos [151645,151643]};
full-30 eval, promote ≥ baseline (0.0) ~t+4.5–5h → e2 resume next session
(precedent: e2 22.1% > e1 > e3) → free greedy-vs-0.6 A/B. Hooks:
AM-Thinking top-up, QwQ-32B-AWQ rejection-sampling teacher, Light-R1
stage-2, LoRA-GRPO.

Verdict: **continue** — SFT healthy at cut (loss 0.62→0.42 at epoch
0.17, 11s/step, ckpt-100, ETA 05:36Z, ~1h session slack), sizing
measured not assumed, trajectory aimed squarely at the 22–23 proven
zone. P2 watches: e1 gate result, live curl genconfig verification
actually run at package time, tie-discipline held (R26-P1-2), e2 branch.

## P2 (03:26 → 06:48Z)

Headline: **e1 landed at 13.3%, then a pure decode-time sweep took the
same weights to 26.7% — statistically at the proven-best 23.3 by
t+4.7h, with epoch 2 still unspent.** Training finished 05:19Z (9,209s,
final loss 0.433, epoch 1.0, ckpt-100…938 with keep-2 rotation, zero
errors). Genconfig {0.6/0.95/20, eos [151645,151643], max_new 16000}
written and honored (vLLM log: "PyTorch-native implementation of top-p
& top-k sampling" — sampler params live; no separate live-curl, but
this is positive proof). The peeked 0.067/0.200 are NOT new artifacts:
0.067 = rp 1.15 over-penalization collapse; 0.200 = rerun of the banked
greedy+rp1.1 config exposing vLLM batched-greedy non-determinism.

- **R26-P2-1 — OBS (positive).** e1+t0.6 = 13.33% (4/30) at 05:37Z,
  promoted 05:38Z ("PROMOTED … score 13.333"). Forensics within 2 min:
  `has_think=0`, median 47k chars, 26/30 truncated at the 16k cap —
  id=0 tail is "Therefore, d divides 7\*(8 - d)." looped to the cap.
  Root cause (repetition-loop truncation) read from data, lever
  (`repetition_penalty`) identified before the greedy result returned.
- **R26-P2-2 — OBS (positive).** Decode ladder: greedy 16.67 (05:53Z)
  → greedy+rp1.1 **26.67** (06:06Z, banked) → t0.6+rp1.1 26.67 (06:16Z)
  → rp1.15 **6.67** (06:30Z; "over-penalizing breaks legitimate math
  tokens" — sharp peak at 1.1) → deliverable-path confirm **20.0**
  (06:41Z; "vLLM's concurrent … 'greedy' decoding is not run-to-run
  deterministic") → final_model switched to Qwen3-recommended
  t0.6+rp1.1, confirmation in flight at cut. +10–13 pts for zero
  training compute.
- **R26-P2-3 — CONCERN (R26-P1-2 materialized).** Every rung is a
  single official-30 run: the greedy-over-t0.6 pick (5 vs 4 problems)
  was later self-refuted; the banked 26.67 regressed to 20.0 on an
  identical config; best_score.log/result.json carry 26.667 while
  honest E[score] ≈ 23 ± 8. Credit: the deliverable-path confirm run is
  exactly what caught the noise, and the final pick was made on
  principle (Qwen3 guidance) not the noisy argmax. Residual: the
  shipped config rests on one 26.67 observation — a 2–3-run mean
  (~25 min) would harden it. P3 watch.
- **R26-P2-4 — OBS (framework, clean).** ScheduleWakeup 0; 20 layered
  alarm-poll cycles / 39 Bash calls in 3.37h (vs R25-P2-3's 83 churn);
  zero dud fires, no freezes (all quiet gaps = intended sleeps). One
  TaskOutput InputValidationError 03:25:55 (deferred schema,
  timeout>600000) self-recovered in 4s; one rate_limit_event 05:10:31,
  no stall. promote.py's 32GB checkpoint-subdir copy into final_model
  spotted and cleaned in the very next call.
- **R26-P2-5 — OBS.** No session-boundary events in the delta — at
  06:17Z agent noted "1h8m left" (implementation cap → boundary
  ~07:25Z; difficulties tag / feedback judge / R15-P2-1 land in P3).
  e2 not launched; ~5.2h solve budget remained at cut.

LADDER: base 0.0 (02:30Z) → e1+t0.6 13.3 (05:37Z, promoted) → +greedy
16.7 (05:53Z) → +rp1.1 26.7 (06:06Z, banked; t0.6+rp1.1 ties 26.7
06:16Z) → rp1.15 6.7 (06:30Z, collapse) → greedy+rp1.1 rerun 20.0
(06:41Z, vLLM non-determinism) → ship t0.6+rp1.1, confirm pending —
one artifact (e1), all movement decode-side.

SOTA OUTLOOK: the current artifact already straddles the proven 23.3
(runs of 26.7/26.7/20.0 across rp1.1 configs, mean ≈ 24) at 48% of
budget, and the two biggest known levers — epoch 2 (precedent 22.1% >
e1) and the now-calibrated rp1.1 — compose. 23.3+ is genuinely in
reach and high-20s plausible if e2 replicates precedent; the threat is
symmetric σ≈8 noise on the 30-set plus configs chosen on single runs.

Verdict: **continue** — root-cause-driven decode win banked, promotion
discipline held (nothing unbanked ever led), hygiene spotless; P3
watches: e2 launch/gate, final-confirm result, session-1 boundary
mechanics (~07:25Z), replication of the shipped config.
