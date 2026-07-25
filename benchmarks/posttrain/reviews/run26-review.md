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
