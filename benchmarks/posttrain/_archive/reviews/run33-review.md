# Run #33 review — gpqamain × Qwen3-1.7B-Base (gpqamain-qwen3-1-7b-base-07252105)

Failover-stack debut alongside run #32 (3-token oauth file verified on-VM).
Launched 2026-07-25 21:05Z us-east4-a. Smallest model on the 2nd-heaviest
benchmark; same format-parse watchlist as #32.

## P1 (t+0 → t+2.7h)

Headline: **iteration 1 was killed 48 min in by an Anthropic `API Error:
529 Overloaded` — and the boundary machinery turned it into a 4-minute
blip: difficulties captured, feedback judge graded it honestly, iteration 2
relaunched the same selected plan with added resilience discipline.**

- **R33-P1-1 — OBS (external-kill recovery, cleanest specimen yet).** The
  exp_0 CLI died at ~21:55 (529 storm; same Anthropic instability window
  as #32's ideation-member 500). Feedback judge (238s, $0.82): "Not
  gradable on the merits — only 48 of 600 minutes were used before an
  external kill, and the run died on-plan"; verdict carried the full
  restart playbook (baseline eval command, stage order, atomic-swap
  priority) plus new resilience rules: run training under `nohup` + PID
  file so an agent death never kills the GPU job, checkpoint every ~15
  min, commit scripts/PLAN to disk immediately. Node banked score=None,
  evaluation_valid=True, iteration 2 live at 21:59:16 (budget 8.6%).
  This is R17-P3-1 fix A+C behaving exactly as designed under a real
  provider outage.
- **R33-P1-2 — OBS (ensemble delivered 2/2 + 2/2 here).** Unlike #32,
  both members delivered; lens plan (fable-5, web-enabled) produced the
  same two-family structure as the AIME plans, science-flavored: L1
  published long-CoT science distillation (OpenScienceReasoning-2,
  OpenThoughts3-science, MegaScience), L2 local-teacher verified
  generation + metric-mechanics ownership.
- **R33-P1-3 — OBS (selected plan).** Staged full-parameter assistant-
  loss-only SFT on contamination-filtered 4-choice science reasoning,
  eval-register rendering (single `ANSWER: $LETTER` line), exclusion-only
  decontamination vs all GPQA configs.
- Iteration 2 conduct so far: teacher-trace curation at scale — pass 1
  accepted 12,602/13,815 (91.2%), retry pass on the 1,213 failures, ~15.5k
  total traces with part 1. Invariants block carried verbatim across the
  boundary (base-model-only, no GPQA in training, no evaluate.py edits,
  'qwen' in artifact paths for jinja routing).
- Zero session-limit events; failover not yet exercised.

## P2 (t+2.7h → t+6.5h)

Headline: **iteration 2 (the relaunched plan) shipped a real result:
0.2867 (43/150) vs base 0.1867 — +10pp, ABOVE the 25% random floor, within
~2pts of the proven ~30.5 band.** Boundary audit md5-verified final_model
== champion `qwen3_distill_c` and confirmed 0.2867 is the max across every
candidate metric file (a=0.16/0.12, b=…). Verdict CONTINUE; iteration 3
live at ~03:24.

- **R33-P2-1 — OBS (measured-claims contract catches a bad assumption
  live).** Iter-3's inherited plan ASSUMED OpenScienceReasoning-2 is
  "exactly four A-D choices"; recon measured rows carry options A-J —
  only **30.2% are clean 4-choice** with consistent answers. Plan updated
  before any training spend. This is the Coverage/MEASURED-vs-ASSUMED
  machinery (816500d1) paying for itself at the exact failure point it
  was designed for.
- **R33-P2-2 — OBS (iter-3 shape).** Decode-fix first (champion copy +
  generation_config temp/rp verified by live reads), paired larger-n
  confirm gate before any re-bank (AIME pooled-gate lesson generalized),
  consolidate by T−45.
- 1.7B currently OUTSCORES the 4B run (0.2867 vs 0.16 at n=150) — the
  smaller model's run banked the format+distill recipe cleanly while 4B
  burned iter-1 on subset-noise promotions.

## P3 (close-out: RUN_DONE 06:59Z → rescore 07:5xZ)

**FINAL: official 22.32 ±2.0 via rescore · both judges clean · 4 nodes
(None → 0.2867 → 0.2767 → None).** +8.2 over base 14.1; well under the
29.5/29.5/29.4 proven band — but the shipped artifact was defended
flawlessly, and the gap decomposes into two measurable causes.

- **R33-P3-1 — the @150 subset bias is the headline gap.** In-run
  champion read 28.67 @150; official full-448 = 22.32. Unlike #32, this
  run NEVER ran a full-448 confirm — every eval was --limit 150 — so the
  bank carried the subset's optimism to the end. Cross-run evidence now
  2/2 (#32: 37.33@150 → 32.14 official). Lesson institutionalized in
  PRIOR_RUN_INSIGHTS (full-set confirm before freeze). At 22.3 vs a
  hypothetical honest ~23-25 bank the ARTIFACT was still the right ship;
  what was lost is decision quality (iter-3's 0.2767 "loss" vs 0.2867
  was inside the subset's own bias band — a full-448 A/B might have
  re-banked).
- **R33-P3-2 — endgame champion defense, best specimen yet.** The final
  gambit candidate was caught catastrophically broken MID-EVAL via
  partial-log preview at 80/150 (accuracy 0.0125, parse 0.29, 74%
  truncated, p50 completion 54,556 chars of repetition); eval aborted,
  champion verified (md5 e1f06d96 + fresh-process load + greedy
  genconfig intact), best_score.log floor untouched, GPU swept to 0 MiB,
  closed with 22 min spare. The negative result was committed to docs.
- **R33-P3-3 — 529-kill recovery (P1) held up end-to-end**: the
  externally-killed iter-1 cost 48 min + $0.82 total; its replacement
  banked the champion. Boundary machinery 2-for-2 under provider
  outages.
- Watcher discipline: layered completion + startup-check + dead-man
  watchers on the last eval (bqf9tk4ji/ba4y27rzi/btz9wesz1), zero dud
  fires, notification-driven waits throughout.
- Same infra notes as #32: final-eval serving failure (6/6), first
  rescore attempt died on the gated-gpqa auth gap (fixed 88fe3914),
  zero session-limit events, failover unexercised.

