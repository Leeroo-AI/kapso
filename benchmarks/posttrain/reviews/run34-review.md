# Run #34 review — gpqamain × SmolLM3-3B-Base (gpqamain-smollm3-3b-base-07260818)

Launched 2026-07-26 08:18Z us-east4-a on the rebuilt stack (carries the
@150-subset-bias PRIOR_RUN_INSIGHTS line and #32's shared-cache teacher
traces). Cell refs: base 4.9, proven band 29.0-30.6, human 33.3.

## P1 (t+0 → t+2.7h)

Headline: **format hazard confirmed at its most extreme yet — base SmolLM3
reads ~5.9% accuracy with a 20-30% parseable-ANSWER rate — and the plan
owns the whole delivery layer before any training.**

- Ensemble delivered 4/4 (both members 2/2 — no repeat of #32's fable-500);
  lens plan science-flavored two-family structure as before.
- **R34-P1-1 — OBS (selector discipline, 4th consecutive).** Rejected C2
  (6-8h monolithic two-stage FT that "banks its first score last" +
  fragile programmatic hard-negative MCQ construction); synthesized C4
  backbone + C3's recon rigor (empirical jinja render resolution,
  scorer-regex dump, eos hardening) with the teacher pipeline as terminal
  fallback. Time-to-first-bankable-score is now a standing selection axis.
- **R34-P1-2 — OBS (SmolLM3 eos owned).** Plan pins
  `eos_token_id=[128001,128012]` handling explicitly (base tokenizer's
  default lacks `<|im_end|>`); gate eval (--limit 20) checks termination
  rate + parseable rate + accuracy before any promotion — the
  parse-before-promote pattern institutionalized.
- Watcher discipline: notification-driven (gate eval queued behind
  training on a verified-free GPU); 8 wasted calls in 2.7h — mild
  repetitive "I'll stop polling" self-talk but nothing like #28's 188.
- State at cut: run-1 (2-epoch checkpoint) training completing ~11:00Z;
  eos patch + limit-20 gate to follow automatically.
