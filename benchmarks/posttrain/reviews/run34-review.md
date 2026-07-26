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

## P2 (t+2.7h → t+6.3h)

Headline: **iter-1 banked 0.1667 (16.67% full-448) — still BELOW the 25%
random floor, and the trace pins why: parse rate, not knowledge, is the
ceiling.** An early exp checkpoint measured `parse=0.20, term=0.98,
acc_given_parseable=0.30` → 0.06 overall @50 — i.e. ~80% of completions
never emit a scorer-conformant `ANSWER: X` line, the exact
below-random-is-format-failure signature the GPQA watchlist named. The
2-epoch bank lifted it to 16.67 but the parse wall stands. This is the
hardest small-base format cell of the campaign.

- **R34-P2-1 — OBS (subset-bias lesson fully internalized).** The
  container's PRIOR_RUN_INSIGHTS @150-bias line is quoted operationally:
  "@150 stderr ≈ ±4pp; @448 ≈ ±2.2pp. @150 biased high vs full →
  full-448 pre-freeze gate" and "Mandated first action: full-448 bank
  eval." Iter-1 promoted on the full-448 number, not a subset — the
  #33 mistake pre-empted by the lesson written from #33 itself.
- **R34-P2-2 — OBS (iter-2 direction).** Full-FT of SmolLM3-3B-Base on
  curated DeepSeek-R1-0528 science-MCQ traces (nvidia/OpenScienceReasoning-2),
  byte-exact inspect+smollm.jinja render, completion-only loss, max_seq
  6144; the training-data format check confirmed anchored rows give a
  guaranteed `ANSWER: X<|im_end|>` tail (60 tokens) — directly attacking
  the parse wall. --limit 20 termination/parse gate before any promote.
- **R34-P2-3 — OBS (watcher discipline).** Condition-waiter (eval-finish
  OR training-start) + bounded dead-man alarm; genuine turn-ending waits
  ("I'll stop polling and genuinely wait"). Clean.
- No session-limit events; failover idle.
- State at cut: iter-2 training queued behind the bank eval; GPU-free
  window used to pre-write the candidate-prep + gate/confirm/promote
  helpers.
