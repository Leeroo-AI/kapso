# Run #37 review — aime2025 × gemma-3-4b-pt (aime2025-gemma-3-4b-pt-07271403)

**Shared-learning round 2: the engineer playbook.** Same cell/stack as runs
#28 (cold, 0.0) and #36 (recipe-seeded, 0.0); the intervention this time is
`knowledge/aime2025.md` carrying "Gemma playbook v2" — capability-INJECTING
ideas no prior run tried (same-vocab soft-target KD from gemma-3-27b-it,
band-targeted RFT, in-generation self-consistency, answer-prior fallback,
frontier-only GRPO) plus the record-provenance study (the proven 3.3 = an
answer-only guesser selected on the hidden set; no reasoning attempt has
ever beaten 0/30). Launched 2026-07-27 14:03Z us-east4-a; fresh main token
(7% weekly) + alpha recovery chain verified on-VM.

## P1 (t+0 → t+2.5h) — the playbook's #1 and #2 levers, selected and executing

Headline: **the selector picked the playbook's top-ranked lever and the run
is executing the ladder in order — teacher KD corpus → bootstrap SFT (bank)
→ pass@k band probe → RFT harvest — with the first real capability lift on
this cell: MATH-L4/5 12.3% vs base 0.0.**

- **R37-P1-1 — lever selection (the experiment's question): YES.** Selector
  verified the doc at source ("prior_run_learnings.md, confirmed on disk:
  text-imitation SFT alone is falsified on this cell — run #36: 0/30 across
  3 iterations") and synthesized C3's staged spine with C1's
  answer-verified 27b-it teacher + cached top-20 logit KD — the playbook's
  #1 idea. It also caught C1's register flaw itself (boxed-centric vs the
  harness's `ANSWER: N`) and fixed it in the synthesis.
- **R37-P1-2 — execution so far, per stage:** 27b-it served locally,
  **1,542 answer-verified teacher traces** (keep-rate 0.67; ~51
  prompts/min); assistant-masking pre-validated during generation (0
  prefix mismatches); stage-1 trained; dev = **MATH-L4/5 12.3% (17/138) vs
  base 0.0** (hard-slice; not comparable to #36's full-MATH 29%); banked as
  insurance; ONE official read (0/30 at t+2h — a milestone read, not
  selection); **pass@k band probe found 71/256 problems in the RFT sweet
  spot (solves 1-7 of 8)**, 134 correct chains in chunk 1, ~500+ expected
  → RFT round 1 declared worthwhile and harvesting (K=8, shortest-correct).
- **R37-P1-3 — OPEN (verify at P2): was the KL/logit term actually
  implemented in stage-1 training, or did "KD" reduce to answer-verified
  teacher-trace SFT (sequence-level only)?** The plan specifies cached
  top-20 logits + KL; the trace so far shows the corpus + training but no
  explicit kl_loss/teacher_logits implementation lines. The dev lift is
  real either way, but which mechanism produced it matters for the
  campaign's conclusions.
- **R37-P1-4 — discipline:** zero wasted calls in 2.5h (campaign best);
  bounded waits; PID-targeted GPU cleanup between stages; teacher traces
  registered to shared cache for reuse; promote.py scratch-path collision
  self-caught and fixed in <1 min; AIME-25 used exactly once, as a
  milestone read on the banked artifact.
- Zero session-limit events (fresh token); no stalls.

Contrast at the same clock: #36 was building a text-imitation corpus (the
falsified recipe); #37 has a banked capability-lifted model and a
measured RFT frontier. Watch for P2: RFT round-1 outcome on the band,
whether the schema/answer-prior stages fire, the KL question above, and
official-read count staying disciplined.
