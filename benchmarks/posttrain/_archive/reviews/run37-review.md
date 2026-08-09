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

## P2 (t+2.5h → t+6.6h)

Headline: **iteration 1 closed 0/30 with the first real capability ladder
this cell has ever produced (base 0 → KD 12.3 → RFT-1 19.6% MATH-L4/5,
AIME-24 1/30 by genuine reasoning); iteration 2 pivoted, on measured
grounds, to long-CoT scaling — and the logit-KD lever remains untried.**

- **R37-P2-1 — iter-1 close-out quality.** Four official reads, all 0/30,
  all milestone-grade (no fishing); feedback audit verified the eval
  untampered and "all candidates genuinely read 0/30"; RFT-2 (2× corpus)
  measured 17.4% vs RFT-1's 19.6% → dev-gate refused promotion — RFT
  saturates after one round on this band (useful negative). Teacher corpus
  + pipeline registered to shared cache.
- **R37-P2-2 — R37-P1-3 RESOLVED: the KL/logit-KD mechanism was NEVER
  implemented.** Iter-1's "KD" was answer-verified teacher-trace SFT
  (sequence-level); the boundary note says soft-target KD was "correctly
  deferred to iteration 2's fresh session" — but iter-2's selector then
  chose long-CoT SFT instead, so the playbook's #1 mechanism is still
  virgin after two iterations. Campaign conclusion must NOT claim logit-KD
  was falsified here; what iter-1 tested is teacher-trace SFT + RFT.
- **R37-P2-3 — iter-2's causal pivot is well-grounded but notably close to
  run #36's ground.** Thesis: iter-1 proved format solved (26/30 clean
  ANSWER lines) and capability the sole bottleneck; every iter-1 target was
  CONCISE (teacher p50=650 tok) while the eval grants 16k — so reasoning
  LENGTH is the untapped axis → long R1-style CoT SFT (OpenR1 verified,
  ≤8k gemma tokens, exact ANSWER register, 3 epochs guaranteed by sizing
  arithmetic). Distinction from #36 (which also trained R1 traces and got
  0/30): #36 length-FILTERED to ≤6k and optimized conciseness; this bets
  the opposite end. The selector's rejection of the schema candidate was
  causally clean ("format solved → vote engineering doesn't attack the
  bottleneck").
- **R37-P2-4 — discipline still campaign-best**: sizing arithmetic honored
  (471 steps / 3 epochs / 94.5 min, loss 0.547), waits bounded and
  productive (gate scripts syntax-checked, RFT replay pre-fixed during
  training), zero wasted calls, zero limit events at t+6.6h.
- State at cut: dev gate (2 decode points @16k) firing on the long-CoT
  model; ~3h to session close; RFT top-up and final official read queued
  behind the gate.

## P3 (close-out: RUN_DONE 00:07Z → rescore 01:2xZ) — FINAL VERDICT

**FINAL: official 0.0 via rescore (clean 30/30, 0 retries) · both judges
clean · 3 iterations (0.0 / 0.0 / soup promoted, 0/30).** Shipped
final_model = weight-space soup of the two training lineages (byte-verified,
temp 0.8/rp 1.05) — dev-best of the campaign on this cell.

What run #37 established:
- **Capability moved for the first time in ANY gemma-AIME attempt**: base 0
  → 19.6% MATH-L4/5; AIME-24 = 1/30 by genuine reasoning (no official
  attempt, ours or the leaderboard's, ever produced that). The playbook's
  mechanisms work as capability levers.
- **The official cell did not convert**: 7 milestone reads + the rescore,
  all 0/30. ~3% concentrated capability doesn't intersect AIME-2025's
  specific 30. With the record now known to be an answer-only selection
  artifact, the truthful leaderboard statement is: NO artifact has ever
  solved an AIME-2025 problem on this cell by reasoning.
- **Falsified this run**: teacher-trace SFT (sequence-level), RFT (works
  once, saturates), long-CoT SFT, checkpoint soup — as OFFICIAL-score
  levers at 4B. **Still unfalsified: true logit/soft-target KD** (deferred
  in both iterations; never implemented).
- **R37-P3-1 — codex ideation 401 in iter-3** (ChatGPT auth.json staled
  mid-run; graceful degradation to prior candidates). The codex-auth
  analogue of the OAuth-expiry problem — flag for an auth.json refresh
  before long runs.
- **R37-P3-2 — official-read count 7**: highest of the campaign; every
  read was a distinct banked artifact milestone (no repeated draws of one
  artifact), but future runs at a noise floor should cap explicitly.
- Zero session-limit events / zero swaps across ~10h (fresh token).
  Serving bug 9/9; rescore clean.

## Experiment series verdict (runs 28 → 36 → 37)

Cold (0.0) → recipe-seeded (0.0, process transformed) → playbook-seeded
(0.0, capability transformed). Shared learning demonstrably improves each
layer it can reach — plans, process, now measurable capability — and the
official score stays pinned at the base model's ceiling. Cell disposition:
CLOSED for 10h/1-GPU attempts unless someone implements the one untried
mechanism (logit-KD) or the rules/base change. The honest campaign claim:
our 0.0-by-reasoning ties every reasoning attempt ever made, and our
artifact is the only one with demonstrated AIME-class capability.
