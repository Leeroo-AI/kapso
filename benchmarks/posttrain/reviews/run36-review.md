# Run #36 review — aime2025 × gemma-3-4b-pt (aime2025-gemma-3-4b-pt-07262239)

**The shared-learning experiment.** Same base + cell + core stack as run #28
(which scored 0.0), the ONE intervention being `knowledge/aime2025.md`
seeded into the shared cache as an OPTIONAL offer (the other three AIME
cells' learnings + gemma's own run-28 lessons, reframed to a concrete bar).
Launched 2026-07-27 22:39Z us-east4-a on the failover stack. Question: does
offered cross-run knowledge lift gemma off 0.0?

## P1 (t+0 → t+2.5h) — UPTAKE CONFIRMED

Headline: **gemma READ the offer (twice) and adopted the sibling recipe
wholesale at iteration 1 — the exact levers run #28 discovered late or not
at all are front-loaded here from the seed.**

- **R36-P1-1 — the seed was consumed, not ignored.** The offer reached the
  prompt (shared_cache artifacts.json + the "an OFFER, not an instruction"
  brief). Within 30s of session start the agent listed
  `$KAPSO_SHARED_CACHE_DIR`, found `prior_run_learnings.md`, and `Read` it
  (22:49:03), then Read it AGAIN when grounding iteration 1 ("Let me ground
  myself in the shared-cache learnings", 22:51:53). The plan's rationale
  cites it directly: "the prior gemma-3-4b-pt run scored 0.0 and its
  post-mortem (prior_run_learnings.md)...".
- **R36-P1-2 — recipe adoption is comprehensive.** In-trace counts: MATH-500
  dev proxy (32), OpenR1 (21), conciseness anneal (14), 3 epochs (14),
  rep_penalty 1.1 (13), OpenMathReasoning (12), DeepScaleR (11), LIMO (6),
  corpus-sized-to-step-time (4). Every headline lesson from the doc is in
  the plan. The selected Core Idea: "Conciseness-first SFT: fine-tune
  gemma-3-4b-pt on ~4k length-filtered (≤6k tokens), verified R1 reasoning
  traces from OpenR1-Math-220k, re-registered to end with a bare `ANSWER: N`
  line" — the LIMO/conciseness recipe the siblings used, not the static
  approach run #28 flailed on.
- **R36-P1-3 — arch traps PRE-EMPTED (the biggest run-28 time sink).** The
  plan explicitly schedules "after every save, copy
  processor/preprocessor/tokenizer configs + generation_config.json into
  the checkpoint dir (vLLM dies without them — MEASURED trap)" and sizes
  the corpus to a 30-50-step throughput probe so 3 epochs fit in 60% of
  session time (writing the arithmetic into PLAN.md). Run #28 hit the
  processor-config vLLM crash live and got only 1 undertrained epoch;
  #36 designs both out from the start.
- **Direct contrast with run #28 (no seed):** #28 iter-1 OOM'd on fp32
  logits, discovered the multimodal freeze + processor-config traps
  reactively, trained 1 undertrained epoch, and had no dev-resolution
  proxy. #36 front-loads fused/chunked-CE awareness, the processor-config
  copy, guaranteed 3 epochs, and MATH-500 dev gating — all from the offer.
- State at cut: training healthy, step 51/675 (3 epochs), loss 0.57↓,
  rep_penalty 1.1 baked; base probed at ~0/30-1/30 (matches #28); watcher
  armed for training completion (~90 min). Zero session-limit events.

**Experiment status: the delivery→uptake half is a clean YES.** Whether it
converts to a SCORE lift (beat #28's 0.0, reach ≥1/30, chase the 3.3 proven
ceiling) is the open question — and on this cell the score is n=30-noisy
(one problem = 3.3pp), so recipe-quality is the more reliable readout than
the raw number. P2/P3 track the ladder and the official.

## P2 (t+2.5h → t+6h) — uptake converts to PROCESS quality, not (yet) score

Headline: **iteration 1 shipped 0/30 official (ties #28 and base) — but the
seed turned the failure into a FAST, precisely-diagnosed one, and iteration
2 is attacking the real wall from the start instead of discovering it.**

- **R36-P2-1 — iter-1 official 0/30, seed-grade forensics.** Verdict (Node
  0, two pooled official reads): "0/30 … ties base and prior gemma — no
  improvement." But the diagnosis is exactly what the doc taught: model has
  ~29% MATH-500 capability (rp 1.1 confirmed optimal), yet **30-73% of AIME
  generations truncate at the token cap → guaranteed zeros**, and the corpus
  came out 78% olympiad proof-style (integer-filter rejected 63% of OpenR1,
  skewing away from AIME register). Same two defects run #28 found — reached
  in one clean iteration here, not several.
- **R36-P2-2 — decode lever validated on the dev proxy.** AIME-24 dev:
  greedy rp1.1 = 1/30 (winner, least truncation); rp1.2 = 0/30 ("breaks
  generation — measured"); T0.6 rambles to 93% truncation. Exactly the
  seed's "rp 1.1 sweet spot, 1.2 too much" lever, independently re-measured.
  Greedy rp1.1 baked into generation_config.
- **R36-P2-3 — n=30 discipline held.** All decode/checkpoint decisions made
  on MATH-100 / AIME-24 dev proxies, AIME-25 kept as pure held-out ("0/30
  vs 1/30 is noise" — the doc's own line, followed). No promoting on n=30.
- **R36-P2-4 — iter-2 attacks the wall directly.** Continue-FT the banked
  exp1 checkpoint (Stage-A "paid for", ~29% MATH) with a Stage-B corpus
  fixing both measured defects — ~2k verified decontaminated AIME-register
  traces + budget-forcing against truncation. The RIGHT next move,
  front-loaded from iter-1's evidence; running at epoch ~1.4, ~05:40
  completion. Zero session-limit events / swaps.

**Experiment read so far:** the shared-learning offer fixed the PROCESS
decisively — no OOM/arch-discovery detour (run #28's iter-1 sink), dev-gated
from the start, decode already optimal, right problem (truncation + corpus
register) identified in iteration 1 instead of iteration 2-3. What it has
NOT yet done is move the OFFICIAL score off 0/30: the ~29% MATH capability
does not transfer to competition-hard AIME for a 4B, and truncation economics
cap even the attempts that reason correctly. This mirrors the arena-gemma
finding — recipe/knowledge transfers cleanly; the residual gap is a
capability/data wall, not a process deficit. iter-2's anti-truncation Stage-B
is the last real shot at ≥1/30; P3 has the official.

## P3 (close-out: RUN_DONE 08:19Z → rescore) — EXPERIMENT VERDICT

**FINAL: official 0.0 (0/30) via rescore (clean full-30, 0 retries) · both
judges clean · 3 iterations, ALL 0/30 · final_model = exp1 (not demoted on a
tie).** Identical official score to run #28 (no seed). The shared-learning
intervention did NOT move the number.

But the experiment answered its question cleanly, and the answer is
informative:

- **Uptake: total.** Gemma Read the offered doc twice, built iteration 1 on
  it, and carried the recipe through all 3 iterations (rep_penalty 1.1,
  MATH-500/AIME-24 dev gating, conciseness, corpus-sizing, arch-trap
  pre-emption). Every seeded lever was adopted; nothing was ignored.
- **Process: transformed vs run #28.** #28 spent early iterations
  discovering the multimodal/processor-config traps (live OOM crash) and
  got 1 undertrained epoch before finding the corpus/truncation defects.
  #36 pre-empted all of that from the seed, dev-gated from the start, and
  spent iters 2-3 attacking the RIGHT wall — anti-truncation Stage-B cut
  truncation from ~73% to ≤30%. Same destination, reached faster and
  cleaner, budget spent on the real problem.
- **Score: unchanged — the wall is CAPABILITY, not knowledge.** The model
  measured ~29% on MATH-500 (real capability, rp 1.1 optimal) and solved
  exactly 1/30 on the AIME-24 dev proxy, but 0/30 on AIME-25 across every
  checkpoint and both decode configs. Reducing truncation didn't help
  because the underlying competition-AIME capability of a 4B simply isn't
  there — MATH-500 (~29%) does not transfer to AIME difficulty. A knowledge
  doc transfers recipe and process; it cannot transfer capability.
- **R36-P3-1 — promotion discipline exemplary.** iter-3's 0/30 fell below
  the ≥1/30 gate → NOT promoted; incumbent exp1 kept ("never demote on a
  tie"); best_score.log unchanged; final_model integrity verified
  (Gemma3ForConditionalGeneration, vocab 262208, 13 serving files);
  GPU 0 MiB, all jobs consumed, clean close.
- **R36-P3-2 — runner.py:344 repr bloat, worst instance yet.** solve_out.txt
  is ~100 MB (one end-of-run SearchNode dump line ≈ 99 MB). The documented
  fix is overdue; forensics now require line-truncated greps to avoid OOM.
- **R36-P3-3 — serving failure 8/8, gemma rescore clean.** metrics.json
  MISSING post-solve (model-agnostic bug); fresh-VM rescore served gemma
  first-try (only SmolLM3's arch crashes the rescore container). Official
  0.0 confirmed on a clean 30/30 read.
- Zero session-limit events / swaps across the full run (main token had
  headroom; failover idle).

## Verdict on the cross-run-knowledge hypothesis

This is the controlled result the arena-gemma postmortem predicted:
**offered cross-run knowledge transfers completely and improves process
quality decisively, but cannot overcome a capability/data wall.** On a cell
with real headroom (the sibling AIME cells, or GPQA) the same mechanism
would plausibly convert to score — here it hit gemma-AIME, the one cell
where the ceiling is capability, not recipe. The seeding mechanism itself is
validated end-to-end (delivery + uptake + faithful execution); the null
score result is a property of the cell, not the intervention. Future
gemma-AIME attempts: do NOT expect a recipe/knowledge doc to move 0/30 — the
bar is a bigger/stronger base or a fundamentally different capability source.
