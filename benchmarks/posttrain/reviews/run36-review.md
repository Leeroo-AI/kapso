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
