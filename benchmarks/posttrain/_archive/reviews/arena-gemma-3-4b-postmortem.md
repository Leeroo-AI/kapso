# Arena-Hard Writing × gemma-3-4b-pt — why 37.3, not 58.7 (or 94.8)

Comparative trace study, 2026-07-26. Sources: our run #24
(`arenahardwriting-gemma-3-4b-pt-07241547`, official 37.28 ±1.5,
reviews/run24-review.md) and BOTH official opus-4.8-max traces for this
cell from `aisa-group/PostTrainBench-Trajectories`
(`claude_non_api_max_claude-opus-4-8_10h_run{1,2}/arenahardwriting_google_
gemma-3-4b-pt_*`). Human row = google/gemma-3-4b-it (the official instruct
model, scored by the same harness).

## 0. Correct the reference frame first

The leaderboard's "opus-4.8-max 47.4" is a 2-run MEAN hiding a 58.74 and a
36.04 (metrics.json of run1/run2). So:

| Artifact | Score | Note |
|---|---:|---|
| opus-4.8-max run1 ("max1") | **58.74** | the real target for this cell |
| ours (run #24) | **37.28** | |
| opus-4.8-max run2 ("max2") | 36.04 | we effectively tie the weak twin |
| human (gemma-3-4b-it) | 94.8 | metric ceiling, see §4 |

Same leaderboard-mean pathology as every other arena cell (1.7B: 74.9
inside 45.0±42; SmolLM3: 74.0 inside 37.2±52). We are not "far behind
opus-4.8-max" as a distribution — we sit exactly at its weak run; we are
21.5 pts behind its strong run. The strong run is what to learn from.

## 1. What max1 (58.7) did — ladder and recipe

Timeline (8h38 total, single session, no boundary):
00:30 start → 02:04 v1 static mix (no_robots + Gryphe) **7.9** → 04:3x v2
(reweighted static) **17.4→18.9** → 06:27 v3 = +Qwen3-8B distillation
**40.2** → 08:19 v4 (distill scaled to 12.8k kept + Magpie/Gryphe/
OpenHermes creative slices, preamble-stripped, repetition-filtered, 19.3k
total, 2 epochs) **54.3** → 09:04 full-250 **61.2 in-run** → official
58.74. Four data versions, each gated by a cheap limit-32 eval (18 of
them), full-250 only at the end.

Recipe pillars, in causal order of the climbs:

1. **The 18.9→40.2 jump IS teacher distillation** (same structural jump as
   our 10.9→19.1→35.5, but theirs landed 2 iterations earlier and higher).
   Teacher = **Qwen3-8B, text-only, already in the official HF cache** —
   served fast, no multimodal OOM, no enforce_eager detour. Cross-family/
   cross-tokenizer distillation at the TEXT level worked fine; same-family
   was never needed.
2. **Judge-loss forensics as a standing loop.** They built
   `analyze_judgments.py` mid-run: re-runs eval with --store-outputs, then
   prints "sample LOSSES with judge reasoning" excerpts. Diagnosis from
   actual judge text: **looping/repetition drives decisive losses** →
   targeted fix.
3. **Decode ownership to the last knob**: shipped generation_config =
   eos [1,106], **temp 0.7, top_p 0.9, top_k 64, repetition_penalty 1.1**
   (A/B'd 1.15 mid-run on the diagnosed looping). Preamble-stripping +
   repetition-filtering applied to the training data too.
4. **Volume + creative diversity**: 19,296 final examples (12.8k distill +
   Gryphe storytelling + Magpie + OpenHermes creative slices), zh-heavy
   multilingual coverage throughout (569 zh mentions in-trace).

Cost of that trace: $154 nominal (opus-4-8, 471k output tokens).

## 2. Ours (37.3) vs max1 (58.7) — the gap decomposed

Ours: 15:47 start → 17:41 static mix **10.9** → +gemma-3-27b-it distill
**19.1** → length-cap **20.9** (t+4.6h, session boundary) → iter-2 scaled
distill (11,119 ex: 6,741 distill + 4,378 no_robots, 2ep) → full-250
**35.5** → official **37.28**.

Ranked by estimated contribution:

- **(a) Repetition penalty left on the table — our own explicit decision.**
  We shipped `temperature 0.9, top_p 0.95, repetition_penalty 1.0`, with
  the coverage ledger literally recording: "Not measured:
  repetition_penalty interaction with legitimate refrains — default 1.0
  shipped precisely to avoid penalizing choruses; revisited only if
  degeneration appears." Degeneration DID appear (our own P3: the
  incumbent "rambles toward the 16384-token cap"; the concise rubric
  punishes it; MAX_REPETITIONS=5 truncation exists in the harness) — but
  no eval ever A/B'd rp. max1 diagnosed looping from judge text and
  shipped rp 1.1. Campaign cross-evidence: the same lever was worth
  +9-13 pts decode-only in our AIME #26 and GPQA #32 runs. Estimated
  cost here: high single digits to low double digits.
- **(b) No judge-loss reading loop.** We read judge MECHANICS at recon
  (rubric, 3× decisive weights, baseline identity — better than max1's
  recon, in fact) but never read the judge's per-question loss
  explanations. That loop is rule 5 of our own best-traces skeleton and
  it's what turns "score is low" into "losses say looping/too-short" —
  max1's rp fix and completeness filters both came from it. Ours found
  the length-cap fix by intuition instead, one lever, later.
- **(c) Teacher throughput → data volume.** gemma-3-27b-it (multimodal,
  52GB) OOM'd → enforce_eager → 86 prompts/min → 56-min first batch;
  iter-2 raised concurrency to ~120/min after measuring. Net distill:
  6,741 examples. max1's Qwen3-8B (text-only, cached) yielded 12.8k kept
  distill + richer creative mixes = 19.3k total, 2 epochs over it. Their
  cache advantage is real but small (52GB pulled in ~1 min on our VM) —
  the actual tax was serving a multimodal 27B for generation. Same-family
  turned out to be an over-cautious prior: text-level SFT transfers
  cross-tokenizer.
- **(d) Iteration count under the same clock.** max1: 4 data versions,
  18 cheap limit-32 gates, zero session boundary. Ours: effectively 2.5
  data versions, limit-50 gates (which ALSO deceived us — the seed-42
  first-50 is unrepresentatively hard, difficulty #5), one session
  boundary (~30-40 min amortized), plus ~1h of iter-1 spent on the
  static-corpora + DPO-prep detour before self-correcting to the proven
  skeleton (R24-P1-1). The 89.6 Qwen3-4B recipe (Qwen3-30B-A3B teacher,
  BoN) existed in OUR OWN campaign five days earlier but each run starts
  from an empty store — the cross-run knowledge gap, again
  (docs/research/cross-run-knowledge-design.md is the standing fix).

What we did BETTER than max1: recon depth (byte-parity template assert,
eval_profile.md, language mix measured 73/14/10 and matched in the built
set 76/12/10), integrity discipline (vision-freeze verified in every
artifact, 47GB promote bug caught, honest abort disclosure), and better
than max2 on everything above. Max2's failure modes — locking temp 1.0,
"ceiling is 0.35-0.45" resignation, late DPO/32B-retrain bets — we simply
didn't have.

## 3. Why max2 (36.0) landed where we did

Same agent, same budget, one day apart: picked Qwen2.5-14B-Instruct as
teacher, mis-diagnosed verbosity as the main failure, locked in temp 1.0
to protect an 0.305, talked itself into a 0.35-0.45 ceiling, and spent the
tail on a 32B retrain + DPO fork that didn't land. 58.7 vs 36.0 on
consecutive days is the cell's real variance envelope — creative-writing
judging + small-model distillation is a high-variance game, and ANY single
number (theirs or ours) carries ±10+ pts of recipe-luck. Our 37.3 with the
rp/forensics levers unexploited is consistent with "solid median execution
of the right skeleton, missing the two things that separate the tail."

## 4. Why human is 94.8 and why that's not a 10h-reachable number

The "human" row is google/gemma-3-4b-it — Google's own post-training of
this exact base: large-scale multilingual SFT + RLHF + distillation from
larger Gemma/Gemini-class teachers (per the Gemma 3 tech report), months
of iteration by a team, millions of curated/preference-graded examples.
The harness then asks gpt-5-mini: "which answer is better, this or stored
Qwen3-1.7B-instruct?" A production instruct model beats a 1.7B baseline on
essentially every creative-writing prompt in every language → 94.8 is the
metric saturating, not a 10× better recipe. The realistic frame: 10h/1×GPU
agent runs on this cell live in the 30-60 band (evidence: 58.7 best-ever,
47.4 mean-best, our 37.3, everything else ≤30.9); the human row measures
what unconstrained post-training buys and is out of scope for the
benchmark's own agents by construction. gemma is also the only arena cell
where the instruct model was tuned by the base model's OWN vendor at full
scale — on the qwen cells the "human" rows (86.8/50.0/49.2) are the same
phenomenon at smaller magnitude, which is why we could pass them there
(better base-relative headroom) but not here.

## 5. If we ever re-run this cell (levers, in order)

1. Ship rp 1.1 + temp ~0.7/top_p 0.9 from the start; A/B rp {1.05, 1.1,
   1.15} on a checkpoint copy — our own cross-benchmark evidence says
   this alone re-prices the run.
2. Make judge-loss reading a standing step: --store-outputs on every gate
   eval + a loss-excerpt reader; fix what the judge SAYS, not what we
   guess (this is best-traces rule 5; enforce it in reviews).
3. Text-only fast teacher: Qwen3-30B-A3B-Instruct-2507 (our 89.6/67.8
   recipe) or Qwen3-8B; never serve a multimodal 27B for bulk generation.
4. Creative-diverse mix upsampled over no_robots (Gryphe/Magpie/
   OpenHermes slices), preamble-stripped, repetition-filtered, ~20k, 2ep.
5. Cheap limit-32 iteration gates; full-250 only to promote (both runs'
   subset traps — our hard-first-50, their 54.3→44.3@64 — say subsets
   mislead in BOTH directions).
6. Expected value honestly: those five levers put the *median* outcome
   in the mid-40s and the tail at ~60; they do not approach 94.8 (§4).

Cell disposition unchanged: GPQA-class cells still out-rank an arena
gemma re-run on ROI (weight .0904, and we already hold 4 of 4 arena rows
at proven-#1/#2). This doc exists so the levers are pre-loaded if the
priority flips.
