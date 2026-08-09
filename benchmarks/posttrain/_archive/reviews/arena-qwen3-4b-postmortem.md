# Postmortem — ArenaHard-Writing × Qwen3-4B: why 49.7 and not 86

Scope: cell-level gap analysis across run #13 (stopped, 0.3789 held) and run #16
(official **49.67 ±1.7**, clean), against the leaderboard's 86-tier. Evidence:
full solve_out traces of both runs, `reviews/run13-review.md` /
`run16-review.md`, PTB eval + rules source (arena-hard-v2.0.yaml,
gen_judgment.py, judge_utils.py, prompt.txt, disallowed_usage_judge/prompt.txt),
and posttrainbench.com `scores.js` (primary leaderboard data). Analysis date
2026-07-18.

## 1. The cell, primary source (scores.js)

| row | score | notes |
|---|---|---|
| base | 3.42 | |
| base + few-shot prompting | 19.17 | prompting alone, no training |
| naive-scaffold frontier agents | 1–26 | opus-4.5 2.0, gemini-3-pro 1.5, sonnet-4.6 17.5, opus-4.7 21.9, opus-4.8-max 22.6 |
| opus-4.8 (naive PTB scaffold) | 41.56 ±3.1 | same model that drives kapso's coding agent |
| **kapso + opus-4.8 (run #16)** | **49.67 ±1.7** | single official run, clean |
| gpt-5.4-high-reprompted | 49.53 | single run |
| glm-5.2 | 54.25 ±16.48 | 3-run mean — runs plausibly spread ~38–71 |
| fable-5 | 86.24 | **single run**, partial leaderboard entry (no full-suite aggregate) |
| human (official instruct) | 86.84 | |

Kapso lifted opus-4.8 by ~+8 over its naive-scaffold score, with low variance,
to the top of a visible **48–54 shelf** that three different scaffolds all hit.
Above the shelf there is nothing until fable-5 at 86.24 — within 0.6 points of
the official instruct row.

## 2. Eval mechanics that shape the ceiling (verified from source)

- Pairwise gpt-5-mini (temp 0, effort medium) vs precomputed
  **Qwen3-1.7B-instruct** baseline answers; two position-swapped games per
  question; no reference answer.
- Verdict weighting is not flat: decisive verdicts (`A>>B` / `B>>A`) count
  **3×**. Parity-quality answers (draws + slight wins/losses vs the baseline)
  mathematically cap near 0.50 — reaching 86 requires *mostly decisive* wins.
- Rubric includes "Concise means the response is clear and not verbose or
  excessive"; creativity/novelty; instruction obedience. Eval passes no
  sampling params — the artifact's own `generation_config.json` governs
  decoding (the lever both runs had to discover).

## 3. Run #16 score ladder (from the trace)

| t (UTC) | event | score |
|---|---|---|
| 16:13 | base, limit-50 | 0.0282 |
| 17:13 | SFT v1 (11.1k mixed, 2 ep) | 0.0331 |
| 17:49 | same weights + baked sampling t0.7/top_p0.8/top_k20/rep1.05 | 0.1316 |
| 19:01 | SFT v2 creative-heavy, t0.7 | 0.4368 |
| 19:35 | DPO on v2 | 0.4213 (not promoted) |
| 19:44–20:03 | decoding variants t0.85 / t1.0 / t0.9 | 0.4882 / 0.1161 / 0.5051 |
| 20:24 | **v2 t0.9 full-250** | **0.4815** |
| 20:32 | DPO on t0.9 | 0.4036 (not promoted) |
| iter 2 (21:15→01:16) | v3 45%-creative 0.2365@150 · incumbent control 0.4731@150 · v3.5 76%-creative 0.3538 full | all lose; incumbent kept |
| post-run | official harness eval of the same checkpoint | **49.67 ±1.7** |

SFT v2 mix (10,728 ex, 2 epochs): Opus-WritingPrompts 3,913 +
ChatGPT-4o-Writing-Prompts 2,776 + magpie_creative (Qwen2.5-72B) 2,468 +
tulu_if 1,571 — i.e. **static, published, 2024-tier distillation corpora**.
Length was checked and is not the gap (baseline median 637 tokens; our outputs
in-band). The agent's own dev-slice inspection: "competent-but-generic",
repetition degeneration on one sample, "'slop' … heavily penalized, and my
outputs have exactly that."

## 4. Why it plateaued — the diagnosis

**The artifact writes at the quality tier of its training sources, and that
tier draws against Qwen3-1.7B-instruct instead of decisively beating it.**
Four independent static-corpus SFT attempts inside one run (v2 0.4815, v3
0.2365, v3.5 0.3538, DPO 0.40–0.42) never beat 0.4815; run #13's different
static mix was heading to the same shelf (0.3066 → 0.3789); and the score was
*monotonic in creative fraction* (45% → 0.2365, 76% → 0.3538, 85% → 0.4815) —
the mixture axis explored in iteration 2 tunes within the basin, it does not
change tier. Under a 3×-decisive-win metric, tier is everything.

What the 86-tier most plausibly does (**inference** — no traces exist for those
rows): produce instruct-tier outputs on the eval distribution. fable-5 landing
0.6 points from the official instruct row is the signature of **local teacher
distillation** — generate tens of thousands of writing-domain answers with a
locally-served strong instruct model (Qwen2.5-32B/72B-Instruct-AWQ,
Llama-3.3-70B-int4, gemma-3-27b-it all fit one H100 quantized; ~2–3h of vLLM
generation both our runs left unused), SFT the base on them, optionally
on-policy re-rank with a local RM. Rule geometry: agent-visible Rule 7 forbids
downloading "an instruction tuned version of `{model}`" (the sibling), but the
enforcement judge explicitly allows "Just downloading model to do inference /
testing" and only bans fine-tuning M / shipping M / investigating M's
internals — a **non-sibling teacher is inside both texts' letter**. Human at
86.8 (not ~95) says the ceiling is "be a competent instruct model" — nothing
exotic, which is why a distilled 4B can tie it. In-house precedent: run #9's
BFCL 96.0 came from xlam-24k **+ 12k agent-synthesized examples** — the
synthesize-with-a-frontier-teacher pattern was already proven inside this
campaign.

## 5. What we missed (ranked)

1. **Strong-teacher local distillation never entered the candidate pool** —
   zero mentions across both runs' traces; the closest was run #13 consuming a
   *static* Qwen2.5-72B dataset. Eight ideation candidates over run #16's two
   iterations were all variations of static-corpus SFT + heuristic DPO.
   → owning module: **ideation priors** (ensemble prompts / researcher).
   Expected impact: the 0.50 → 0.80+ tier jump; everything else tunes the
   basin.
2. **Run #13's staged RM pipeline died with its VM.** Its selected plan had
   on-policy best-of-8 scored by a locally-run Skywork-Reward-V2-Llama-3.1-8B
   → DPO (both pair sides on-policy — a direct optimizer of the pairwise
   preference the benchmark measures). The RM was prefetched and the scripts
   written; the stage never ran (session burned on the then-unknown sampling
   trap, stopped at t+5h). Run #16 started from "No experiments found" + a
   header-only gotchas section, never regenerated the idea, re-derived the
   sampling lever (~35 min) and the creative-mix conclusion (~1.5h total
   retread). Runs #13 predate the difficulties-capture upgrade; stopped runs
   still leak everything.
   → owning module: **cross-run memory** (R11-P1-1, now with hard cost
   figures). Expected: +0.05–0.15 equivalent plus ~1.5h reclaimed per re-run,
   compounding across the campaign.
3. **Iteration 2 bet all 4.6 remaining hours on an evidence-free feedback
   diagnosis.** The judge's top directive — "Arena prompts are
   instruction-driven, dilute fiction toward professional/concise data" — was
   empirically backwards (score monotonic in creative fraction), and its
   supporting premise ("likely undertrained: 1 epoch on ~15k") was factually
   wrong (2 epochs on 10,728). Meanwhile the one *verified* novel lever —
   think-block planning that `evaluate.py` provably strips before judging —
   was deferred by the selector on defensible EV grounds and never attempted.
   Cheapest unexploited signal: the **verdict tally** (counts of
   A>>B/A>B/tie/B>A/B>>A — aggregate labels on disk, compliant with the
   per-sample leak ban) was never computed, so the run never knew whether its
   losses were slight or decisive despite the 3× weighting making that the
   single most informative aggregate.
   → owning module: **feedback judge** (evidence-gated diagnoses), with budget
   policy as accomplice (no reserve for a second bet).

Correctly handled, for the record: the DPO dead-end cost only ~41 GPU-minutes
and both DPO checkpoints were properly gated out by the promotion rule; the
limit-50 → full-250 confirmation discipline held; contamination stayed clean
under both judges.

## 6. Module recommendations (in order of expected impact)

1. **Ideation priors for judge-scored evals** — seed the ensemble/researcher
   recipe space with (a) "distill a locally-served stronger open instruct
   teacher (non-sibling; legality-check against the enforcement judge's
   allowed-uses text)" and (b) "on-policy best-of-N + local reward model →
   DPO". These are recipe classes, not benchmark-specific hints.
2. **Cross-run memory that survives dead runs** — harvest stopped/killed runs'
   PLAN.md, scripts, and best-score ladders into the experiment store at
   teardown (or from archives); fix repo-memory empty evidence quotes
   (R16-P2-2).
3. **Feedback judge: evidence-gated guidance** — require causal claims to cite
   measured aggregates; require a verdict-distribution tally before
   prescribing direction on judge-scored evals; verify checkable premises
   (epochs, dataset sizes) against the trace before asserting them.

Framework overhead also visible in this cell: the 24-min post-result CLI hang
(R16-P2-1; fixed by the declared-completion reap in 580a74a3, not yet
rebuilt into assets) and run #13's 5h stop mid-stage.
