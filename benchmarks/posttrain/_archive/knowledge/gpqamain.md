# GPQA Main — learnings from finished runs (OPTIONAL reading)

Distilled, score-attached observations from finished kapso runs on this
benchmark (across base models) plus the public official-baseline traces.
These are suggestions to draw inspiration from, not instructions: your base
model differs, so verify anything you adopt against your own recon, and
deviate freely when your measurements disagree.

Provenance: kapso GPQA-main runs — Qwen3-4B-Base 32.14, SmolLM3-3B 29.69,
gemma-3-4b-pt 29.02, Qwen3-1.7B-Base 22.32 (all official, judges clean,
2026-07-26). Cell refs: base ~8.5 avg, proven band ~29-34, human ~36.

## The format hazard is the #1 early failure — fix it before anything else

GPQA scores BELOW the 25% random floor when the model doesn't emit a
parseable answer, not because it lacks knowledge. Measured base rates:
Qwen3-4B 8.67%, SmolLM3 ~5.9% with only a 20-30% parseable-`ANSWER:` rate.
The scorer wants a terminal `ANSWER: <LETTER>` line (inspect-ai
`multiple_choice(cot=True)` + `choice()`, first-match-wins, MULTILINE).
- Train with an anchored terminal tail — a guaranteed `ANSWER: X<eos>` after
  the reasoning (SmolLM3's win used anchored rows giving a 60-token
  `ANSWER: X<|im_end|>` guaranteed tail).
- Gate on parseability BEFORE promoting: run `--limit 20`, check termination
  rate + parseable rate + accuracy_given_parseable. A below-random score is
  a format bug to fix, not a dead end (SmolLM3 recovered 16.67 → 29.69).

## Subset bias: `--limit N` is a BIASED estimator, not just noisy

`--limit N` takes the FIRST N of 448 in a fixed seed-42 shuffle — a biased
subset. Measured on the SAME artifacts, @150 read HIGH vs full-448 both
times: Qwen3-4B 37.33@150 → 32.14 official; Qwen3-1.7B 28.67@150 → 22.32
official. **Spend one full-448 eval before the freeze and promote/report on
that number.** This is the single lesson that cost the 1.7B run its band
position — it never ran a full-448 gate. (Cross-benchmark twin of the AIME
n=30 lesson; it transferred cleanly — later GPQA runs referenced "full-448"
hundreds of times and gated on it.)

## The winning recipe skeleton

Full fine-tune on curated science-MCQ reasoning traces, rendered byte-exact
to the eval template, completion-only loss, terminal `ANSWER: X`.
- **Data**: DeepSeek-R1-0528 science traces (nvidia/OpenScienceReasoning-2)
  and 4-choice science MCQ (nvidia/OpenScience, ARC). NATIVE R1 reasoning
  traces carry the lift — a SmolLM3 ablation that DROPPED them fell 29.69 →
  27.46, so keep them.
- **4-choice filter (measured, gemma run)**: OpenScienceReasoning-2 rows
  carry options A-J (up to 10 choices); only ~30.2% are clean 4-choice A-D
  with consistent answers. Filter to those — GPQA is strictly 4-choice.
- **Decontaminate** exclusion-only vs GPQA main/extended/diamond (all
  configs/splits), normalized-text + n-gram; aggregate counts only.
- 2 epochs, full FT, bf16, FA2, completion-only loss, max_seq ~6144.

## Decoding and the A-bias

- temp 0.7 or greedy both land near the same place; the greedy-vs-temp gap
  is within the ±2.2pp full-448 noise floor (SmolLM3 greedy 31.03 vs
  temp-0.7 29.69 — NOT promoted, correctly, on a within-noise gain).
- Watch letter-position bias: the gemma run measured a residual ~41% A-bias
  (over-predicting choice A). Letter-shuffle-debiased training data + model
  souping cut it.

## Model souping is a cheap, real final lever

gemma's best artifact was a zero-training uniform soup of two same-init
full-FTs (0.5·Stage-B + 0.5·Stage-C, differing only in a training-data
letter-shuffle): 30.36 full-448 in-run, above both parents, md5-verified a
genuine average. Souping two same-init siblings costs no GPU and reduced the
A-bias.

## Promotion discipline

n=448 stderr ≈ ±2.2pp; @150 ≈ ±4pp AND biased high. Gate on full-448, never
promote on a sub-noise gain, and keep the incumbent `final_model` intact
(atomic swap only on a confirmed win). GPQA is knowledge-bound for a 3-4B
model — expect a ~28-30% concise plateau; further gains need bigger/cleaner
data, not decode tricks.

## Per-base-model notes

- Qwen3 bases: eos [151645, 151643]; strongest cell (4B reached 32.14, ~1σ
  under proven #1 34.1). 1.7B is the weakest capacity — full-448 gating
  matters most there.
- SmolLM3: eos [128012, 128001]; parse wall is steepest here (base ~5.9%),
  so the anchored-tail format fix is highest-value. **Rescore caveat**: vLLM
  0.11 in vllm_debug.sif hits a CUDA-graph illegal-memory crash on SmolLM3's
  NoPE-interleaved attention — the fresh-VM rescore can't serve it, so use
  the in-run full-448 official-`evaluate.py` score of record (the model
  serves fine in the agent's own container).
- gemma-3-4b-pt: eos [1, 106]; multimodal arch
  (Gemma3ForConditionalGeneration) — freeze vision_tower +
  multi_modal_projector, copy processor/preprocessor configs into saved
  checkpoints (vLLM dies without them), pt ships NO generation_config.json,
  262k vocab OOMs fp32 logits (use fused/chunked CE). Surprisingly strong
  on this cell (weakest base 1.6 → 29.02) — science-MCQ knowledge favors
  gemma's pretraining.

## Infra note

The post-solve harness final eval fails to write metrics.json (model-
agnostic serving bug, 7/7 across the campaign) — always plan to rescore on
a fresh VM (or, for SmolLM3, fall back to the in-run full-448). Verify
`final_model` integrity with a fresh-process load before finalizing.
