# Arena-Hard Writing — learnings from finished runs (OPTIONAL reading)

These are distilled, score-attached observations from finished kapso runs on
this benchmark (across base models, plus model facts from sibling
benchmarks) and from the public official-baseline traces. They are
suggestions to draw inspiration from, not instructions: your cell may
differ, so verify anything you adopt against your own recon, and deviate
freely when your measurements disagree.

Provenance: kapso runs on Qwen3-4B-Base (official 89.6, above the human/
instruct row), Qwen3-1.7B-Base (67.8, cell record, +10.7 over the best
baseline agent), SmolLM3-3B-Base (56.6, +7.4 above human); official
opus-4.8-max traces for cross-checks.

## The skeleton that won all three runs

Serve a strong open instruct teacher LOCALLY with vLLM → generate over a
decontaminated, language-matched real-user prompt pool → assistant-only-loss
SFT rendered byte-exact through the eval's jinja template → 2-3 epochs full
FT → bake decoding into generation_config.json → promote behind a full-250
gate. Ladders: 4B static-SFT 85.5 → +BoN-distill 88.6; 1.7B 3.2 → first
distill SFT 40.7 → judge-gated round 62.5 → 68.6; SmolLM3 → 56.6 with the
same shape. Static open corpora alone plateaued at ~10-11 in every run that
tried them.

## Teacher choice (measured)

- Qwen3-30B-A3B-Instruct-2507 (the FP8 build fits comfortably on one H100)
  was the teacher in all three winning runs.
- Cross-tokenizer transfer is PROVEN: the SmolLM3 run distilled the Qwen
  teacher into a different tokenizer family at text level — same-family
  teachers are not required.
- Avoid multimodal teachers for bulk generation: a 27B multimodal teacher
  OOM'd, forced enforce_eager, and generated at 86-120 prompts/min — about
  half the effective throughput, for no measured quality edge.

## Decoding is a free lever — ship it deliberately

- Both qwen-cell winners shipped `temperature 0.7 / top_p 0.9 /
  repetition_penalty 1.05` plus the FULL eos list, "exactly as evaluated".
- The best official trace on the hardest cell diagnosed looping from judge
  output and shipped rp 1.1 (winning run 58.7). Do NOT ship rp 1.0 "to
  protect refrains" without an A/B on a checkpoint copy — one run did and
  its artifact rambled into the 16k cap on many prompts (the concise rubric
  punishes it and the eval slows ~2×).
- Base models under-learn turn-end tokens: force the eos list in BOTH
  generation_config.json and config.json, and verify stop rates on ~12
  long-form prompts before any eval (an under-trained eos once produced a
  ~0.0 score from an otherwise fine SFT).

## Data quality loops beat data volume

- Judge-gated distillation + best-of-N: the 1.7B run's dual-teacher
  judge-gated pool + RAFT (best-of-8 self-samples, teacher-judged with the
  verbatim rubric) moved 40.7 → 62.5 in ONE round — the largest measured
  single-iteration gain on this benchmark. The 4B run's teacher best-of-3
  produced its final winning dataset.
- Working volumes: 9-19k examples, 2-3 epochs, full FT. Strip teacher
  preambles; filter repetition; cap response length near the baseline
  answer band (median ~640 tokens; the judge rewards conciseness).
- Match the eval's language mix (~73% latin / 14% zh / 10% ru, measured on
  question.jsonl): the stored 1.7B-instruct opponent is weakest outside
  English, so zh/ru coverage is cheap points; one run's es/fr-only slice
  measurably left the zh/ru axis unfixed.

## Preference stages: 0-for-2, measured

DPO produced NO gain at 4B (85.24 vs SFT's 85.51, loss flat, not promoted);
the 67.8 run skipped preference optimization by design and spent the time
on judge-gated SFT data instead. Budgeting DPO/RM stages up front has never
paid on this benchmark; treat them as a tail option at most.

## Evaluation protocol traps (both directions)

- limit-N subsets are BIASED, not just noisy: the seed-42 first-50 is
  unrepresentatively hard (21.05@50 → 35.50@250 on the same artifact; the
  incumbent 20.9@50 → 23.33@250); official traces saw the reverse too
  (54.3 → 44.3@64 → 61.2@250). Use small limits only as coarse gates;
  promote and report on full-250.
- Never conclude "plateaued" from a ≤50-question read.
- Never kill a running full-250 eval on one stale check: a completed 88.64
  result was once discarded because the agent checked a single time, 5
  minutes early, under a phantom deadline.
- The judge's loss explanations are readable: run the eval with
  --store-outputs and read WHY you lose (a short loss-excerpt reader
  script pays for itself). The rp-1.1 fix in the 58.7 official trace came
  directly from this; guessing failure modes did not find it.

## Mechanical traps that recur

- evaluate.py resolves its data path relative to CWD — always invoke from
  /home/ben/task (this exact FileNotFoundError has hit multiple runs).
- Promotion copies must exclude checkpoint-*/optimizer subdirs (46-47GB
  final_model bloat has happened twice; use ignore_patterns).
- Auto-saved generation_config after training can be defective (duplicate
  eos entries, missing temperature) — always overwrite it explicitly and
  diff before/after.

## Per-base-model notes

- Qwen3 bases: eos [151645, 151643]; render `<think>\n\n</think>` per the
  eval jinja's semantics (A/B'd and adopted in the 67.8 run).
- SmolLM3: eos [128012, 128001]; cross-tokenizer distillation proven here.
- gemma-3-4b-pt: eos [1, 106]; architecture is multimodal
  (Gemma3ForConditionalGeneration) — freeze vision_tower +
  multi_modal_projector, copy processor/preprocessor configs into every
  saved checkpoint (vLLM dies without them), the pt checkpoint ships NO
  generation_config.json at all, and the 262k vocab OOMs fp32 logits at
  the loss step (use a fused/chunked cross-entropy).

## Variance honesty

The same official agent scored 58.7 and 36.0 on this benchmark's hardest
cell on consecutive days. Treat any single sub-250 number as ±10; bank
conservatively, confirm promotions at full-250, and prefer levers with a
mechanism you verified over levers that merely scored well once.
