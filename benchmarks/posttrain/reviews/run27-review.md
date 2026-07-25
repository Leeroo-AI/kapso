# Run #27 review — aime2025 × SmolLM3-3B-Base (us-east4-a)

Second AIME cell, first on a non-qwen base (base 3.3 | best proven 16.7
opus-4.8-max/fable-5 | human 26.7; weight .2265). Launched 2026-07-25
02:08Z, same upgraded stack as runs 25/26. SmolLM3 hazard watch: own
template (smollm.jinja), eos 128012 `<|im_end|>` — qwen-eos copy-paste
from OpenR1-corpus recipes is the named killer.

## P1 (t+0 → ~t+78min)

Headline: **zero qwen-eos contamination** — not one 151645/151643
anywhere in 3,649 lines. Every eos mention is SmolLM3-native, measured
from the tokenizer (base eos `<|end_of_text|>`=128001; `<|im_end|>`=128012
NOT eos → artifact must ship [128012,128001]), and the banked final_model
verifies `VERIFY_OK arch=SmolLM3ForCausalLM eos=[128012, 128001]`.
Template parity is byte-exact and quirk-aware: unclosed system header in
the no-tools path, default `/think` system prompt, bare
`<|im_start|>assistant\n` generation prompt (model self-emits `<think>`),
`{% generation %}` tags → assistant_only_loss, all re-proven by an
implementer-written train/eval byte-parity regression test (02:36,
PASSES). Run-25's eos-lever/scorer-read playbook fully re-derived for the
new family at ~2-3 min implementer re-read cost, offset by that new test.

Boot clean (CUDA/H100 checks, tools=30, gates run on empty history).
Lens planner (fable-5 web-enabled, 128s/$0.30) produced two math-native,
family-aware lenses: L1 distillation-SFT done to perfection (OpenR1-220k
backbone, Light-R1 curriculum, template/token-budget pinning) / L2
token-economy + verifiable-reward (s1 budget forcing, GRPO polish, n=30
noise discipline) — with explicit lens→member matching. Members 2/2 + 2/2
(codex:gpt-5.6-sol 641s; fable-5), 4 pooled, 0 dropped. Negative-space
Coverage: all five families on every candidate with substantive
Not-measured closers, e.g. Metric mechanics: "MEASURED — equal-weight
accuracy over 30 samples; 1 question = 3.33pp; single-run σ ≈ 7-8pp at
p≈0.2-0.3 … Not measured: temp-0.6 run-to-run variance for the tuned
model — quantified from the first repeated eval pair."

- **R27-P1-1 — OBS (positive).** See headline; the classic family-swap
  killer was defused at ideation and re-verified at bank time.
- **R27-P1-2 — P3 (recipe).** Gate design weaker than run 25's: promotion
  rides the 30-item test set (+2-question bar, repeat-eval on ±1); AIME
  2024 is only an "optional held-out" (excluded from training for that
  purpose). 4-6 evals of selection pressure on n=30 can promote noise —
  run 25 used a 90-problem AIME 2022-24 held-out at n=4.
- **R27-P1-3 — OBS.** Member tooling asymmetry recurs (R25-P1-3): codex
  candidates' harness claims sourced from public mirrors, not local
  files; selector compensated with its own reads ("I verified the
  harness directly", 02:29) before crowning C3.
- **R27-P1-4 — OBS.** Only error all segment: jinja TemplateSyntaxError
  on `{% generation %}` during member probing (02:17:57), self-recovered
  ~10s and converted into the assistant_only_loss verify item.
- **R27-P1-5 — OBS (watch).** Training killed at step 120/1143 to run
  the format gate on the single H100 (correct per plan; final_model
  banked first — safe vs 0.0 baseline). Resume-from-checkpoint
  discipline is now the P2 watch.
- **R27-P1-6 — FRAMEWORK (positive).** ScheduleWakeup zero. R23-P3-1
  clean: no log-only alarms — all waits are notifying background tasks
  (28 task_notification events), dead-man alarm set 02:43 and refreshed
  02:48; bounded sleep-90 polls inside notifying tasks.

Base floor measured, not assumed: full-set baseline 0.0% at 02:42 AND
base completions inspected for length/failure profile — settling a
declared Not-measured item and fixing run 25's R25-P1-2 blemish.
Decontamination exclusion-only: exact + digit-normalized + 10-word
shingles vs AIME 2024+2025; 382 low-Jaccard hits reviewed, 0 removed,
logged. Data build: 28,206 verified-complete traces / 107M tokens,
median 3,158 tok, 1k-10k window. Loss 0.48→0.42 by step 120; 1 epoch ≈
2.44h fits the session; format-gate eval launched detached 03:21:39
(t+73min) against a banked, eos-fixed model.

SELECTED PLAN: C3 chassis (fable) + C1's Light-R1 hard-anneal graft +
C2's OpenMathReasoning frontier filters (pass_rate_72b_tir 0.25-0.625),
GRPO dropped on EV. Stage-1: full-FT SFT on 28.2k length-filtered
verified OpenR1-Math-220k R1 traces, byte-exact smollm.jinja render,
terminal `ANSWER: N`, assistant_only_loss, lr 1.2e-5 cosine (launched
config matches plan), eos [128012,128001] + temp 0.6/top_p 0.95 in
artifact generation_config. Stage-2: Light-R1 stage2-3k + ≤1k OMR hard
band at lr 5e-6, time-boxed 1-1.5h; early format gate ≥90%
ANSWER-terminal / ≤10% cap-hits; consolidation reserve ~45-60 min.

Verdict: **continue** — SmolLM3's published ~36.7 AIME25 via R1-trace
SFT on this exact base means beating the 16.7 proven best needs only
~half the published gain; everything format-critical is measured, banked
early, and family-correct. P2 watches: format-gate result + eos stopping
end-to-end, resume-from-checkpoint after the gate kill, stage-2 branch,
and whether promotion discipline holds against 30-item noise (R27-P1-2).
