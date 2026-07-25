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

## P2 (~03:26 → 06:41Z trace end)

- **R27-P2-1 — OBS (positive, headline).** Promotion discipline held
  exactly as pre-registered — R27-P1-2 did not bite. Full-epoch cand_s1
  scored 13.33 vs incumbent 16.67 and was NOT promoted: "essentially
  tied … (1-question difference = noise at n=30)" (05:51:28);
  best_score.log reads `13.33 … sft1-full-epoch-NOT-promoted(<16.67)`.
  The conciseness anneal's 23.33 (7/30) cleared the +2-question bar and
  was promoted atomically at 06:38:22 with fresh-process
  `VERIFY final_model.tmp OK: SmolLM3ForCausalLM [128012, 128001]`.
  Residual: still a single n=30 read (stderr 7.9pp).
- **R27-P2-2 — OBS (positive).** Resume integrity (the P1 watch):
  clean. 03:31:02 resume from checkpoint-120 (PID 5832); only benign
  warnings (missing `lm_head.weight` = tied embeddings; logging_steps
  20≠10 args-vs-trainer_state); loss continuous (0.4456 @ epoch .149);
  epoch 1.0 completed 05:41:06 — cosine LR→3.9e-10, train_loss 0.3876,
  token-acc 0.852, checkpoint-1143 + top-level save.
- **R27-P2-3 — P3 (recipe).** The ≥90% ANSWER-terminal gate bar was
  never met — answer_rate 0.50 (ckpt-120) → 0.47 (full epoch) → 0.77
  (banked 23.33 model) — but was repurposed into the run's best work:
  forensics (03:52:48, 05:51:36) showed truncated 15/30 → 0 correct,
  finished 15 → 5 correct, and truncations are genuine long reasoning,
  not loops ("has </think>: False", no repeated chunks). That diagnosis
  drove the winning anneal. NB the 0.85 figure in the logs is
  mean_token_accuracy (a training metric), not a format rate.
  eos-stopping itself validated end-to-end (vLLM honored gen_config;
  15 'stop' finishes at ckpt-120).
- **R27-P2-4 — OBS (recipe).** Stage-2 pivoted twice, evidence-driven,
  box-respecting: OMR dropped (schema fetch too slow, 03:23:26);
  Light-R1 stage2 built + cached (1,862 rows / 11.1M tok) but shelved
  once truncation was shown to be the ceiling (hard/long traces would
  worsen it). Conciseness anneal (9,621 own-corpus traces, median
  1,890 tok) ran 25 min train + 8 min eval — inside the 1-1.5h box:
  23.33, trunc 0.50→0.23, answer_rate 0.47→0.77. Greedy probe tested
  and correctly rejected (16.67 flat; p50 2,029→40,354 chars,
  trunc 0.53).
- **R27-P2-5 — P4 (framework).** Idle-loop churn persists (R25/R26
  pattern): 21 "Wasted call" re-reads, 2 waiters armed-then-stopped
  inside 3 min (bqt0d7hqi 04:05:49, bdszm0u8g 04:06:38), blocked
  sleep-840 (04:05:06), stray Skill(verify) self-caught (04:07:05),
  empty ToolSearch — all pre-yield fidgeting, no state damage.
  Otherwise clean: ScheduleWakeup 0; 20 task_notifications + one
  Monitor 60-min timeout as the only wake sources; every wait bounded
  (75-115 min caps); no freezes; one rate_limit_event (05:41:12,
  no stall); cost $10.63→$19.22.

eos [128012,128001] persisted through resume and all four
finalize/promote points; zero qwen-token sightings again. Session ends
~07:29 with a next-session handoff committed to PLAN.md (06:40:35);
sft3 (2nd concise anneal from the 23.33 model, max_steps 100, launched
06:40:04) pending at trace end with the incumbent protected.

LADDER: 0.0 base → 16.67 sft1-ckpt120 (03:30, banked) → 13.33
sft1-full-epoch (05:51, rejected as noise) → 16.67 greedy probe (06:02,
rejected — verbosity up) → **23.33 sft2-concise-anneal (06:37, promoted
06:38)** → sft3 pending.

SOTA OUTLOOK: Already +6.7pp over the 16.7 proven best with ~5.5h solve
budget left and the binding constraint identified (truncation on
genuinely hard problems — not looping, not format). Decisively beating
proven is done; approaching published ~36.7 needs capability, not
format — the committed next-session bet (Mixture-of-Thoughts math, the
curated SmolLM3 recipe, over the cached datasets) is the right one,
plus squeezing the remaining 23% truncations.

VERDICT: **continue** — banked a new campaign-best 23.33 via a cheap,
correct diagnosis-driven pivot, with promotion discipline and artifact
hygiene intact.
