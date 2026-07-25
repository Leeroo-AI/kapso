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

## P3 (06:48Z → end) + closing

Coverage: GCS bytes 425K-900K (06:40→08:04, both session boundaries'
window), delta 900K-3.5M (08:04→11:29:29Z), last 1.5MB tail, plus
probes at 20/80/140MB inside the 165MB middle. Live agent stream ends
11:29:29Z; solve time 09:17:52; cumulative cost $59.32.

- **R27-P3-1 — OBS (positive, headline).** Endgame honesty closed the
  loop the run itself opened. sft3 scored 13.33 and was mechanically
  not promoted ("over-shortened", best_score.log 07:03). The agent then
  spent its session-1 tail re-measuring its OWN incumbent: three
  full-set runs of the same weights = **23.33 / 16.67 / 10.00, pooled
  16.67% (15/90)** (07:20), wrote result.json with score 16.667 + the
  runs array, and named it in difficulties ("single-run promotion
  decisions rode luck… the honest reported score is the pooled 16.67%,
  not the best single draw of 23.33%"). Session 1 finished 07:22:46 —
  7 min before its cap — with fresh-load verify and GPU at 0 MiB.
- **R27-P3-2 — OBS (positive).** Both session boundaries ran the full
  upgraded chain. Boundary 1 (07:22→07:30): all five tags extracted
  incl. a 9-item quantified technical_difficulties; feedback judge was
  a real session (352.7s, $1.14, 11 tools) that read the actual
  evaluate.py and artifacts before ruling DECISION: CONTINUE with three
  ranked prepared bets. Boundary 2 (11:23:57→11:29:21): judge audited
  integrity ("EVALUATION: VALID / untampered — governed evaluate.py
  and templates/ NOT modified"), set node-1 score to the robust 6-run
  pooled 0.1556, and produced the TIME-ALLOCATION GRADE + LEDGER +
  "#1 UNTRIED LEVER" + verbatim INVARIANTS handoff. Rule-6 full-field
  renders confirmed: judge-2's prompt carries node 0's complete
  solution/feedback/technical_difficulties JSON.
- **R27-P3-3 — P4 (framework, recurring).** R15-P2-1 struck at BOTH
  boundaries: each judge's first Read was
  kapso_campaign/kapso_evaluation/evaluate.py → "File does not exist"
  (07:24:21, 11:24:06), recovered in one step. Deterministic wasted
  call, 4th run family it has recurred in; the deferred fix (resolve
  evaluation_script_path against the session dir) is now overdue.
- **R27-P3-4 — P2 (framework, new — the 170MB trace).** solve_out is
  170.6MB but the live stream is only ~1.2MB. At 11:29:29Z the
  consolidation finally-block — `print(orchestrator.search_strategy.
  get_experiment_history())`, benchmarks/posttrain/runner.py:344 —
  emitted the full SearchNode list repr as ONE ~169.4MB line. The
  carrier field is `code_diff` (strategy.py:713 stores the full
  `git diff` vs parent): sessions committed inspect eval logs
  (logs/*.json holding complete per-sample AIME-2025 completions,
  looped generations included) plus iter2's RSFT generation JSONL into
  the campaign repo — session-1's .gitignore fix was exp_0-specific
  (`output_data_generic_exp_0/*.jsonl`), so exp_1's data files were
  diffed wholesale. All three mid-file probes are raw completions.
  Neither the eval wrapper nor forensics cats bloated the live stream;
  the whole overrun is this single end-of-run journal echo. Fix: print
  a summary repr (drop code_diff/agent_output — a log print, not an
  LLM input, so rule 6 permits) + a session-agnostic data .gitignore.
- **R27-P3-5 — OBS (hygiene).** ScheduleWakeup 0 across the entire
  remainder. Session 2's only wake sources are 23 task_notifications;
  3 result:error in 3h40m (benign file-modified-since-read, one
  transient, judge path miss); 2 rate_limit_events, no stalls; the
  lingering session-1 CLI was reaped by the framework after 60s; the
  orchestrator honored the finalization reserve at 11:29:29 and the
  summary JSON closed with final_model_present: true. Residual: 28
  "Wasted call" re-reads — the R25→R27-P2-5 idle churn, unchanged.
- **R27-P3-6 — P3 (recipe, iteration-2 bet).** The selector (07:42)
  chose C3 — fresh natively-concise 34.8M-tok stage-1 + RSFT/STaR +
  AIME-2024 held-out selection — over the judge's #1 bigger-stage-1
  bet, using an n=30 noise readout as anti-scale evidence ("full epoch
  13.33 < ckpt120 16.67" is a one-question difference) while, in the
  same breath, correctly tightening promotion to 3-run means. The bet
  lost: a1ep = 12.22% (11/90) < incumbent, root-caused as capability
  sacrificed by the shrunken corpus; the #1 lever went untried and the
  closing judge's ledger said so — "Opportunity cost ≈ the campaign's
  entire remaining upside."
- **R27-P3-7 — OBS (recipe, execution).** Iter2 conduct itself was
  clean: stage-1 345 steps → format gate 13.33 (08:42); RSFT
  auto-downsized when vLLM fell back to the transformers backend
  (~6.7k tok/s — SmolLM3 has no native vLLM kernel in 0.11), 2 rounds
  → 391 verified traces / 1.56M tok (47.8% prompt pass); held-out
  sweep (3 cands × temps {0.3,0.45,0.6} × 4 seeds, AIME-2024): light
  anneal cp13@0.6 = 11.67 wins, 2-epoch over-train caught, lower temps
  measurably worsen truncation; mechanical non-promotion at 12.22;
  leftover budget spent re-measuring the incumbent (fresh 3-run 14.44
  → **6-run pooled 15.56%, 28/180**). A real flow bug was caught
  before harm: promote.sh re-ran finalize with hardcoded temp 0.6,
  which would have silently discarded any selected temperature —
  fixed by threading --temp.

CLOSING VERDICT. **Official: 0.1667 ±0.069 (5/30), CLEAN.** The
fully-integrated re-pinned judge ran as a real 693KB session (911k
input tokens) and itself wrote "no contamination detected" / "only
allowed use detected"; final_eval was clean (30/30 samples, 0 retries,
artifact generation_config honored end-to-end). **This TIES the proven
#1 row (16.7, opus-4.8-max/fable-5) — same 5/30 — it does not beat
it**; human 26.7 stands. Claim the row as a tie for best agent.

The 23.33→16.67 story is regression to a measured mean, not an unlucky
draw: the banked 23.33 (7/30) was the top of ~8 single-run draws whose
robust estimate the run itself produced — 6-run pooled 15.56% (28/180)
— and the official single run landed +0.4σ from it (and inside one
single-run σ≈7.9pp of 23.33). Kapso's own honesty machinery predicted
the official number in-run; treat any future "banked best" as a draw,
never a level, until pooled.

Key levers for a re-run that decisively BEATS 16.7 (needs official
≥20.0 = 6/30, i.e. a true mean in the mid-20s): the run's own #1
untried lever — a larger/better stage-1 from open-r1/Mixture-of-
Thoughts 'math' (the curated recipe behind this base's published
~36.7) or OpenR1-220k 'all', 1-2 epochs of FULL-length traces, then
exactly ONE concise anneal; keep temp 0.6 + eos [128012,128001];
promote only on 3-run pooled means; avoid the proven negatives
(shrunken concise stage-1 12.22, stacked anneals 13.33, 2nd epoch
13.33, greedy). Framework pre-reqs before the next launch: the
consolidation journal print (R27-P3-4) and R15-P2-1.
