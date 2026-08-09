# Run #28 review — aime2025 × gemma-3-4b-pt (us-east4-a, launched 2026-07-25 08:17Z)

Hardest cell of the sweep: base 0.0 | best proven 3.3 gpt-5.4-h-rp |
human 10.0. Same upgraded stack as runs 25-27. Combined P1+P2 review on
a trace running to ~17:36Z (~45 min before solve end — exp2's session
close/boundary not in view). Hazard watch going in: gemma multimodal
arch + vision freeze, eos [1,106], gemma3.jinja (run 24), and the AIME
16k-truncation/gate-noise/eos-extraction levers (runs 25-27).
P3 close-out appended below after RUN_DONE (19:07Z) + rescore.

**FINAL: official 0.0 (0/30) via rescore · both judges clean
(integrated re-pin) · $95.66 · completes the AIME table
(6.67/20.0/16.67/0.0; 4-model avg 10.85 edges opus-4.8-max's proven
10.8 best-agent average).**

## P1 (t+0 → ~t+85min)

Headline: **every known gemma trap was owned at ideation, not
self-recovered downstream — and the run found a new one.** Zero qwen-eos
contamination (0 hits for 151645/151643 in 13,977 lines); vision_tower +
multi_modal_projector frozen in train.py with live proof at every launch
("[model] trainable=3.88B frozen=0.42B"); packaging plans reassemble the
full Gemma3ForConditionalGeneration with genconfig {temp 0.6/top_p
0.95/top_k 64, eos [1,106]}. The new trap (hit in P2, root-caused same
hour): a locally packaged gemma-3 fine-tune must also carry
`preprocessor_config.json` + `processor_config.json` — vLLM builds an
image processor for the multimodal arch and dies with IndexError without
them; the base only worked because vLLM auto-fetched from the hub.

Boot clean (H100 + cuda-tensor write). Lens planner (fable-5,
web-enabled, 139s/$0.36) produced math-native lenses with the
measurement layer built in: L1 curated long-CoT distillation
(OpenThoughts3/OMR/OpenR1/Light-R1, traces under the 16k budget,
byte-exact gemma3.jinja + `ANSWER: N`) / L2 teacher generate-and-verify
+ own-the-measurement-layer (+GRPO/RLVR polish if time). Members 2/2 +
2/2 (codex:gpt-5.6-sol 711s; fable-5), 4 pooled, 0 dropped. All four
candidates shipped coverage families with substantive Not-measured
closers (e.g. metric mechanics: "n=30 → 3.33 pts/item, stderr ≈7-8 pts…
Not measured: temp-0.6 run-to-run variance — repeat full eval before
final promotion").

- **R28-P1-1 — OBS.** R24-P1-3 half-echo: the lens text again never
  names the multimodal-arch hazard — but this time both Claude
  candidates carried vision-freeze + full-arch reassembly natively, so
  nothing needed downstream rescue. Lens blind spot persists; candidate
  layer now covers it.
- **R28-P1-2 — OBS (3rd AIME recurrence of R25-P1-3/R26-P1-3).** Codex
  member again couldn't read the evaluator and guessed the scorer
  (Inspect `answer()` — "wrong in detail" per selector); selector
  compensated by checking Claude's line-numbered MEASURED claims against
  source and crowning C4 on groundedness ("decisively superior").
  Member tooling asymmetry is now 3-for-3 on AIME.
- **R28-P1-3 — OBS.** Baseline banked early and honestly: base model
  3.33% (1/30) at 08:56Z, written to best_score.log as a BASELINE line.
  This is the peeked 0.0333 — never a shipped artifact (final_model
  stayed empty until 11:23Z; rule 7 forbids shipping the base).
- **R28-P1-4 — OBS.** Launch churn, all self-recovered in ~9 min: first
  train launch OOM'd at the loss step (fp32 logits [2,8192,262144] ≈
  17GB from gemma3's 262144 vocab) → liger fused-CE + bf16; then
  nohup block-buffered stdout mistaken for a stall → relaunch with
  `-u`. Both captured as difficulties #1/#3 with correct root causes.
- **R28-P1-5 — CONCERN (framework, new churn signature).** A
  read-poll compulsion: after arming a notifying monitor the opus-4-8
  implementor re-Reads the task output file between vows to stop — 11
  "I'll stop/await" thinking lines each followed by another `Wasted
  call` Read in 09:14-09:17 alone; at 09:16:55 it says "I keep
  re-reading prematurely" and keeps going. 78 wasted calls in hour 09.
  The harness dedup guard defuses the content, not the turn.

SELECTED PLAN (C4 backbone + C3/C1/C2 grafts): full-FT of the text
tower on 10,000 OpenR1-Math-220k R1-671B pre-verified traces
(correctness_math_verify, integer answers, ≤2/problem, ≤8K gemma
tokens, triple-deconned vs aime25+aime24 — 691 removed), adamw_bnb_8bit
lr 1e-5 cosine bs2×16 seq 8192, 2 epochs; package with eos [1,106] +
sanity-generate gate; AIME-2024 val harness; GRPO stage-2 gated on
SFT ≥ baseline+7 AND ≥3h left; consolidation A/B. First promotion
projected ~3.5-4h.

Verdict at t+85min: **continue** — plan grounded, data built (median
3,724 tok, p95 7,547), training running post-OOM-fix, though measured
throughput (~12-16s/step: gemma3 sliding-window sdpa + grad-ckpt +
huge-vocab, ~17% MFU) already means ~1 epoch this session, not 2.

## P2 (~09:46 → 17:36Z trace cut)

Headline: **the 0.0333 was never beaten — seven full-30 official reads
on five fine-tuned artifacts all returned 0/30 — but iteration 2
converted a resolution-free metric into a real signal (MATH-500 dev
41%) and shipped the measurably best checkpoint anyway.** Session 0:
ckpt-100 0/30 (09:54, format read: 21/30 end with a number), ckpt-400
0/30 (11:21; ANSWER-rate up, ramblers 12→8, "real attempts" — promoted
as insurance 11:23, best_score.log honestly says
`insurance-below-baseline`), ckpt-600 greedy 0/30 + temp-0.6 0/30
(12:24/12:37; ship temp-0.6 for the variance lottery, greedy
"deterministically rambles to 0"), temp-1.0 lottery eval aborted when
vLLM startup ran slow and the principled temp-0.6 config restored to
protect finalization. Iteration 2 (13:37→cut): OMR corpus (4,318
verified AoPS problems, 684 deconned vs aime25+aime24+**MATH-500** —
the dev set entered the blocklist the moment it became an instrument),
FA2 adopted after a 20-step probe (measured 10.87 s/step vs iter-1's
16), fresh full-FT 4 epochs on the 3,048 non-easy rows (14:06-16:25),
both 3ep/4ep packaged in parallel, dev-adjudicated, 3ep shipped:
MATH-500 41% vs 33% (4ep overfit), AIME-24 pass@1 tied 0.0167, official
0/30 both (17:08, 17:32).

- **R28-P2-1 — OBS (mandate item: the 10:01→11:14 quiet wait).**
  Single wake source, but a hybrid: bg task `bzio6ufmu` armed 09:58:07
  as "single watcher + dead-man's alarm — polls proc-alive every 60s,
  notifies on checkpoint-400 or proc-exit, bounded to ~80 min" (agent's
  own words, 10:00:50). The wake landed 11:14:07 on actual ckpt-400
  readiness (77 polls) — 7 min past the agent's ~11:07 ETA (training
  drift, not framework) and 3 min inside the bound. Zero-lag delivery
  (task_notification → next turn same second). Thinner than run 26's
  layered two-source idiom — a silently-dying monitor would have left
  no wake until the session cap — but the proc-exit + 80-min bounds
  mean it was never a bare until-loop. Zero dud fires all run.
- **R28-P2-2 — CONCERN (framework, cost).** Session-0 cost $73.47 /
  413 tools / 4.51h — ~3× run 24's same-model impl0 ($23.53/4.7h).
  Driver visible in the ledger: 188 `Wasted call` re-reads (ALL in
  session 0: 78 in hour 09, 52 in hour 12, 41 in 13h) plus micro-turn
  chatter; cumulative cost jumped $14→$30.7 during the worst burst
  window (09:17→10:01). Session 1, same model, post-feedback: zero
  wasted calls, quiet 48-min turns, $10.25 by 16:24. Framework ask:
  after N consecutive dedup hits on one path, the guard should
  instruct the turn to END, not just point at the earlier result.
- **R28-P2-3 — OBS (boundary mechanics, clean; mandate item).**
  Session 0 self-closed 13:12:16 (16,247s, 29 min under cap) with 5/5
  XML tags; technical_difficulties = 6 root-caused items (fp32-logits
  OOM, processor-config IndexError + the note that the pt hub
  genconfig already ships eos [1,106]*, buffered logs, 16s/step
  throughput ceiling, resume quirks, "the result itself is genuine").
  Feedback judge (503s, $1.41, 19 tools) did real forensics — read
  training code/loss/data/sample, re-checked every ckpt × decode combo
  ("all 0/30… no better artifact sitting unpromoted"), built a
  counterfactual ledger (~55 GPU-min of non-discriminating evals ≈
  0.35 epoch foregone), called out undeclared drift (2ep→1ep, 4→6+
  evals) — and its five priorities visibly steered iteration 2 (dev
  suite #1, 3 epochs/12K cap/FA2/lr 2e-5 #2, GRPO deferral #4 all
  executed). **R15-P2-1 4th recurrence:** the judge's first Read
  guessed `kapso_campaign/kapso_evaluation/evaluate.py`, errored,
  recovered in 1s. (*run 24's review recorded "pt ships NO
  generation_config" — contradicts this run's difficulty #2; reconcile
  offline, zero impact here since both runs shipped explicit [1,106].)
- **R28-P2-4 — OBS (mandate item: GRPO).** ~50 trace mentions, zero
  executed steps, $0 GPU. Iter-1 gated it (SFT ≥ baseline+7 — never
  opened); feedback #4 named the real reason ("needs non-zero reward
  density — at 0/30 there's almost no signal"); every iter-2 candidate
  + the synthesis carried the deferral explicitly. First run to weigh
  GRPO seriously, and the pass was the right call at a 0/30 floor.
- **R28-P2-5 — OBS (truncation economics on gemma).** Iter-1's ≤8K
  data cap backfired — it rejected 11,257 long traces, exactly the
  AIME-like stratum, leaving a 79%-generic-olympiad corpus (183
  amc_aime rows) — the feedback's root cause #3. Eval side: ~8/30
  generations rambled to the 16K cap. Iter-2 fixed the data half
  (2,500-11,000 tok band, max_seq 12288 > measured max 11,345); the
  16K eval cap never became binding because scores never left zero.
- **R28-P2-6 — OBS (iteration-2 conduct, two honest deviations).**
  (a) Plan said "exactly ONE official eval"; two ran (4ep 17:08, 3ep
  17:32) — the second priced the shipped artifact after the dev flip;
  declared, ~12 GPU-min, defensible. (b) Pre-registered selection key
  was "AIME-24 pass@4 primary, MATH-500 tiebreak"; on coverage@4 4ep
  actually wins (2/30 vs 1/30) — the agent framed pass@1 as tied and
  let MATH-500 decide for 3ep. Right on the merits (2 successes in 120
  is pure noise; MATH-500 n=100 is the only instrument with
  resolution; official metric is pass@1) but it IS a quiet inversion
  of its own key. Both officials read 0/30, so material harm nil.
- **R28-P2-7 — OBS (iteration-2 hygiene).** Recon-before-commit
  exemplary (OMR field literals verified on 2K rows; "na" pass-rate
  band discovered dominant → fallback invoked as a declared
  deviation). Two vLLM-0.11 bugs cost ~25 min total, both root-caused
  in ≤2 min from logs: `generate()` dropped the `prompt_token_ids`
  kwarg (latent in iter-1's never-run val_aime24.py — it crashed the
  base dev-floor eval, whose orphaned 65GB EngineCore was PID-hunted
  and killed), and TQDM_DISABLE=1 → tqdm elapsed=0 → ZeroDivisionError
  inside vLLM's throughput display. Both memorialized to memory
  (vllm-011-eval-gotchas.md). Residual miss: the base MATH-500/AIME-24
  dev floor was never re-measured, so dev 41% has no in-run base
  reference (3ep-vs-4ep comparison unaffected).
- **R28-P2-8 — OBS (framework counts).** ScheduleWakeup 0;
  InputValidationError 0; 12 rate_limit_events, no stalls; 85
  task_notifications; one blocked fg sleep-chain (09:56:49) adapted to
  run_in_background in 6s. Trace-cut state clean: GPU 0 MiB, no
  lingering ML procs, final_model 8.1GB/2 shards, base-env load +
  1-token generate PASS, arch Gemma3ForConditionalGeneration, eos
  [1,106], OMR pool cached to shared_cache, memory files written.

LADDER: base 3.33 (1/30, unshippable) → ckpt-100 0/30 → ckpt-400 0/30
(insurance) → ckpt-600 {greedy 0/30, t0.6 0/30} (ship t0.6) →
[boundary] → exp2 OMR 4ep {dev M500 33%, aime24 cov@4 2/30; official
0/30} → 3ep {dev M500 41%; official 0/30} → **ship 3ep t0.6, expected
official 0.0**.

SOTA OUTLOOK: no path to the 2/30=6.7 record was ever visible — the
best artifact's AIME-24 coverage@4 is 1-2/30, i.e. true pass@1 sits far
below one problem on a 30-set, and the proven 3.3 is itself exactly one
problem (lottery territory at 4B). What the run actually bought: a
precise capability map (MATH-500 8×-ish over floor at 41%, AIME still
0), the gemma packaging/OOM/FA2 recipe set, an OMR pool + dev suite
cached for reuse, and strong evidence this cell needs a qualitatively
bigger data/compute swing (OpenThoughts3-scale) — or should cede its
re-run slot to cells with headroom. Expected official: 0.0 vs cell base
0.0 (the harness scores final_model; the artifact provably serves —
gemma did not reproduce the qwen run-end vLLM failure in-run).

Verdict: **continue** — ~45 min from a clean close at cut, integrity
spotless, honest-zero shipped over unmeasured lotteries twice. P3
watch: exp2 boundary artifacts (difficulties/judge), the official
final eval on a gemma artifact (25/26's 2/2 qwen serving failures vs
27's success — gemma is the tiebreaker datapoint), and the two
framework asks: the wasted-call turn-ender (R28-P2-2) and the R15-P2-1
wrong-cwd first-Read now at 4 sightings.

## P3 (close-out: 17:36Z trace cut → RUN_DONE 19:07Z → rescore)

Watch items from P2, resolved in order:

- **Exp2 boundary: clean.** Report filed 17:46:09 with 5/5 XML tags,
  stream closed 17:55; feedback judge (17:52) ran the orphaned-value
  audit across every artifact — resolved the `0.0-pending` 4ep
  insurance entry against its official 0/30, confirmed shipped-3ep is
  the right pick, and pinned the honest framing for any future
  attempt: **the fine-tune (0/30) is a regression below the in-run
  base measurement (1/30)**, so a successor's bar is "recover ≥1/30",
  not "beat 0". Difficulties chain dense again (vLLM 0.11
  `generate(prompt_token_ids=…)` removal + EngineCore corpse,
  TQDM_DISABLE ZeroDivisionError — both memorialized in-run).
- **R28-P3-1 — the gemma tiebreaker landed AGAINST the harness: final
  eval failed 9×, 4th occurrence, now model-agnostic.** In-run
  officials served this same artifact 7 times without incident; the
  post-solve harness eval context (different cwd/container flow) died
  with vLLM exit 1 on every retry (error.log confirms `Failed to
  start vLLM server`; also a cosmetic upstream bug-let there —
  `kill: failed to parse argument: 'Failed'`, a PID var holding an
  error string). P2's "gemma did not reproduce the qwen failure
  in-run" now reads precisely: the failure was never about the model
  or artifact — it is the harness's final-eval serving context.
  Rescore VM `ptb-rescore-192314` (us-east4-a) served the identical
  path first-try: **accuracy 0.0, stderr 0.0** in ~20 min. Mitigation
  is institutional and battle-tested 4/4.
- **Judges:** `no contamination detected` / `only allowed use
  detected` — integrated gpt-5.6-sol re-pin, third consecutive
  fully-integrated success.
- **R28-P3-2 — `<evaluation_script_path>` EMPTY in both final
  reports** (13:12, 17:44); the extractor lists the tag as present
  and downstream carries on off `evaluation_output`. Not the
  R15-P2-1 wrong-cwd shape, but the same field showing a soft
  contract (empty silently accepted). Fold into the R15-P2-1 fix:
  empty ⇒ treated as missing, fail loud or re-derive.
- **R28-P3-3 — session-limit near-miss; failover priority validated
  by 25 minutes.** 13 `rate_limit_event` notices all run, every one
  empty; zero "session limit". Exp2's CLI closed 17:55 — the shared
  token's exhaustion window opened 18:20 and starved all three GPQA
  runs booted from the same secret. This run escaped on schedule
  luck alone; the OAuth failover (adapter swap-to-recovery via
  `--resume`, commit 81ea0bb5) exists for exactly this and ships in
  the next asset build.
- **R28-P3-4 — SearchNode repr bloat, worst measured instance.**
  solve_out.txt is 37 MB; line 15058 (the end-of-run
  `[SearchNode(node_id=0 …)]` dump) is **35.4 MB — 96% of the
  artifact** (rest of the trace: ~1.5 MB). The documented
  runner.py:344 fix (summary repr) graduates from cosmetic to
  forensics-hazard; any untruncated grep over the trace detonates.
- **Cost close-out:** $95.66 total (implementation 89.84 / ideation
  3.07 / feedback 2.50 / llm_backend 0.24) — campaign-high per-run
  cost, and the ledger says why: R28-P2-2's 188-wasted-call session-0
  churn ($73.47 for exp1 vs $10-15 for the better-behaved exp2 at the
  same model). The dedup-guard-ends-turn ask is now backed by a
  dollar figure.

Cell disposition: **done, no relaunch.** Proven #1 is one question
(3.3); our best artifact's AIME-24 coverage@4 was 1-2/30, so a retry
buys a lottery ticket, not a row. The run's real yield: the gemma
packaging/OOM/FA2/processor-config recipe set, the OMR pool + dev
suite in shared_cache, and the regression framing — spend future
slots on GPQA (headroom cells) instead.
