# Run #18 review — arenahardwriting × Qwen3-1.7B-Base (0718134)

Rebuilt stack (580a74a3 + 3ba24c4d). Dual-mandate reviews per
`arena-best-baseline-traces.md`; best-known trace for this cell: opus-4.8
74.85 (cached Qwen3-4B-Instruct-2507 teacher, 19-language self-instruct).
Launched 2026-07-18 13:43Z.

## P1 (t+0 → t+78min)

Headline: BOTH ideation members proposed local teacher distillation
(fable-5: Qwen3-30B-A3B-Instruct-2507 SFT backbone; codex: 30B-FP8 + DPO);
all four candidates priced the teacher as a DOWNLOAD (61GB→FP8 31GB) —
nobody stumbled on the core-scope cache. Teacher downloaded in ~2.5 min;
generation_config baked into the FIRST artifact (t0.7/p0.8/k20/rep1.05,
eos [151645,151643]) plus an 8/8 stop-rate smoke test; floor eval already
running at t+76min — ~90 min AHEAD of the 74.85 trace, which only pivoted
to distillation at t+2h31m. Baseline 0.0 confirmed. Fable-5 argued
legality unprompted ("Local open-model inference for ranking is explicitly
permitted").

- **R18-P1-1 — P1 major (recipe), 14:40:37.** No one measured the eval's
  language distribution; multilingual slice fixed by assumption at ~20%,
  landed at 16% ("total 8049 | multilingual 1288 (16.0%)", aya-sourced,
  not test-weighted) vs the eval's ~33% non-English. The 74.85 recipe's
  own verdict: "more balanced multilingual distillation was the main
  lever." Cheap fix next iteration.
- **R18-P1-2 — P2 (framework/stream), 13:55:12.** Codex member output not
  streamed (only "candidates=2/2 … 482s"); its candidates reconstructable
  only from the selector's critique. RECURRING (= R17-P1-2).
- **R18-P1-3 — P2 (framework/selector), 13:56:03.** Selector claimed
  "every factual claim checked out" while its own probe printed "baseline
  Qwen3-1.7B answers: 250 median token_len: 0" — a degenerate value (wrong
  field) it never questioned; C3's "median 632 tokens" went unverified.
  Selection outcome unaffected.
- **R18-P1-4 — P2 (recipe deviations), 14:41:26.** SFT launched 2ep/lr
  1e-5/maxlen 2048 vs the 74.85 recipe's ~3ep/2e-5 (and the plan's own
  3072); WildChat-1M gated (no HF token in-run, 14:16:12) forced source
  swap to OpenHermes/no_robots/writingprompts/aya. Phase-2 DPO (~2h)
  planned but floor-gated and promote-only-if-better.
- **R18-P1-5 — OBS, 13:47–13:50.** Five empty ToolSearch probes at fable-5
  member boot (tools not exposed in ideation). RECURRING (= R17-P1-4).
- **R18-P1-6 — OBS (framework, positive), 14:11:47.** env_defaults clock
  policy live ("Blocked: sleep 30 … use Monitor"); agent adapted with
  Monitor/ScheduleWakeup. No auto-background surprises.
- **R18-P1-7 — OBS, 14:41:25.** Agent momentarily read wrapper-PID exit as
  "SFT process exited", self-corrected to the python PID within 12s.

Framework otherwise clean: boot checks pass; 2/2+2/2 candidates; selector
synthesis sound (C3 backbone + C2's FP8 teacher + C4's DPO minus a Skywork
download); implementation opened with date -u + timer (9:49 left);
registered-eval wrapper used. R15-P2-1/R16-P2-2 not observable (no
feedback cycle yet).

Verdict: **continue** — winning recipe class at iteration 1, ~90 min ahead
of the best-known trace; push the multilingual mix toward the eval's
actual distribution in iteration 2 (R18-P1-1).

## P2 (15:01Z → 15:52Z + whole-session wakeup forensics; 3h hang incident)

Headline: CLI silent from 15:52:31Z to the ~18:54Z session deadline. Root
cause is two-layer: the ScheduleWakeup timer is DEAD (0/7 fires this
session; cross-run: 0/8 in run #19 — systemic, = R19-P2-1), and from 15:28
the agent's watcher tasks were unfinishable `until ! pgrep -f "<pattern>"`
loops whose own bash -c cmdline matches the pattern (agent-side self-match
— cousin of the old self-pkill class). At 15:52 the only pending wake
sources were {2 unfinishable watchers} + the dead timer → nothing could
ever re-invoke the model.

- **R18-P2-1 — P1 framework-major.** Wakeup forensics: 7 ScheduleWakeup
  calls, 0 fires (three unambiguously no-fired with empty windows, masked
  by late task-notifications; 16:03 was the first with no rescuer). All 6
  actual re-invocations were task-notification-driven. 19 background
  tasks, 29 notifications, 9 clean `[result]` turn-enders — the fatal one
  shape-identical to the prior 8 (Read → Edit PLAN.md → git commit
  "Iter2: launch richer v2 teacher" → ScheduleWakeup 16:03). Consequence:
  v2 teacher generation finished on disk (~16:15) but its chained filter
  never ran; ~3h GPU idle to the deadline kill.
- **R18-P2-2 — P2 (recipe), 15:47:45.** v2 launched "reusing the same
  pool" (8,977 prompts, 16.8% multilingual) — R18-P1-1 (~1/3 non-English)
  still unaddressed; v2 targets length only.
- **R18-P2-3 — OBS (positive).** Floor secured pre-silence: 0.1706 @50
  promoted at 15:24:42 (`[promote] … score=17.06, sft-floor-teacher-
  distill`), best_score.log written. Judge-text forensics used well (now
  rules-legal): 31 clear losses → "(1) insufficient length/completeness,
  (2) repetition/thin multilingual quality" → v2 first-chunk median 643
  tok vs v1's 367, matching the baseline's 632.
- **R18-P2-4 — OBS.** v1's 0.17 rung sits far below the best trace's
  first distill rung (0.4732): short v1 responses, not teacher strength,
  are the gap. Framework hygiene otherwise clean; agent session-aware.

INHERITED STATE for session 2: final_model = v1 @ 17.06 (safe deliverable);
teacher_raw_v2.jsonl complete on disk (median ~643 tok); PLAN.md committed
with diagnosis + v2 plan → unambiguous ~2h finish (filter → SFT → eval →
gated promote) inside session 2's budget.

Verdict: boundary self-healing should recover the run — the hang cost
~3h idle GPU, not state. Watch session 2 for: deadline-kill end-mode
labeling (truthful?), difficulties FALLBACK (first live firing — no
self-authored tag possible), feedback judge behavior (R15-P2-1 check),
and whether v2 finally fixes the multilingual mix.

## P3 (18:57Z → end) + closing

Headline: OFFICIAL 0.4240 ±0.016, both judges clean. The boundary worked
exactly as designed: deadline kill truthfully labeled 18:57:02 ("Deadline
of 18000s reached — terminating Claude Code"); difficulties FALLBACK fired
18:58:17 on the missing tag (first live firing, $1.93/403.6s/25 tools) and
its output drove the whole recovery; session 2 (CLI 19:14:27→23:16:43,
14536.4s, $21.16, 228 tools) cashed in the stranded v2 asset within 41 min
and climbed 17.06 → 38.23 in-run. Campaign closed by the orchestrator at
23:20:06 ("finalization reserve reached"), cumulative $43.03.

- **R18-P3-1 — P2 (framework/fallback forensics).** The fallback's 6-item
  reconstruction is operationally excellent — it pinned the stranded state
  ("`sft_data_v2.jsonl` was **never created**", "GPU sat idle from 16:04
  to at least 19:01 (~3h)", "**Unresolved** — final_model … 17.06% floor
  … is valid"), quoted the exact offending watcher verbatim (`until !
  pgrep -f "generate_teacher.py.*teacher_raw_v2"; do sleep 45; done`
  with its 0-byte output log), and #5 nailed the recipe lesson (concision
  target undershot the judge-rewarded ~632-token register; v2 at 643 "was
  **never trained or scored**"). But the hang MECHANISM is misattributed:
  it never named the watcher self-match trap, claimed the 16:03 wakeup
  "fired" and read the 18:57 kill-flush init as "the wakeup-driven
  re-invocation [that] did no work" — blaming the agent, when P2 forensics
  show 0/7 wakeups ever fired (dead timer) and the loop was unfinishable
  from birth. The behavioral advice downstream was right anyway, but the
  permanent lesson artifact gives the framework's dead timer (=R19-P2-1)
  zero reinforcement.
- **R18-P3-2 — P2 (framework/feedback).** Node-0 judge (163.7s, $0.66):
  independently verified untampered eval + valid model, banked
  <score>0.1706</score>, stop=false, and item 2 ("Cash in the
  already-completed v2 work FIRST … ~40 min") set session 2's opening
  move. Same mechanism garble though — "killed by its own ScheduleWakeup
  deadline" (twice) conflates the harness clamp with the wakeup API.
  R15-P2-1 RECURRED at BOTH boundaries: node-0 first probe ran in the
  campaign root and missed final_model/evaluate.py (19:05:15, exit 2,
  recovered 9s); node-1 first Read of `kapso_campaign/kapso_evaluation/
  evaluate.py` → "File does not exist" (23:18:05). Deterministic wasted
  first step, four sightings campaign-wide now.
- **R18-P3-3 — OBS (framework, positive): the lesson chain closed.**
  get_top_experiments served the fallback's difficulties IN FULL to both
  ideation members (rule 6 verified in the render); both fable-5
  candidates encode the fix ("no ScheduleWakeup chains, no fire-and-forget
  continuation (this exact pattern killed iteration 2 of the prior
  session)"; "bounded ≤5-min polls … kill by recorded PID only") and C3
  found two REAL silent length gates by reading repo code
  (`filter_data.py --max-tokens` default 1400 vs v2 p90 ≈ 1481;
  `train_sft.py --max-length` 2048 silently skipping long examples) —
  both load-bearing: the v2 filter ran at 1900 and training at 3072 kept
  8120/0-skipped. Selector synthesis: C3 backbone + C4's conditional DPO
  + C2's length-ratio guard. Recurring noise: codex member again not
  streamed (=R18-P1-2), 3 empty ToolSearch probes at fable-5 boot
  (=R18-P1-5).
- **R18-P3-4 — OBS (framework, positive): watcher pattern fixed, session
  freeze-free.** Session 2 made ZERO ScheduleWakeup calls (dead timer
  routed around, not fixed) and every watcher became a bounded,
  file-condition loop — `for i in $(seq 1 90); do if grep -qE "Score
  \(winrate\) is:" …` / `until grep -qE "\[dpo\] saved to|Traceback|CUDA
  out of memory" …` — self-terminating, failure-aware, and structurally
  incapable of pgrep self-match. All long waits ended in task
  notifications that re-invoked the CLI (longest quiet stretch 55.7 min,
  22:19→23:14, ended by the @150 monitor firing); the one monitor timeout
  (21:30, @100 eval slower than its window) returned control and was
  hand-checked — bounded-by-construction working as intended. env sleep
  guard fired once (19:17:32) and was respected.
- **R18-P3-5 — OBS (framework, positive): endgame + exit clean, no
  R16-P2-1.** result.json written BEFORE the risky @150 confirmation
  (22:17:01), final_model integrity proven by full transformers load
  (1,720,574,976 params), best_score.log appended, GPU verified 0 MiB,
  final commit 3871354, then the 5-tag result — technical_difficulties
  SELF-AUTHORED this time, 8 evidence-grade items (two workspace roots;
  HF_HOME at /home/ben/hf_cache nearly aborted DPO; nohup block-buffering
  broke "wait for first loss line" → watch checkpoint dirs instead;
  `round(lr,4)`→0.0 false alarm; promote.sh `cp -r` dragged 23G of
  checkpoints into final_model; temp-0.5 evals ~20s/answer; DPO reward
  acc 0.51 inherent to single-model on-policy pairs; benign tokenizer
  warning). CLI exited AT the final message (23:16:43 = Finished
  23:16:43); extraction recovered all 5 tags; node 1 banked 0.3823,
  stop=false, evaluation_valid=true.
- **R18-P3-6 — P1 recipe-major: R18-P1-1 never corrected.** Session 2
  reused the 8,977-prompt pool untouched (~17% aya-sourced multilingual
  after filtering) vs the eval's ~33% non-English; no test-weighted
  language mix, no self-instruct prompt expansion, one distill rung
  (8.1k) vs the 74.85 trace's three (8.4k→13.4k→18.2k, 19 languages
  weighted zh 260/ru 260/…), generic `--rich` teacher prompt vs its
  rubric-shaped one, and R18-P1-4's 2ep/1e-5 persisted into v2. The
  74.85 ladder prices what that leaves on the table: its balanced-
  multilingual scale rung alone was +19 (0.4732→0.6638). Node-1 feedback
  aims the next run at ranker strength and corpus diversity but never
  names the language-mix share explicitly.
- **R18-P3-7 — OBS (recipe).** Session-2 ladder, all timestamped: v2 SFT
  0.3457 @50 (scored 19:55:32, promoted 19:55:59, "sft-v2-teacher-rich")
  → decoding grid on identical weights: temp 1.0 = 0.0810 (20:10), 0.5 =
  0.3778 (20:33, promoted 20:33:39), 0.3 = 0.3138 (20:56) — sweet spot
  found by trend-following, not the planned fixed grid → @100 confirm
  0.3696 (21:35) → DPO (1200×4 on-policy samples, 504 pairs after the
  length-ratio guard dropped ~58%, ~32 steps) = 0.3804 @50 (22:15),
  reward accuracies 0.51 → correctly NOT promoted ("promoting DPO on
  noise risks a silent regression") → @150 confirm 0.3823 ±0.021
  (23:14:53). Official 0.4240 vs in-run 0.3823: +4.2 official-vs-dev
  skew (full 250 + official judge pair vs gpt-5-mini @150) — worth
  remembering when calibrating in-run numbers against leaderboard rows.

CLOSING VERDICT — **official 42.40 ±0.016, both judges clean**, banked as
the cell's sixth official result. Placement: below the opus-4.8 cell mean
(45.0), human row (50.0), and fable-5 (57.1); ~32 points under the 74.85
best-known trace, with the gap fully accounted for by the missing
multilingual/self-instruct scale rungs (R18-P3-6). The hang's measured
cost: 2h53m idle H100 (v2 teacher done 16:04:03 → kill 18:57:02) plus ~17
min of boundary machinery — almost exactly one distill-scale rung of the
74.85 ladder, i.e. the difference between ~42 and plausibly ~60. What
session 2 proved: boundary self-healing WORKS end-to-end — kill labeled
truthfully, fallback reconstructed the state sight-unseen well enough
that ideation's first move was the stranded asset, and the recovered
session ran freeze-free with the watcher pathology structurally
eliminated, doubling the score in 4h02m with zero restart waste. The
open framework debts are unchanged and now precisely scoped: dead
ScheduleWakeup timer (R19-P2-1, routed around rather than fixed),
feedback-cwd path mismatch (R15-P2-1, four sightings), codex ideation
streaming (R18-P1-2). The recipe debt is one sentence: weight the
teacher pool to the eval's language distribution and scale it — the
best-known trace says that alone is worth more than everything else
this run left undone.
