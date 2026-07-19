# Run #17 review — arenahardwriting × Qwen3-4B-Base (07181341)

First run on the rebuilt stack (lifecycle batch 580a74a3 + relaxed rules
3ba24c4d). Dual-mandate reviews per
`arena-best-baseline-traces.md` §Review protocol; best-known trace for this
cell: fable-5 86.24 (local Qwen3-30B-A3B-Instruct-2507-FP8 teacher).
Launched 2026-07-18 13:43Z.

## P1 (t+0 → t+86min)

Headline: relaxed rules semantically effective — both fable-5 ideation
candidates proposed local Qwen3-30B-A3B-Instruct-2507 teacher distillation,
codex proposed Gemma-27B distillation; no self-censoring anywhere. Selector
synthesized the 86-tier skeleton (teacher gen → exact-template SFT → baked
generation_config temp 0.7/top_p 0.9/rep 1.05/eos [151645,151643]) with a
gated optional BoN→DPO stage 2.

- **R17-P1-1 — P1 major (recipe), 13:50–14:02.** Nobody measured the eval's
  language distribution; multilingual slice is an unmeasured ~20% guess vs
  the eval's actual ~33% non-English ("keeping ~20% non-English to match
  the multilingual arena distribution"; pool `writing=7705 general=1495
  ml=2300 total=11500`). Also 8k prompts vs the 86-tier's 21k. Correctable
  in iteration 2 — carry to P2 watch.
- **R17-P1-2 — P2 (framework/stream), 13:44–13:53.** Codex ideation
  member's output NOT streamed — only start line + "candidates=2/2
  (dropped 0), 510s, ok"; its candidates auditable only via the selector's
  summary. Observability gap vs the expected members-streamed behavior.
- **R17-P1-3 — P2 (recipe watch), 13:54:51.** Selected plan carries a
  Stage-2 BoN→DPO preference stage (lost in every best trace). Mitigated:
  gated "only if ≥3h remains", SFT promoted first, promote-only-if-beats.
  Watch the <1h time-box in P2/P3.
- **R17-P1-4 — P2 (framework, wasted steps), 13:44:53.** Fable-5 member
  burned 3 ToolSearch calls hunting research tooling ("No matching deferred
  tools found"; "The wiki/research tools aren't exposed in this session").
  Ideation prompt/tool-surface mismatch; trivial cost.
- **R17-P1-5 — OBS, 14:03→14:08.** Baseline eval launched then killed 5 min
  later ("blocking the GPU, which is my critical path") — sequencing
  inefficiency, self-corrected; baseline floor unmeasured at segment end.
- **R17-P1-6 — OBS, 14:57→15:09.** First SFT (6,771 ex, 2ep) rambles
  long-form; limit-50 eval killed pre-score ("failing to emit <|im_end|> …
  format-drift failure mode"); root-caused to short teacher data ("median
  350, only 1.2% capped … under-learned stopping") → safety-net promote
  (score 0.0 "sft-stage1-safety-unverified", generation_config.json
  included) + harder retrain planned; promote.py checkpoint bloat fixed
  (53G→7.6G). Healthy forensics; no score yet at t+86m.
- **R17-P1-7 — OBS (framework, positive), 14:02:52.** env_defaults clock
  policy active and non-disruptive (single blocked `sleep 200` guided to
  background pattern; no 120s auto-background surprises). Boot clean; both
  members 2/2 candidates; implementation session init clean.
- **R17-P1-8 — OBS, 13:45:06.** Repo-memory core.architecture header-only
  stub (fresh campaign, expected). R15-P2-1/R16-P2-2 not exercisable yet
  (no iteration boundary in segment). Selector verified claims against
  evaluate.py / judge_utils.py / qwen3.jinja before deciding.

Verdict: **continue** — winning recipe class at iteration 1, strong tempo
(teacher gen + SFT + diagnosis inside 86 min); material deviation is the
unmeasured/undersized multilingual slice (R17-P1-1).

## P2 (15:09Z → 18:30Z)

Headline: ladder 0.8551@50 → full-250 **0.8697** on final_model itself
(≈ best-trace's 0.8704 self-eval); DPO tried and correctly rejected inside
54 min; upside-only v3 in flight with promotion bar ≥87.5%. No session
boundary in segment (ends ~18:55), so technical_difficulties / feedback /
R15-P2-1 checks remain unexercisable.

- **R17-P2-1 — OBS (recipe), 15:41→15:59.** R17-P1-6 closed: retrain
  (3ep, 2×LR) fixed stopping — pre-eval diagnostic "stopped=12/12
  capped(2048)=0" at temps 0.4/0.6/0.7; v2 "Score (winrate) is: 0.8551"
  @50 promoted 15:50:45 over the 0.0-scored safety net (justified
  upside-only exception to the ≥150q rule); temp-0.5 ablation 0.8519 →
  kept 0.7.
- **R17-P2-2 — OBS (recipe), 15:59:41→16:53:16.** R17-P1-3 resolved
  cleanly: Stage 2 BoN(1500×6, temp 1.0)→teacher-judge (1,264 pairs)→DPO
  (~10 min train) = 54 min total, inside the <1h box and the ≥3h gate;
  0.8524 vs 0.8551 → "I **keep the SFT model**". Matches best-trace
  evidence that preference stages don't pay here.
- **R17-P2-3 — OBS (recipe, positive), 17:20:27→17:24:28.** Full-250
  confirm 0.8697 run against final_model directly (vLLM serving the
  deliverable doubles as a load test); separate transformers load check
  (4.7s, 4.02B params, clean stop); result locked before iterating.
- **R17-P2-4 — P2 (recipe), whole delta.** Upside iteration v3 = more
  teacher data (3,000 responses on held-out PREF prompts → 9,328 ex, 3ep),
  NOT the language-mix fix: zero mentions of language/multilingual in
  3.4k delta lines — R17-P1-1 (~20% guessed vs ~33% actual non-English)
  will close out the run unaddressed. The one known systematic gap was
  not the lever chosen.
- **R17-P2-5 — OBS (recipe, discipline), 17:41→18:15.** Exemplary
  end-game hygiene: v3 promotion bar set above noise ("promote v3 only if
  it reaches ≥87.5%, else keep the twice-verified 86.97%"); recovery
  instructions written to PLAN in case the session ends mid-v3; v3 eval
  writes a separate metrics file "so it doesn't clobber the deliverable's
  result.json".
- **R17-P2-6 — P3 (framework, waste).** Watcher hygiene: superseded
  sleep-watchers never cancelled → 5 stale wakeups (16:55:49–17:53:08,
  "That was a stale notification from the earlier DPO wait"), each
  burning a turn; plus 133 reflexive Reads of just-launched empty task
  outputs ("shorter than the provided offset (1)"). Cost $125.09 by 18:30
  vs $39.89 for the whole best-trace run. The launch hint "To check
  interim output, use Read on that file path" invites the reflex.
- **R17-P2-7 — OBS (framework, positive).** Notification machinery
  healthy (run-18 contrast): all four >10 min silent gaps (max 16.0 min,
  16:58:53→17:14:55) ended by a real [system:task_notification]; zero
  blocked sleeps or auto-background surprises in the delta.

Verdict: **continue** — 0.8697 verified on disk with a protected
upside-only v3 pending (result ~18:40, ~15 min margin to the 18:55
deadline; promote is a fast file op).

## P3 (18:30Z → end) + closing

Headline: OFFICIAL **0.8964 ±0.009, both judges clean — CELL RECORD**
(+3.40 over fable-5's official 86.24, +2.80 over human 86.84). Answer to
the P3 question: **v4 did NOT beat the bar** — full-250 "Score (winrate)
is: 0.8781934306569343" (21:34:07) vs bar >88.85 → "Per
promote-only-if-better, **v4 is NOT promoted; final_model stays v3**"
(21:35:36). The record is the **v3 weights with a hill-climbed decoding
default (temp 0.7→0.6)**: self-evals 89.84 (source ckpt, 22:05:28) and
91.04 (final_model confirm, 22:34:20, stderr 0.85); official 89.64 sits
inside the demonstrated 1.2-pt run-to-run band and above both temp-0.7
readings (88.64/88.85) — the decoding gain held out-of-sample.

- **R17-P3-1 — P1 (agent near-loss), 18:36:07→18:43:55.** Session 1
  killed the v3 eval at 18:36:11 ("too slow … unlikely to beat the
  verified 86.97%") — 7 min before its own ~18:43 ETA and 12 min before
  its own 18:48 cutoff. The kill no-oped ("gpu already free" =
  generation had just finished); API-side judging completed 18:41:30 →
  v3_metrics.json 88.64. Its own completion watcher then fired
  (~18:41:42) and the agent — conditioned by R17-P2-6's stale-wakeup
  noise — dismissed it four times: "Stale notification from a leftover
  polling task — no action needed", shipping v2 86.97 with v3's 88.64
  sitting on disk. Feedback judge named it: "misclassified the v3
  eval-completion background notification as a 'stale notification' …
  ~1.7 points were left on the table."
- **R17-P3-2 — P2 (framework, capture chain), 18:38:39→18:45:19.** The
  agent DID author a complete final report at 18:38:39 (score 86.97 +
  7-item technical_difficulties) but it rendered on the thinking
  channel, and six stale-notification micro-turns followed it, so the
  CLI's terminal payload was a 111-token brush-off ("Finished … 2+111
  tokens"). Extractor: "Could not extract result from agent output" /
  "using raw output" / "technical_difficulties missing — running
  fallback reconstruction". Primary capture path failed on a session
  that had authored everything; fallbacks caught it. Fix candidates:
  cancel pending watchers once a final report is emitted; extract tags
  from the whole stream, not the last payload.
- **R17-P3-3 — OBS (framework, positive): the boundary rescued the
  run.** Fallback reconstruction (18:45:19→18:53:03, $2.32) truthfully
  opened "The session ran to a self-declared completion (not a crash)"
  — correct: NO deadline kill anywhere in the run; session 1's CLI
  exited naturally at 17,344s of the 18,000s cap — and flagged "***
  BIGGEST LOSS *** — SUPERIOR v3 MODEL (88.64%) DISCARDED". Feedback
  judge 1 (18:53:03→18:57:37, $1.13) independently verified artifacts
  (read v3_metrics.json 18:55:29), scored the node truthfully **0.8697**
  (what final_model held, not the orphan), and issued the exact ACTION:
  "copy sft_run3/ckpt → final_model verbatim … so 88.64% reproduces"
  plus the never-dismiss-notifications lesson. Blemish: reconstruction
  called the 18:55 deadline "false/binding" — it was the real 18000s
  session cap (13:54:51→18:54:51); its substance (kill premature,
  promote would have fit) stands.
- **R17-P3-4 — OBS (recipe), 18:57:37→19:06:02.** Iteration-2 boundary
  clean: ensemble again (codex + fable-5, 4 candidates pooled), full
  experiment-0 feedback rendered through to members (rule 6 — the
  winning candidate quoted the bank ACTION verbatim). Selector picked
  Candidate 3 (bank v3 + best-of-3 teacher-BoN → v4 14-15k ex, bar
  strictly >88.64 full-250), folded Candidate 4's free decoding
  hill-climb in as gated step 9, and step 10 carries "Never dismiss a
  pending background notification".
- **R17-P3-5 — OBS (recipe), 19:06→21:36.** Disciplined execution:
  banked v3 at **19:09:00** (3 min in; "88.64 sft-v3-promote"),
  reconfirmed final_model **88.85** full-250 (19:36:50). Discovered v3's
  training data hadn't survived the branch handoff → justified
  deviation at 19:11:44: "best-of-3 on a full ~15k pool is infeasible
  (~150+ min teacher time)… single-sample distillation at larger
  scale". v4 = 15,500-prompt pool (ml 3,100/15,500 = 20% — R17-P1-1's
  guessed mix, never fixed, closes out unaddressed) → 12,785 ex,
  train_loss 0.855, stop-diag 12/12, full-250 **87.82** → rejected. No
  DPO retry (time went to the decoding sweep instead — correct EV).
- **R17-P3-6 — OBS (recipe): the record lever, 21:36→23:03.** Protected
  decoding hill-climb with pre-registered gates: ship temp 0.6 only if
  ≥89.35 (21:38:45); source-ckpt eval **89.84** (22:05:28) → shipped
  22:06:39; confirmation on final_model itself **91.04** (22:34:20);
  temp-0.5 probe tied 91.04 → "TIE-kept-temp0.6" (23:03:13). Shipped
  config: temp 0.6 / top_p 0.9 / rep_pen 1.05 / eos [151645,151643].
  best_score.log ladder pristine: 85.51 → 86.97 → 88.64 → 88.85 →
  (87.82 not-promoted) → 89.84 → 91.04.
- **R17-P3-7 — OBS (framework, positive; R16-P2-1 non-regression).**
  Session-2 endgame textbook: FINALIZE commit e83c257, result.json
  91.04 with provenance notes, integrity check (4.02B params, loads in
  base env), all-5-tag XML report in the final payload (1310+9016
  tokens) incl. self-authored 5-item technical_difficulties
  (lost-training-data, v4 negative result), extraction succeeded
  ("Extracted agent result from XML tags"), CLI exited the same second
  — zero trailing turns. Feedback judge 2 ($0.97) added the prophetic
  winner's-curse caveat (1.2-pt swing on identical weights — official
  89.64 landed inside it), scored 0.9104, stop=false with sane
  priorities; orchestrator truthfully "Stopping: finalization reserve
  reached" at 23:10:22; consolidation confirmed final_model present.
- **R17-P3-8 — P3 (framework, systemic).** (a) Wakeup corroboration for
  R18-P2-1/R19-P2-1: ZERO "wakeup" occurrences in the full 19,381-line
  trace (13:44→23:10) — every wait is a bash sleep-watcher + task
  notification. (b) R15-P2-1 recurred mildly: feedback judge 2's first
  Read resolved the relative evaluation_script_path against the
  campaign cwd ("File does not exist", 23:06:21), self-recovered in
  ~10s. (c) Poll-churn persists at lower cost: 159 "Wasted call"
  re-reads + 37 empty-offset reads in session 2 (vs 90+176 in session
  1), but $44.98/356 tools vs $147.49/697. Run total **$196.71** vs
  best-trace $39.89 — the record cost 5×, dominated by session-1
  watcher churn.

CLOSING VERDICT: Official **0.8964 ±0.009 (both judges clean)** — the
arenahardwriting × Qwen3-4B-Base cell record, +3.40 over fable-5's
official 86.24 and +2.80 over the human 86.84. Deliverable: SFT v3 —
Qwen3-4B-Base full-FT from base, 3 epochs on 9,328 exact-eval-template
examples distilled from the local Qwen3-30B-A3B-Instruct-2507 teacher —
served at temp 0.6. The two decisive differences vs the fable-5 86.24
recipe: (1) the **two-iteration structure with an artifact-inspecting
boundary** — the fallback reconstruction + feedback judge recovered an
orphaned +1.7-pt checkpoint that a single-session run would have lost
outright (86.97 would have merely tied the old record); (2) the **gated
decoding-defaults hill-climb** (temp 0.7→0.6, +1.5–2 genuine on 2×2
full-250 runs) — a free lever no prior trace exploited, reachable only
because iteration 2 started from a banked 88.64 instead of from scratch.
Underneath both: full-250 promote-only-if-better discipline that banked
every gain and rejected every regression (v4 87.82, temp-0.5 tie).
Framework health at the close: the boundary fallback chain and feedback
judges carried the run — the campaign's first material framework save;
the two fixes this run argues for are the primary result-capture path
failing under trailing stale notifications (R17-P3-2) and the
watcher/poll churn that both caused the near-loss and burned ~$100
(R17-P3-1, R17-P2-6). Wakeup timers remain absent systemically; endgame
discipline at natural completion is fully intact. Record banked; the
bank-first / decoding-sweep / variance-bounding lessons are in the store
for the next cell.
