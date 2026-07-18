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
