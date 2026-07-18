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
