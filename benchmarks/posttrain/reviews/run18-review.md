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
