# Run #32 review — gpqamain × Qwen3-4B-Base (gpqamain-qwen3-4b-base-07252103)

First run on the failover stack (81ea0bb5: 3-token oauth file verified on-VM
— records IWBB8/Q_yac/JRM9t; fd-level streaming I/O). Launched 2026-07-25
21:03Z us-east4-a. Cell context: GPQA weight .2246 (2nd-heaviest), best
proven band ~30.5, human 36.2-row avg; watchlist headline = answer-format
parse failure (below-25%-random signature).

## P1 (t+0 → t+2.7h)

Headline: **the format hazard was confirmed live within the first half hour
— base Qwen3-4B scores 8.67% (13/150), far below random 25% — and by t+2.5h
the run had already diagnosed a wrong-logs-dir eval artifact, promoted a 24%
model, and pivoted to data scaling.**

- **R32-P1-1 — OBS (API instability, framework degraded gracefully).** The
  fable-5 ideation member died with `API Error: 500 Internal server error`
  at 21:17 (`under-delivered: 0 of 2 candidates`); the pool ran with
  codex's 2 candidates only. Same ~21:15-21:55Z window as run #33's 529
  implementation kill — Anthropic-side flakiness, not our stack. Warned
  loudly, no crash, selector proceeded.
- **R32-P1-2 — OBS (selector source-verification catches both candidates).**
  Both codex candidates assumed GPQA-**Diamond**; the selector verified at
  source: `gpqa_main`, HF `Idavidrein/gpqa`, 448 q, `--limit 150` stable
  order (±4 pts 1σ), `ANSWER: $LETTER` parser is MULTILINE **first-match-
  wins** (training completions must contain exactly ONE such line),
  qwen3.jinja adds no forced `<think>`, choices shuffled. Third consecutive
  run where selector source-reading corrected member assumptions.
- **R32-P1-3 — OBS (time-fit judging).** Selector rejected C1 (7-9.5h
  monolithic full-FT, "nothing bankable until 2 epochs finish") for C2's
  staged plan with a phase-one bankable checkpoint — the session sizing
  rule doing its job at selection time.
- **R32-P1-4 — OBS (eval-wrapper artifact, self-diagnosed in 6 min).**
  Phase A scored byte-identical to baseline (8.67%, 13/150) → agent flagged
  "red flag, not coincidence" → found its wrapper read the stale baseline
  log from the wrong `logs/` dir; the fine-tune HAD run (230k output tokens
  vs 86k baseline). Correct re-read gave the real Phase A score; promoted a
  24% model by 23:45.
- **Watcher discipline**: notification-driven waits (`await the b6i2o8owc
  notification`), no read-poll churn observed in the sampled windows.
- Zero session-limit events; the 13+ `rate_limit_event` notices are the
  usual empty CLI lines. Failover not yet exercised.

State at cut: 24% model promoted (≈3× base, 1 pt under random floor — the
format fix landed, knowledge lift is next); curating a relaxed-cap
short-trace corpus (~35k examples; tight cap had dropped 26.5k valid) and
auditing non-ASCII garbage leakage in completions.

## P2 (t+2.7h → t+6.5h)

Headline: **iteration 1's 24-28% promotions did NOT hold at n=150 — the
official iteration score landed at 16.0% (below random), and the boundary
audit caught the gap honestly** ("No mis-promotion: final_model=Exp4=16.0
@150 IS the best banked artifact; Exp3=14.67 worse; the 24.0/28.0 in
best_score.log were smaller-n reads"). The subset-vs-150 noise trap (AIME's
n=30 lesson at GPQA scale: ±4pts 1σ at n=150, worse on subsets) is now this
run's central hazard.

- **R32-P2-1 — OBS (iter-2 boundary quality).** Ideation harvested the dead
  session's workspace facts/risks explicitly (prev session's jsonl files
  lived on ephemeral /tmp — flagged as a risk), and the new plan front-loads
  a bankable Phase A (1 epoch over a 24k mix, kill-gates included).
- **R32-P2-2 — OBS (buffered-log lesson applied).** At 03:25 the training
  log went silent ~9 min; the agent checked GPU (97%/629W = training alive),
  identified Python block-buffering as the cause, and switched to
  trainer_state.json for ground truth — run 28's difficulty #3, learned
  cross-run via the difficulties chain, applied without a wasted restart.
- Training healthy at cut: step 150/509, loss 0.83→0.53, token-acc
  0.79→0.83; ~3.5h session time left.
