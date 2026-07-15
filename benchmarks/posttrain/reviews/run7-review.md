# Run #7 review log — gsm8k × Qwen3-1.7B-Base, 3h, flex-start

**OFFICIAL RESULT: 53.37% ± 1.37 accuracy, full 1319-problem GSM8K test set**
(rescored via 40_eval_only.sh after fixing F10; the agent's in-run estimate
was 58% on a subsample). Reference points: base model ≈10-20%; the public
leaderboard's GSM8K column across 10-HOUR runs tops out at 55.9 (GPT-5.2)
with Opus 4.6/Claude Code at 41.0 — those cells average 4 base models, so
not exactly comparable, but a 3-hour kapso run landing at 53.4 on its first
functional attempt is a strong signal.

Live review of the kapso loop under the newly applied fixes. Monitored every
10 minutes; each finding records evidence (trace/checkpoint), severity, and a
suggestion. Written as the run progresses.

## What this run validates

| Fix under test | Pass signal |
|---|---|
| Ideation salvage (framework) | a deadline-killed ideation yields a "Salvaged from a deadline-terminated ideation session" solution, not the generic fallback |
| Parent fallback ladder (framework) | with no scored node, iteration N+1 branches from the latest committed experiment branch, not `main` |
| Budget-proportional caps (harness) | runner prints `session caps: ideation=1620s implementation=8100s`; no phase eats the run |
| Native finalization reserve (harness) | loop stops with `finalization_reserve` semantics, consolidation completes inside the guard |
| PLAN.md discipline (handler prompt) | PLAN.md appears in the workspace early in the first implementation session |
| HF hub timeouts + warm cache (prior fixes) | no multi-minute silent stalls on dataset/model downloads |

## Run parameters

- Budget: 3h harness timer → ~175 min orchestrator budget (5 min guard),
  finalization reserve 15 min (hmm: 0.10×175=17.5 → capped 15)
- Session caps: ideation 27 min, implementation 135 min
- Model: claude-opus-4-6 via Claude Max OAuth; repo-memory litellm channel
  intentionally left dead (accepted degradation)

## Findings

**F0 (infra, minor) — flex-start queue wait is now real.** 11 minutes PENDING
at 21:00 UTC (vs ~50s in morning runs). Not billed while pending, but it eats
wall-clock if a sweep is scheduled naively. → Suggestion S1.

**F1 (pass) — budget shaping live and correct.** Runner printed
`budget=174 min of a 3h run (guard=5, finalization reserve=15)` and
`session caps: ideation=1620s implementation=8100s` — exactly the designed
values for a 3h run.

**F2 (pass) — ideation completed inside its cap.** 579.6s (~10 min) of the
27-min cap, 38 tools, **real cost telemetry ($1.38)** because the session
reached its terminal event. Run #6's failure shape (30-min kill, research
discarded, $0 cost) did not recur; salvage stayed dormant as intended.

**F3 (pass) — PLAN.md discipline adopted immediately.** The implementation
session's second action (21:11) was `Write PLAN.md` in the workspace root.

**F4 (observation) — repo-memory channel confirmed empty.**
`sections consulted: []` — expected consequence of the accepted decision to
leave the litellm side-channels dead under OAuth-only auth. Ideation relied
on handler context + fresh reading instead; no visible struggle yet.

**F5 (agent planning, REVISED after live correction) — training config
mis-sized, then self-corrected.** At 21:15 the agent launched `python
train.py` sized at 7734 steps (~3h+ measured) against a 2h06m session cap —
with no pre-flight duration estimate. It then sat **84 minutes blocked** in
the foreground call (21:15→22:39) until Claude Code returned control with
the process still running. At 22:40 the agent read its own log, computed
"step 2172/7734, ~3 hours total — that's too long", and executed a textbook
recovery: `pkill train.py` → verified GPU cleared → **copied
checkpoint-2000 to final_model** → launched an eval on it, with ~40 min of
session left to iterate. Verdict: the *sizing* error and the *foreground
launch* are real (S2/S3 stand — a background launch would have enabled the
same correction ~30-60 min earlier, around checkpoint-2000's creation);
the recovery shows the handler's final_model/timer discipline working.

**F9 (pass, big one) — mid-run checkpoint promotion is live.** The
"maintain final_model as best-so-far at all times" instruction produced
exactly the intended behavior under time pressure: an interrupted training
run still yielded a submittable model *before* any deadline forced the
issue, and the agent measured it rather than assuming.

**F6 (pass, discipline) — durable state is excellent.** PLAN.md written as
the second action; `train_log.txt` streamed; full-model checkpoints (with
config.json + tokenizer) every 500 steps under
`artifacts/exp1-metamathqa-sft/`. A deadline kill loses the process, not the
weights: checkpoint-4500-ish will exist.

**F7 (harness bug, found by this review) — consolidation scan too shallow.**
`consolidate_final_model()` only checks `artifacts/*/` for `config.json`;
Trainer checkpoints live one level deeper (`artifacts/<exp>/checkpoint-N/`),
so the runner's fallback would MISS them in exactly this scenario. Recovery
in run #7 depends on the agent (or a short iteration 2 driven by feedback)
promoting a checkpoint to final_model. Fix queued: bounded-depth recursive
scan preferring the newest loadable dir.

**F8 (observability) — long tool calls blind the external trace.** 80+
minutes of stream silence during training; only SSH + nvidia-smi + artifact
listing distinguished "working" from "hung". The agent-side files (PLAN.md,
train_log.txt) were the reliable signal — the monitor should read them.

**F10 (harness bug, critical, found at run end) — relative RESULTS_DIR broke
the official eval in every run to date.** run_startup.sh exported
`POST_TRAIN_BENCH_RESULTS_DIR=results`; the harness eval runs with cwd
`src/eval/tasks/<task>` and passes `--model-path $EVAL_DIR/final_model`, so
the path resolved to a nonexistent location, vLLM treated it as a hub repo
id, and the server exited 1 within seconds — 9/9 attempts, `metrics.json`
absent. Latent since run #1; only observable once a real final_model
existed. Fixed: absolute `/opt/ptb/results`. The 58% model survives in GCS;
rescored via the new `40_eval_only.sh`.

**F11 (pass, selection discipline) — the agent kept the better model.** Its
training continuation scored 54% vs the checkpoint's 58%; `best_score.log`
records both and final_model retained the 58% checkpoint. Exactly the
"best-so-far, atomically" contract.

**F12 (pass, machinery) — reserve gate, lineage, honest feedback failure.**
`last_stop: finalization_reserve` (native reserve fired); iteration 2 ran
inside the reserved envelope and branched from `generic_exp_0`
(parent=node 0) — work continued in place; the final feedback call failed
tags-after-retry and was recorded as an explicit failure (main's fix)
instead of a fabricated verdict. Agent cost telemetry intact: $39.43.

**F13 (observation, kapso core) — node.code_diff is unbounded.** Node 0's
recorded diff is 6.48M chars (the agent committed eval logs/JSONs beside
code); it is persisted into every checkpoint write and flows toward
feedback/history surfaces. Candidate upstream issue: exclude obvious
non-code artifacts or window the diff for prompt use.

## Suggestions backlog

**S1.** Add a multi-zone/multi-region fallback ladder to `10_launch_run.sh`
(us-central1-a/b/c → us-east4/5, us-west1/4) for evening-hours capacity;
alternatively schedule sweeps in the UTC-morning window where the queue was
~50s.

**S2 (handler prompt).** Long-running commands must not block the session:
instruct the agent to launch training with `nohup … &`, then poll its log in
a loop that also checks `timer.sh`, so it can early-stop, adjust, or promote
a checkpoint when the clock demands it.

**S3 (handler prompt).** Require a duration pre-flight before full training:
run ~50 steps, measure s/it, set `max_steps` (or sample count) so projected
time ≤ 60% of the session's remaining runway. (Same philosophy as kapso's
new fidelity timing model, applied at the agent level.)

**S4 (harness).** Fix `consolidate_final_model` depth (F7); extend the run
monitor to report artifacts/ checkpoint count + train_log tail each beat
(F8).
