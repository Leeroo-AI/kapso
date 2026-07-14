# Run #7 review log — gsm8k × Qwen3-1.7B-Base, 3h, flex-start

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

**F5 (major, agent planning) — blocking 3h training in a 135-min session.**
At 21:18 the implementation agent launched `python train.py` as a blocking
foreground call: 7734 steps at 1.44s/it ≈ 3h06m of training against a session
deadline 2h06m away (23:24) and a run budget ending 23:37. Two compounding
errors: (a) no pre-flight duration estimate (steps × s/it was knowable after
one logging interval), (b) a blocking call surrenders all agency — the agent
cannot read its own train_log.txt, check timer.sh, or early-stop until the
deadline kills the whole session at ~step 4600/7734 (~60% of the cosine
schedule, LR not fully annealed). The budget clamp will contain the damage;
the plan quality caused it.

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
