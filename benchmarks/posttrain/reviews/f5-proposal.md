# F5 fix proposal — REFINED (v2)

Status: awaiting approval. Nothing applied.

Review provenance: Codex review was attempted twice (first attempt could not
read the un-pushed file; the resumed attempt showed zero tool activity for
46 minutes and was cancelled). The adversarial pass below was performed
in-session against the same five questions, grounded in run #7 trace
forensics that arrived after v1 (finding F14).

## Revised causal model (what actually cost what)

| Cost | Cause | Fixable by |
|---|---|---|
| ~2 min | blocking `python train.py` until Claude Code's ~2-min auto-backgrounding kicked in (verified: `system:task_started` fires ~2 min after every long eval call in the trace) | P2′ (teach the behavior), P1 (explicit background launch) |
| **~80 min** | **Claude Max rate-limit stall of the model's next API turn** (zero model turns 21:15→22:39; the trace's only mid-session `rate_limit_event` at the wake-up; event density 128/5min → 39/85min) | P4 |
| the whole hazard | training sized at 7,734 steps ≈ 3.1h against a 135-min session cap the agent was never told about, with no pre-flight arithmetic | P1 |

Key reframe: with correct sizing (P1), the stall would have been *harmless*
— training would have completed within the cap and the wake-up would have
found it done. Sizing is the fix; the stall is a separate channel.

## P1 — handler contract: session caps, sizing, detached launches (APPLY)

Wiring: `runner.main()` already computes `session_timeouts`; pass them into
`PostTrainBenchHandler(..., session_caps=...)` and render real numbers.

Replace/extend the handler's "Session discipline" section with:

1. **State the caps**: "implementation sessions are hard-killed at
   ~{impl_min} min, ideation at ~{idea_min} min; the kill takes down every
   process you started, including training. Only files survive."
2. **Clock anchoring**: first action of every implementation session — run
   `date -u`, write session start + session deadline into PLAN.md next to
   the run-level `timer.sh` reading.
3. **Size before you commit**: before any run projected >15 min, measure
   throughput (≤50 steps or one logging interval), compute
   `total_steps × s_per_step`, and choose max_steps/dataset so projected
   duration ≤ 60% of the session's remaining time (the other 40% covers
   merge + eval + promotion + one corrective action). Write the arithmetic
   into PLAN.md.
4. **Detached launches**: never block on a command expected to exceed
   10 min. `nohup python train.py > {artifacts}/train_log.txt 2>&1 &
   echo $! > {artifacts}/train.pid` (plain nohup — no setsid, so the
   session kill still reaps it), then poll in bounded waits (≤5 min each:
   tail the log, `bash timer.sh`, compare against PLAN.md's deadline) and
   do useful work between polls.
5. **Kill-and-promote rule**: at any poll, if projected completion exceeds
   session-remaining minus 20 min, kill training, promote the best
   checkpoint to final_model, evaluate it.
6. **Checkpoint cadence**: ≤15 min of work between checkpoints.

## P2 — WITHDRAWN (was: lower BASH_DEFAULT_TIMEOUT_MS to 15 min)

Auto-backgrounding already unblocks foreground calls at ~2 min — far
tighter than any safe default timeout — and a 15-min default would
timeout-kill legitimate 10–20-min foreground evals. Worst of both worlds;
dropped.

**P2′ (APPLY, prompt-only)**: one paragraph teaching the observed CLI
behavior so the agent isn't surprised by it: "a long foreground command
returns after ~2 min as a background task with an output-file path — poll
that file; prefer explicit `nohup … &` so you control the pattern instead
of relying on the automatic conversion."

## P4 — the rate-limit stall channel (NEW)

- **(a) Operational, apply now**: do not use the Max account interactively
  while a scored run is live (this dev session shares the same plan);
  start official runs on a fresh rolling window.
- **(b) Monitoring, apply now**: the run monitor treats >10 min of trace
  silence *combined with* an idle GPU as a stall alarm; silence with a busy
  GPU is a working state (long tool call), silence with idle GPU is a
  stalled agent.
- **(c) Decision for official runs (user call)**: API-key billing makes
  model-turn capacity deterministic. Run #7's notional usage was $36.77;
  at that scale the 28-combo sweep costs ~$1.0–1.5k in API spend versus
  free-but-throttlable Max. Recommendation: Max for dev runs, API key for
  the official 10h sweep.
- **(d) Deferred kapso-core candidates (separate written proposals)**:
  stall-aware deadline crediting (pause the session-deadline clock while
  the CLI streams zero events — run #7's session paid its full 135-min cap
  but received ~50 min of usable agent time), and a per-session countdown
  line in `_render_budget_status()`.

## Process-group safety check (Codex question 3, self-verified)

kapso starts the Claude CLI with `start_new_session=True` (CLI = process
group leader) and enforces the session deadline with `os.killpg`
(claude_code_agent.py:515-614). A `nohup … &` child stays in that group
(nohup does not setsid), so detached training dies at session kill — by
design, since the next session must own the GPU — and checkpoints on disk
carry the work forward. The harness's outer apptainer `--pid` namespace
guarantees nothing outlives the run. The launching Bash call itself exits
in <1s, so tool timeouts never touch the training process. Safe in both
directions; the prompt deliberately never mentions setsid.

## Validation (run #8 trace assertions)

1. PLAN.md contains session start/deadline + sizing arithmetic.
2. Every training launch matches `nohup … &`; no foreground call blocks
   >3 min except explicitly-chosen evals.
3. Bounded-poll cadence visible between training start and end.
4. If training is killed by anything, checkpoint promotion to final_model
   occurs in the same session.
5. Monitor emits no stall alarms — or, if one fires, the post-run report
   quantifies the stall window against the Max plan's state.
