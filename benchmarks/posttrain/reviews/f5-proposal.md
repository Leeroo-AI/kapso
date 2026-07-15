# F5 fix proposal — the agent must own its clock

Status: DRAFT for review. Nothing here is applied.

## The verified causal chain (run #7 evidence)

1. **Information gap.** The agent knows the RUN deadline (`timer.sh`: "10h
   left" style) but is never told its SESSION deadline. In run #7 it planned
   training against ~3h of run budget while its implementation session had a
   135-minute cap (which kapso enforces with a process-group kill). Neither
   kapso's implementation prompt (`implementation_claude_code.md` — zero
   mentions of backgrounding/timeouts/session limits) nor our handler
   context states the cap.
2. **No sizing discipline.** `train.py` was launched with 7,734 steps at a
   measurable 1.44 s/it ≈ 3.1h — knowable after the first logging interval
   (~1 min), never computed before launch.
3. **Blocking-launch trap (our own foot-gun).** `solve.sh` exports
   `BASH_DEFAULT_TIMEOUT_MS=36000000` (10h), so a foreground training call
   may legally block the agent for hours. Trace lines 1376-1380: the call
   blocked 84.5 min until Claude Code *auto-backgrounded* it ("Command
   running in background with ID: …") — an undocumented rescue with
   uncontrollable timing. During those 84 minutes the agent could not read
   its log, check timer.sh, or early-stop; checkpoint-2000 (the eventual
   final model) had existed on disk for ~30+ of them.
4. Post-rescue friction (minor): `sleep 60 && tail` was blocked by the CLI's
   sleep-chain guard; a 342KB Read and an 82KB tail hit size caps. The agent
   adapted, losing ~1 min.

## P1 — handler context: session caps + long-process contract (ours)

Wiring: `runner.main()` already computes `session_timeouts`; pass them into
`PostTrainBenchHandler(..., session_caps=session_timeouts)` and render them.

New handler section (replaces/extends "Session discipline"):

```
## Session discipline & long-running processes
You operate in bounded SESSIONS inside the overall run. Hard caps, enforced
by a process-group kill that takes down EVERY process you started
(including training): implementation sessions ≈ {impl_min} minutes,
ideation ≈ {idea_min} minutes. Only files on disk survive a session kill.
- First action of every implementation session: run `date -u` and write the
  session start time + your session deadline into PLAN.md, alongside the
  run-level time from `bash timer.sh`.
- NEVER run a command expected to exceed 10 minutes in the foreground.
  Launch it detached and log to a file:
      nohup python train.py > {artifacts}/train_log.txt 2>&1 &
      echo $! > {artifacts}/train.pid
  then poll in BOUNDED waits (one wait ≤ 5 minutes, e.g.
  `sleep 240 && tail -5 {artifacts}/train_log.txt && bash timer.sh`).
  Do useful work between polls (prepare eval scripts, update PLAN.md).
- SIZE BEFORE YOU COMMIT: before any training run projected >15 minutes,
  measure throughput (≤50 steps or one logging interval), compute
  total_steps × seconds_per_step, and choose max_steps/dataset size so the
  projected duration fits within 60% of YOUR SESSION's remaining time.
  Write the arithmetic into PLAN.md.
- At every poll apply the rule: if projected completion exceeds your
  session's remaining time minus 20 minutes (reserve for merge+eval+
  promotion), kill training, promote the best checkpoint to final_model,
  and evaluate it.
- Checkpoint at least every ~15 minutes of training (size save_steps
  accordingly) so a kill never loses more than one interval.
```

## P2 — solve.sh: stop legalizing multi-hour blocking calls (ours)

`BASH_DEFAULT_TIMEOUT_MS`: 36000000 → **900000** (15 min).
`BASH_MAX_TIMEOUT_MS` stays 36000000 (agent may still explicitly request
long foreground waits when it has a reason).

Effect: an un-annotated blocking call now fails fast at 15 min with a
timeout error (which itself teaches the contract) instead of freezing the
session for an undocumented rescue to end. Properly detached launches
(`nohup … &`) return in <1s and are unaffected. Risk: a legitimately long
foreground call (a full evaluate.py pass can exceed 15 min) gets killed if
the agent ignores the prompt — recoverable (re-run backgrounded), and P1
explicitly teaches both patterns.

## P3 — kapso core (DEFERRED, separate approval needed)

Dynamic per-session countdown in `_render_budget_status()` ("this session
terminates in N minutes") + an adapter-injected T-minus-10 warning line.
Deferred: P1's static caps + agent-side clock cover most of the value
without touching `src/kapso`.

## Validation (run #8 trace assertions)

1. No single Bash tool call spans >15 minutes.
2. PLAN.md contains session start/deadline and the sizing arithmetic.
3. Poll cadence visible (bounded waits, timer checks between them).
4. If training is killed (by agent or session), a checkpoint promotion to
   final_model happens inside the same session.
