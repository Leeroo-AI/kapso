# Evolve observability — the 80/20 layer

**Status:** DESIGN v2 (minimal cut, user direction 2026-08-25: "minimal
features that give us maximum gain"). v1's full survey + inventory
conclusions are kept in §5-§6; the build surface is §1-§3 only.

The 80% of engineer pain during a multi-hour `evolve()` is three
questions: **is it alive · where is it · how is it doing** — plus one
follow-up: **let me see it work**. The 20% of machinery that answers all
four is ONE new file and ONE read-only command. Everything else from v1
(event log, OTel mapping, callbacks, alert config, gap fixes) is
explicitly deferred until this core proves insufficient.

---

## 1. `status.json` — the one new artifact

`W/.kapso/status.json`, written by a ~100-line `CampaignStatus` helper
(atomic tmp+replace, the checkpoint's own pattern). Updated at the ~8
places the orchestrator/strategy already print phase markers, plus a
`heartbeat()` call in the existing checkpoint-heartbeat daemon thread —
no new thread, no new loop, no other file.

```json
{
  "state": "running",                     // starting|running|done|failed
  "pid": 41211, "heartbeat_at": "2026-08-25T14:31:02Z",
  "iteration": 7,
  "phase": "implementation",
  "phase_started_at": "2026-08-25T14:13:11Z",
  "budget": {"elapsed_min": 142.5, "total_min": 240.0, "cost_usd": 18.40},
  "best":  {"score": 0.89, "node_id": 5, "branch": "generic_exp_5"},
  "last":  {"score": 0.87, "node_id": 6},
  "active_stream": ".kapso/sessions/generic_exp_7/stream.jsonl",
  "recent": [
    "14:05 node 5 completed score=0.89 (new best)",
    "14:06 iteration 6 started (budget 55%)",
    "14:29 node 6 completed score=0.87",
    "14:31 iteration 7 started (budget 59%)",
    "14:31 implementation started on generic_exp_7 (codex gpt-5.6-sol)"
  ]
}
```

Design points, each earning its place:
- **`heartbeat_at`** kills the alive-or-dead ssh/pgrep ritual: fresher
  than `budget.checkpoint_heartbeat_seconds` ⇒ alive; staler ⇒ presumed
  dead (Temporal's rule). The daemon that refreshes it already exists.
- **`phase` + `phase_started_at`** answers "what is it doing and for how
  long" — today unanswerable anywhere (inventory gap G4).
- **`active_stream`** is the artifact index in one field: the pointer
  from "now" to the live transcript (which is already flushed per event).
- **`recent`** is a 10-entry ring INSIDE the file — the human-readable
  tail without a second file or a log format.
- Scripts get everything with `jq` — the file is the API.

## 2. `kapso watch <workspace>` — the one read-only command

```
kapso watch W            # live single-screen view, refresh in place
kapso watch W --follow   # + tail the active session transcript
kapso watch W --json     # print status.json once and exit (scripts/CI)
```

~120 lines in `cli.py` + a renderer: reads `status.json` (and, for the
experiment table, `experiment_history.json` — already on disk), redraws
every 2 s with plain ANSI (no new dependency). `--follow` tails the
`active_stream` file through the claude adapter's existing
`_display_stream_event` formatter — reuse, not new code. Stale heartbeat
renders a red `STALLED?` banner instead of a green heart. Pure reader:
attach/detach freely, works over ssh, cannot touch campaign state.

## 3. Implementation map (one sitting)

1. `execution/observability.py`: `CampaignStatus` — `update(**fields)`,
   `note(line)` (ring append), `heartbeat()`; atomic write.
2. Wire: orchestrator constructs it beside the checkpoint store; strategy
   receives it like `bank_serving`; calls at campaign start/end,
   iteration start, phase transitions (lens/ideation/implementation/
   feedback), node completion (score + best tracking), and inside the
   existing heartbeat daemon. Each call site is one line next to an
   existing print.
3. `cli.py`: the `watch` subcommand (renderer + `--follow` + `--json`).
   Note: the installed console-script needs a reinstall to expose it
   (the on-PATH `kapso` is an older build).
4. Tests (Rule 9): atomic write under concurrent update; ring caps at 10;
   heartbeat staleness math; `--json` passthrough shape; renderer smoke
   on a fixture status; `phase_started_at` resets on phase change.
5. Live proof: run one short evolve, `kapso watch` from a second
   terminal, kill the driver mid-phase and confirm the STALLED banner.

## 4. The flow, assuming it is implemented

An engineer kicks off a 4-hour objective and goes to lunch:

```
$ kapso evolve --goal-file objective.md --time-budget-minutes 240 \
    --output ./campaign &
```

**Check-in from a second terminal (or over ssh):**

```
$ kapso watch ./campaign
┌─ campaign ./campaign ────────────────────────────────────────────┐
│ RUNNING ♥ 9s ago          iteration 7      pid 41211             │
│ phase: implementation — 18m elapsed (started 14:13)              │
│ budget: ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ 142/240 min      cost $18.40          │
│ best: 0.89  node 5 (generic_exp_5)      last: 0.87  node 6       │
│ nodes: 1:0.61  2:0.74  3:0.74  4:0.81  5:0.89  6:0.87  7:…       │
│ recent:                                                          │
│   14:29 node 6 completed score=0.87                              │
│   14:31 iteration 7 started (budget 59%)                         │
│   14:31 implementation started on generic_exp_7 (codex)          │
└──────────────────────────────────────────────────────────────────┘
```

Ten seconds of reading answers all four questions: alive (♥ 9s), where
(iter 7, implementing, 18 minutes into the phase — normal for this
task), how it's doing (best 0.89 and climbing, budget 59% spent), and
where to look deeper (node 7's stream).

**Curiosity — watch it work:**

```
$ kapso watch ./campaign --follow
  … status header …
  [tool:Bash] python evaluate.py --data-dir ./data --seed 0
  [result:ok] accuracy=0.883 fold_std=0.011
  [thinking] The interaction features beat the baseline on every fold;
             now testing whether calibration closes the last gap…
```

**Something looks off — the stall case.** Two hours later the phase
elapsed reads 71 minutes and the heart is stale:

```
│ STALLED?  last heartbeat 14m ago       iteration 9               │
│ phase: implementation — 71m elapsed                              │
```

The engineer tails the active stream, sees the session died with the
provider, and restarts with `--resume` — the checkpoint has everything.
Total diagnosis time: one command, no ssh archaeology, no pgrep.

**Scripted check from CI / a cron:**

```
$ kapso watch ./campaign --json | jq -r '[.state, .best.score] | @tsv'
running	0.89
```

**After completion** nothing changes about today's flow —
`SolutionResult.explain()` is still the summary; `status.json` simply
ends at `{"state": "done", "best": …}` as a durable last snapshot.

## 5. Deferred (build only when the core proves insufficient)

From v1, in priority order if/when needed: `events.jsonl` durable event
history with the OTel-GenAI-mappable vocabulary → `on_event` callback →
config-threshold alerts (error/new-best/budget/stall) → gap fixes G2
(ExperimentRecord cost/duration), G5 (judge transcript artifact paths),
G9 (gitignore additions) → OTLP exporter → multi-campaign `watch --all`.
The status writer is designed so events.jsonl can be added later as a
second sink inside the same helper without touching any call site.

## 6. Foundations (v1 research + inventory, unchanged)

**Platform patterns:** OTel GenAI trace vocabulary (`invoke_agent` /
`chat` / `execute_tool`, token+latency metrics) as the emerging standard;
Langfuse's aggregated-vs-expanded two-view; W&B live dashboards +
threshold alerts; Temporal heartbeats + durable event history as source
of truth; Devin-style read-only mid-run access; MLflow/Aim proving
local-first files suffice.

**Inventory conclusions (code sweep 2026-08-25):** evolve is data-rich —
`run_state.json` (heartbeat-refreshed checkpoint with full node history,
scores, per-phase cost/duration), per-session `stream.jsonl` flushed per
event, `experiment_history.json` per candidate, serving pull log per tool
call — but signal-poor: 13 unflushed prints, dropped INFO logging, no
status surface, no watch command, no phase-start record, no artifact
index, judge transcripts unrecoverable, codex cost reported as 0.0. The
minimal layer exposes the riches; the deferred list repairs the rest.
