# Evolve observability — a layer for engineers running multi-hour campaigns

**Status:** DESIGN for review. An engineer who calls `Kapso.evolve()` on a
long objective today gets un-flushed prints on stdout and a result object
hours later; everything in between requires knowing which of nine files to
poll. This designs the observation layer. Grounded in (a) a survey of
experimentation/agent-observability platforms and (b) a full code-level
inventory of what evolve actually produces (agent sweep, 2026-08-25 @
e3a139ae).

---

## 1. What the platforms teach (research findings)

Five patterns recur across LangSmith / Langfuse / AgentOps, the
OpenTelemetry GenAI conventions, W&B, Temporal, and the long-task agents
(Devin-class):

1. **One hierarchical trace vocabulary.** The industry has converged on
   OTel GenAI: a top-level `invoke_agent` span, `chat` spans per LLM call,
   `execute_tool` spans per tool, attributes `gen_ai.request.model`,
   `gen_ai.usage.{input,output}_tokens`, `gen_ai.response.finish_reasons`;
   metrics = latency + token histograms. Still "in development", but
   emitting *mappable* events future-proofs any backend integration.
2. **Two views of one run** (Langfuse's aggregated vs expanded graphs):
   the run's overall shape versus every call in execution order. For a
   6-hour campaign these are different questions — "where is the campaign
   in its arc" vs "what is this session doing right now".
3. **Live dashboard + threshold alerts** (W&B): metrics and system state
   updating in place; alerts on completion, error, and metric thresholds.
   For search campaigns the analogue set is: score improved / budget
   crossed / error / stall.
4. **Heartbeats and a durable event history** (Temporal): long activities
   emit liveness; a missed heartbeat means presumed-dead and triggers
   recovery. The event history is the source of truth; every UI is a
   projection of it. (We have lived the absence of this: distinguishing
   "thinking hard" from "dead" has cost an ssh + pgrep every time.)
5. **Observation without interference** (Devin): watch progress, inspect
   history, open the workspace while it runs — read-only surfaces over
   live state. And the local-first trackers (MLflow file backend, Aim)
   prove no server is needed to *produce* observability; files an
   optional UI reads are enough.

## 2. What evolve already produces (inventory conclusions)

The full artifact-by-artifact inventory (writers, cadences, formats) is in
the sweep report; the conclusions that shape the design:

**Evolve is data-rich but signal-poor.** The durable state is genuinely
good: `run_state.json` is a heartbeat-refreshed snapshot carrying every
SearchNode with scores, per-phase cost/duration, and evaluation attempts;
per-session `stream.jsonl` files are flushed per event; the serving pull
log is per-tool-call. But the *signal* layer is thirteen unstructured
`print()` markers with no timestamps and no flushing, INFO logging that is
silently dropped (no basicConfig on the evolve path), and **no structured
event log, no status file, no watch command, no callback seam** anywhere.

The gaps that matter for an observer (from the sweep's gap list):
- G1 no structured event log — dashboards must scrape stdout or poll files;
- G2 `experiment_history.json` omits cost/duration/phase telemetry (they
  exist only inside the gitignored, wholesale-rewritten checkpoint);
- G3 budget snapshots and spend time-series never persist;
- G4 **no phase-start record** — a live reader cannot tell how long the
  current implementation phase has been running;
- G5 feedback-judge and single-session-ideation transcripts are
  unrecoverable (streamed to stdout with no artifact path);
- G6 no artifact index (nothing maps iteration → branch → stream file);
- G7 no framework-written campaign log — a bare `evolve()` leaves no
  textual trace unless the caller captured stdout;
- G8 phase prints are unflushed (arrive in 4-8 KB lumps under redirect);
- G9 several `.kapso/` state files are untracked *and* un-ignored;
- G10 codex sessions report cost 0.0 (ledger undercounts — known,
  documented honestly rather than fixed here).

## 3. The design: event log → status projection → watch → hooks

One producing seam, three consuming surfaces. Everything is files +
in-process callbacks; no server, no database, no SaaS (Rule 10; the
optional exporter is §6).

### 3.1 `CampaignObserver` — the single emitting seam

`src/kapso/execution/observability.py`. Constructed once per campaign by
the orchestrator, threaded to the strategy (the same pattern
`bank_serving` uses). Two methods:

```python
observer.emit(event_type, **fields)   # append one event, update status
observer.heartbeat()                  # refresh liveness only
```

`emit` does three things atomically-enough: appends one JSON line to
`W/.kapso/events.jsonl` (append + flush, the pull-log pattern), updates
`W/.kapso/status.json` (atomic tmp+replace, the checkpoint pattern), and
invokes the registered `on_event` callbacks. Every event carries
`{ts, seq, campaign_id, iteration, node_id, event, ...fields}`.

### 3.2 The event vocabulary (OTel-mappable by construction)

| event | key fields | OTel mapping |
|---|---|---|
| `campaign_started` / `campaign_resumed` | goal hash, budgets, workspace, build sha, bank_head_served, kg_index | `invoke_agent` span start |
| `iteration_started` | iteration, budget_progress, parent_branch | child span start |
| `phase_started` | phase = lens_planning \| ideation \| implementation \| evaluation \| feedback, node_id, branch, **stream_path** | span start (fixes G4, G6) |
| `phase_completed` | phase, duration_s, cost_usd, outcome digest (solution chars / files changed / attempts) | span end + `gen_ai.usage.*` |
| `session_started` / `session_completed` | cli, model, effort, stream_path, tokens in/out, cost | `chat` span |
| `evaluation_completed` | node_id, score, metrics, fidelity, valid, provenance | `execute_tool` span |
| `node_completed` | node_id, branch, score, should_stop, cost, duration | span end |
| `budget_snapshot` | elapsed, remaining, cost so far, reserve, cost_by_component | metric points (fixes G3) |
| `governance` | evaluator transition / change request / integrity refusal, detail | span event |
| `serving_staged` | bank_head, gaps | span event |
| `alert` | kind, detail (see §3.5) | span event |
| `campaign_completed` | stopped_reason, stop_detail, best score/branch, totals | `invoke_agent` span end |

Vocabulary is structural (enum in code), not config. Emission sites are
exactly the places that print today — the thirteen markers become
`observer.emit(...)` calls whose *console rendering* is derived from the
event (one formatting function), so stdout keeps working and gains
timestamps + flush (fixes G1, G7, G8: `events.jsonl` IS the campaign log).

### 3.3 `status.json` — the "now" projection

Small, overwritten atomically on every emit and every heartbeat:

```json
{
  "state": "running",              // starting|running|finalizing|done|failed
  "campaign_id": "…", "workspace": "…", "pid": 12345,
  "last_event_at": "…", "heartbeat_at": "…",       // liveness (Temporal)
  "iteration": 7, "phase": "implementation",
  "phase_started_at": "…", "phase_elapsed_s": 1130,  // fixes G4
  "budget": {"elapsed_min": 142.5, "total_min": 240, "cost_usd": 18.40},
  "best": {"score": 0.89, "node_id": 5, "branch": "generic_exp_5"},
  "last": {"score": 0.87, "node_id": 6},
  "active_sessions": [
    {"phase": "implementation", "branch": "generic_exp_7",
     "cli": "codex", "model": "gpt-5.6-sol",
     "stream": ".kapso/sessions/generic_exp_7/stream.jsonl"}
  ],
  "counts": {"nodes": 7, "invalid_evaluations": 0, "alerts": 1}
}
```

A watcher — human, script, or shepherd — answers "alive? where? how far?
what's it doing? where do I look deeper?" from one `cat`. `heartbeat_at`
is refreshed by the existing checkpoint-heartbeat daemon thread (it
already wakes on `budget.checkpoint_heartbeat_seconds`; one extra call).
Stall detection becomes `now - heartbeat_at` — no pgrep, no ssh guesswork.

### 3.4 `kapso watch` — the read-only surface

```
kapso watch <workspace> [--follow] [--events] [--json]
```

- Default: render `status.json` as a live single-screen view (in-place
  refresh: header = state/iteration/phase + phase elapsed; budget bar;
  best/last scores; experiment table from `experiment_history.json`;
  active sessions with their stream paths; recent events tail).
- `--follow`: additionally tail the active session's `stream.jsonl`
  through the adapter's existing event formatter — the "expanded view".
- `--events`: raw `events.jsonl` tail (greppable; the aggregated view).
- `--json`: one status snapshot to stdout and exit (for scripts/CI).

Pure reader over the files — it works over ssh, attaches/detaches freely
mid-run, and never touches campaign state (Devin's affordance). Plain
ANSI in-place rendering, same idiom as the adapter's tee; no new UI
dependency unless review wants one.

### 3.5 Hooks and alerts — programmatic consumers

```python
kapso.evolve(goal=…, on_event=callable)        # every event dict, sync, fail-safe-loud:
                                               # callback errors are reported and re-raised
                                               # at campaign end, never swallowed mid-run
```

Alert events are emitted by the observer itself from config thresholds
(the W&B set, translated):

```yaml
observability:
  heartbeat_seconds: 300          # sourced from budget.checkpoint_heartbeat_seconds
  alerts:
    on_error: true                # any phase failure / integrity refusal
    on_new_best: true             # score improved
    budget_thresholds: [0.5, 0.9] # fraction of time budget consumed
    stall_seconds: 900            # no event AND no session stream write
```

Alerts are just events (`event: alert`) — they land in the log, the
status counts, stdout, and the callback like everything else. Delivery
beyond that (Slack, push) is the caller's `on_event`; the platform ships
the signal, not the transport.

### 3.6 Gap fixes folded in (small, same change-set)

- G2: `ExperimentRecord` gains `cost_usd`, `duration_seconds`,
  `phase_telemetry`, `started_at` — the runs-table fields every tracker
  shows, already computed, just never persisted outside the checkpoint.
- G5: `stream_artifact_path` for the feedback judge and single-session
  ideation (`.kapso/sessions/<branch>/feedback.jsonl`,
  `.kapso/ideation/iter<N>/…`) — transcripts stop being unrecoverable.
- G8: the event-derived console renderer prints with `flush=True`.
- G9: extend the workspace gitignore block to cover
  `.kapso/{events.jsonl,status.json,evaluation_registry.json,`
  `lens_plan*.json*,ideation/,sessions/,serving/}`.
- G10 (codex cost=0): NOT fixed here — but `session_completed` carries
  `cost_known: false` for codex so dashboards render "n/a", never $0.00.

## 4. What an engineer's hour looks like after this

```
$ kapso evolve … &                # or any existing entry point
$ kapso watch ./campaign
 campaign 3f9a  RUNNING  iter 7  implementation (18m elapsed)  ♥ 12s ago
 budget ▓▓▓▓▓▓▓▓░░░░ 142/240 min   cost $18.40
 best 0.89 @ node 5 (generic_exp_5)     last 0.87 @ node 6
 nodes: 1:0.61  2:0.74  3:0.74  4:0.81  5:0.89  6:0.87  7:…
 session: codex gpt-5.6-sol → .kapso/sessions/generic_exp_7/stream.jsonl
 recent: 14:32 evaluation_completed node 6 score=0.87
         14:33 alert new_best=false budget=0.59 crossed
$ kapso watch ./campaign --follow   # drop into the live transcript
$ cat ./campaign/.kapso/status.json | jq .best   # scriptable, zero deps
```

And `learn()` benefits for free: the harvested trajectory now carries
`events.jsonl` — mining gets timestamps, phase durations, and governance
events it currently reconstructs from prints.

## 5. Deliberately out (Rule 10)

- **No web dashboard, no server, no DB.** The files are the API; anyone
  can put Grafana/Streamlit over them later without our involvement.
- **No OTLP exporter yet** — the vocabulary is mappable by construction
  (§3.2 table); an `events.jsonl → OTel` exporter is a follow-up the
  moment a real backend need appears.
- **No metrics beyond what campaigns already compute** — no GPU/system
  telemetry collection; the benchmark boxes have their own monitors.
- **No retention/rotation machinery** — events.jsonl for a 6-hour
  campaign is a few thousand lines; revisit only if measured otherwise.

## 6. Implementation map

1. `execution/observability.py`: `CampaignObserver` (emit/heartbeat,
   event enum, status projection, alert thresholds, callback registry) +
   the console renderer.
2. Orchestrator: construct observer (config + `on_event` from evolve),
   thread to strategy via params (the `bank_serving` pattern); convert
   the 13 print sites; heartbeat call in the checkpoint daemon;
   `budget_snapshot` emission where `BudgetSnapshot` is computed.
3. Strategy/modules: `phase_started`/`phase_completed` around lens
   planning, ideation, implementation, evaluation, feedback;
   `session_*` from the adapter boundary (config already flows there).
4. `kapso.py`: `on_event` kwarg; `SolutionResult.metadata["events_path"]`.
5. `cli.py`: `kapso watch` subcommand (reader only).
6. §3.6 gap fixes.
7. Tests (Rule 9): event-log append + status atomicity under concurrent
   emit; phase pairing (every started has completed/failed); alert
   threshold firing incl. stall; watch `--json` output shape; judge
   transcript file exists after a stubbed campaign; ExperimentRecord new
   fields round-trip; console renderer flushes.
8. Live proof: one short evolve, then `kapso watch` from a second
   terminal + `--json` in a script; verify the harvested bundle carries
   events.jsonl.

## 7. Open questions for review

1. `kapso watch` rendering: plain ANSI in-place (no new dependency, my
   recommendation) or adopt `rich` for the table/bar polish?
2. Should `on_event` failures abort the campaign immediately (strict
   Rule 2) instead of report-and-reraise-at-end? Observability killing a
   6-hour run over a broken Slack hook argues for the latter; strictness
   argues for the former.
3. Multi-campaign view (`kapso watch --all` over a workspace root):
   worth including now, or first ship single-campaign?
