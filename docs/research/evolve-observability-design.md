# Kapso observability — one status layer for every long-running operation

**Status:** DESIGN v3 (user directions 2026-08-25: v2's 80/20 cut, then
"one abstract class, versioned per Kapso functionality"). The build
surface is §1-§4; §7 defers the rest; §8 keeps the v1 research +
inventory foundations.

The facade has three long-running operations — `evolve()` (hours),
`learn()` (tens of minutes to an hour of crews), `learn_knowledge()`
(an hour+ of ingestion sessions) — and during any of them the engineer
has the same four questions: **is it alive · where is it · how is it
doing · show me**. One mechanics class answers them; each operation
contributes only its phase list and its "how is it doing" payload.

---

## 1. `OperationStatus` — one abstract base, three thin profiles

`src/kapso/execution/observability.py` (~150 lines total, all three
profiles included).

```python
class OperationStatus(ABC):
    """Mechanics, shared by every operation (subclasses add NO mechanics):
    - atomic status-file write (tmp+fsync+replace, the checkpoint pattern)
    - state machine: starting -> running -> done | failed
    - phase tracking: phase, phase_started_at (reset on change)
    - heartbeat_at + an OPTIONAL tiny daemon thread for operations that
      have no natural per-minute update site (interval from
      budget.checkpoint_heartbeat_seconds — evolve reuses its existing
      checkpoint daemon instead)
    - `recent`: 10-line human ring inside the file
    - update(**fields) / note(line) / heartbeat() / done(**) / failed(err)
    """
    OPERATION: str            # subclass constant -> written into the file
    PHASES: tuple[str, ...]   # subclass constant -> legal phase values

class EvolveStatus(OperationStatus):
    OPERATION = "evolve"
    PHASES = ("lens_planning", "ideation", "implementation",
              "evaluation", "feedback")
    # payload: budget{elapsed_min,total_min}, best{score,node,branch},
    #          last{score,node}, iteration, active_stream
    # TIME ONLY, no cost fields (user decision 2026-08-25): cost
    # accounting is unreliable at the source — codex sessions report
    # $0.00, so any displayed dollar figure would be a lie for the
    # platform's default implementor. Wall-clock is the honest meter
    # every operation has; the ledger keeps its internal numbers, the
    # observability surface just never shows them.

class LessonStatus(OperationStatus):
    OPERATION = "learn"
    PHASES = ("harvest", "mine", "exam", "lesson", "push")
    # payload: trajectory_id, bank_head_before, repair_round,
    #          cards{created,updated} (filled at lesson end), run_dirs

class KnowledgeStatus(OperationStatus):
    OPERATION = "learn_knowledge"
    PHASES = ("ingest", "merge")
    # payload: sources{done,total}, current_source (name/url),
    #          pages_extracted, active_stream
```

The file schema is shared; `operation` + `phases` in the header tell any
reader what it is looking at:

```json
{"operation": "learn", "state": "running", "pid": 52104,
 "heartbeat_at": "2026-08-25T16:40:12Z",
 "phase": "exam", "phase_started_at": "2026-08-25T16:22:03Z",
 "trajectory_id": "campaign1/20260825T161500_facade",
 "bank_head_before": "3e713c93", "repair_round": 1,
 "recent": [
   "16:12 harvested into store (historical contract)",
   "16:14 mined view written (41 files)",
   "16:22 exam started at bank head 3e713c93",
   "16:38 verifier round 1: 2 block findings — repair round started"
 ]}
```

**Placement** — each operation's status file lives with its own
artifacts, and every facade result records the path:
- evolve → `W/.kapso/status.json` (unchanged from v2);
- learn / learn_knowledge → `<learning.status_dir>/<operation>-<stamp>.json`
  (ONE new config key, default `learning/status`, inside the established
  gitignored `learning/` run-dir home);
- `SolutionResult.metadata["status_path"]`, `LessonResult.metadata[…]`,
  and the pipeline result gain the pointer; each facade also prints
  `status: <path>` at start — copy-paste target for watch.

## 2. Wiring — one line beside each existing marker

- **evolve**: as v2 (~8 sites + the existing checkpoint-heartbeat daemon).
- **learn** (`kapso.py`'s chain): five phase calls at the stage
  boundaries that already print (harvest/mine/exam/lesson/push), `note()`
  for exam admission + repair rounds + the card list, base-class daemon
  ON (crew sessions run 30+ min between updates; without it every crew
  wait would read as a stall).
- **learn_knowledge**: phase `ingest` with `sources.done/total` +
  `current_source` updated per source (the pipeline already loops them),
  phase `merge` for the KG merge session; daemon ON. The per-source
  ingestion session's `stream_artifact_path` — where one exists — lands
  in `active_stream`.
- CLI `learn` subcommands get the same for free: `cmd_learn ingest`
  calls the same facade chain.

## 3. `kapso watch <path>` — one command for all three

`<path>` = a workspace (resolves `W/.kapso/status.json`), a status file,
or a directory containing them. The renderer is generic (state / heart /
phase + elapsed / recent / STALLED banner from heartbeat staleness) plus
one small block chosen by `operation`:
- evolve → budget bar, best/last, node score row (from
  `experiment_history.json`), active stream;
- learn → trajectory id, bank head, phase chain with ticks
  (`harvest ✓ mine ✓ exam … lesson · push ·`), repair round;
- learn_knowledge → `sources 3/5` progress, current source, pages so
  far, active stream.
`--follow` tails `active_stream` when the operation has one (evolve
sessions, ingestion sessions) through the adapter's existing formatter;
`--json` prints the file once. Pure reader, as v2.

## 4. The flows, assuming it is implemented

**evolve** — the lunch check-in:

```
$ kapso watch ./campaign
┌─ campaign ./campaign ────────────────────────────────────────────┐
│ RUNNING ♥ 9s ago          iteration 7      pid 41211             │
│ phase: implementation — 18m elapsed (started 14:13)              │
│ budget: ▓▓▓▓▓▓▓▓▓▓▓░░░░░░░ 142/240 min                           │
│ best: 0.89  node 5 (generic_exp_5)      last: 0.87  node 6       │
│ nodes: 1:0.61  2:0.74  3:0.74  4:0.81  5:0.89  6:0.87  7:…       │
│ recent:                                                          │
│   14:29 node 6 completed score=0.87                              │
│   14:31 iteration 7 started (budget 59%)                         │
│   14:31 implementation started on generic_exp_7 (codex)          │
└──────────────────────────────────────────────────────────────────┘
```

Plus `--follow` into the live transcript, the STALLED banner when the
heartbeat goes stale (diagnosis in one command instead of ssh+pgrep),
and the CI one-liner:
`kapso watch ./campaign --json | jq -r '[.state, .best.score] | @tsv'`.

**learn** — the engineer feeds yesterday's campaign back:

```
$ kapso learn ./campaign &          # facade prints: status: learning/status/learn-20260825T161500.json
$ kapso watch learning/status/learn-20260825T161500.json
┌─ learn campaign1/20260825T161500_facade ─────────────────────────┐
│ RUNNING ♥ 21s ago                                    pid 52104   │
│ harvest ✓  mine ✓  exam (18m) …  lesson ·  push ·                │
│ bank head 3e713c93 (pinned pre-lesson)      repair round 1       │
│ recent:                                                          │
│   16:14 mined view written (41 files)                            │
│   16:22 exam started at bank head 3e713c93                       │
│   16:38 verifier round 1: 2 block findings — repair started      │
└──────────────────────────────────────────────────────────────────┘
```

The previously-invisible questions are now one glance: the exam is in a
*repair round* (not hung), the pin is the pre-lesson head, mining
produced a real view. When the lesson lands, `recent` carries
"7 cards created; bank 3e713c93 -> 51a0be22" and `state: done` — the
same facts `LessonResult.explain()` prints, durable in the file.

**learn_knowledge** — the five-source ingestion that used to be a silent
two-hour wall:

```
$ kapso watch learning/status/learn_knowledge-20260825T130501.json
│ RUNNING ♥ 8s ago      phase: ingest — source 3/5 (14m)           │
│ current: catboost.ai/docs/en/features/categorical-features       │
│ pages extracted so far: 31                                       │
│ recent: 13:22 source 2/5 done (11 pages, all claims verified)    │
$ kapso watch … --follow        # drop into the ingestion session:
  [tool:Bash] python -c "clone(CatBoostClassifier(cat_features=…))"
  [result:error] RuntimeError: Cannot clone object …
  [thinking] Reproduced — this goes in Common Errors with the wrapper.
```

The stall rule is identical everywhere: stale heart ⇒ `STALLED?` banner
⇒ `--follow` or the run dir named in `recent` ⇒ resume/redo. One habit,
three operations.

## 5. Implementation map (one sitting, +~40 lines over v2)

1. `observability.py`: `OperationStatus` base + the three profiles +
   renderer blocks.
2. Wiring per §2 (evolve ~8 sites; learn 5; learn_knowledge 3).
3. Config: `learning.status_dir: learning/status` (Rule 1, one key).
4. `cli.py`: `watch` (path resolution + per-operation block); result
   metadata pointers in the three facades.
5. Tests (Rule 9): base mechanics once (atomic write under concurrent
   update, ring cap, heartbeat staleness, phase-timer reset, daemon
   start/stop); per-profile phase-legality; watch `--json` passthrough;
   renderer smoke per operation on fixture files.
6. Live proof: one short evolve + one `learn()` watched from a second
   terminal; kill one mid-phase for the STALLED banner.

## 6. Why one base class is the right shape here

The three operations differ in *what* progresses, not in *how progress
is observed* — liveness, phase, elapsed, recency, and doneness are the
same physics. Putting the mechanics in one abstract class means: the
stall rule is defined once (one bug fix fixes three surfaces), a new
long-running operation (codify runs, A/B drivers, the development
regime) becomes a ~10-line profile, and `kapso watch` stays a single
habit rather than three commands. The profiles stay thin by rule:
**subclasses may add fields and phases, never mechanics.**

## 7. Deferred (unchanged from v2, now per-operation)

`events.jsonl` + OTel-GenAI vocabulary as a second sink inside the same
base class (no call-site changes) → `on_event` callback → threshold
alerts → inventory gap fixes (G2 ExperimentRecord fields, G5 judge
transcript paths, G9 gitignore) → OTLP exporter → `watch --all` over
`learning/status/` + workspaces.

## 8. Foundations (v1 research + inventory, unchanged)

**Platform patterns:** OTel GenAI trace vocabulary (`invoke_agent` /
`chat` / `execute_tool`, token+latency metrics) as the emerging
standard; Langfuse's aggregated-vs-expanded two-view; W&B live
dashboards + threshold alerts; Temporal heartbeats + durable event
history as source of truth; Devin-style read-only mid-run access;
MLflow/Aim proving local-first files suffice.

**Inventory conclusions (code sweep 2026-08-25 @ e3a139ae):** evolve is
data-rich — `run_state.json` (heartbeat-refreshed checkpoint with full
node history, scores, per-phase cost/duration), per-session
`stream.jsonl` flushed per event, `experiment_history.json` per
candidate, serving pull log per tool call — but signal-poor: 13
unflushed prints, dropped INFO logging, no status surface, no watch
command, no phase-start record, no artifact index, judge transcripts
unrecoverable, codex cost reported as 0.0. The learning and
knowledge-ingestion paths have even less: stage prints only, no
heartbeat at all. The minimal layer exposes the riches; the deferred
list repairs the rest.
