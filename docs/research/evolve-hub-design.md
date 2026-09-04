# Evolve hub — the proxy between evolve modules and the human

**Status:** DESIGN v3 (2026-09-04, revised the same day with the writer,
judge and reply decisions) for review. Nothing built. Written against
the shipped code at a9d127ed (0.4.2), on branch `notif-evolve`.
Companion: the user-flow page ("Waiting on You").

**Decision driver (user, 2026-09-03):** when evolve is blocked on access
or credentials an idea needs, it should ask the user and, once the user
provides it, continue from where it left off. The ask goes through a
**hub**; the user works the hub one item at a time; the hub is the proxy
between evolve modules and the human.

**v3 direction (user, 2026-09-04):** when a session needs something it
puts the need in the hub and stops. If nothing else is running the
campaign stops too, waiting on the user. When the user responds the
campaign resumes and the session is continued through the coding-agent
CLI's own resume, with the response as new input. If the user never
responds, the session is never continued. No wait ceiling, no polling,
no in-session holding. The writer is a tool, so the call itself is the
signal that stops the session. A paused node is judged only when it
completes.

This supersedes v2 (the in-session wait). Everything that existed only
to keep a session alive while a human was away — the re-check interval,
progress notifications, the idle rule, the ordering of three clocks, the
held session deadline, the paused budget clock — is gone. What replaces
it is one fact both CLIs give us: a finished non-interactive session can
be resumed later with its full context.

---

## 1. The problem

The implementation prompt ends with "Do not ask any questions. Implement
everything as specified." There is no sanctioned way for a session to say
it is blocked. Today a missing `OPENAI_API_KEY` ends as a null score or an
invalid evaluation with the real cause inside `technical_difficulties`;
ideation reads that text through the experiment-history tools but nothing
stops it re-proposing the idea, and nothing in the contract forbids the
agent from stubbing the call to get an evaluation out. The user learns
about it, if at all, by reading feedback after the run.

Preflight (`core/preflight.py`, `kapso doctor evolve`) covers what the
**config** needs in seconds. It cannot know what an **idea** will need.

## 2. The rule

A session that hits something only a person can provide records the need
and is stopped. The campaign stops as soon as nothing else is running.
The person does the fix, tells the hub, and resumes the campaign; the
session that stopped is continued with its full context and the person's
response. A need nobody responds to is never continued. There is no
attended-versus-unattended distinction, because nothing ever holds.

| # | Situation | v3 | Later |
|---|---|---|---|
| 1 | Idea needs a key mid-build | records the need, is stopped; campaign pauses; resumed with full context once the check passes | |
| 2 | Nobody at the terminal | identical — a paused campaign costs nothing; `kapso watch` shows it, `on_status` fired once | |
| 3 | Other candidates could run | pauses anyway; v3 has no parking | park the node, run the runner-up, continue it when the check passes |
| 4 | The evaluation suite itself needs the key | the first session records it; one response covers every later session | preflight fails in seconds before any session |
| 5 | The goal names a provider with no key present | recorded when a session hits it | preflight advisory; an environment inventory steers ideation |
| 6 | A human action, not a value (a gated model) | the fix is numbered steps, the check is the access call; same cycle | |
| 7 | Provided but wrong, expired, out of credits | `kapso hub resolve` shows the check failing and why; fix, resolve again; `--resume` re-runs open checks and pauses again if they still fail | |
| 8 | No GPU on this machine | not a need — nobody provides it in minutes; the session reports it | an infeasible-here outcome; the inventory keeps ideation off it |
| 9 | Rate limit on a key that exists | retried inside the session; never a need | |
| 10 | The user declines, or hands back an alternative | `kapso hub reply 3 "note"`; `--resume` continues the session with the note; a replied key never pauses the campaign again | replied keys join the environment inventory |

Principles that survive: every ask carries a fix for the human and a
check that decides whether it is satisfied; secrets never pass through
Kapso; faking is never the cheaper path; ask once.

## 3. Channels compared

Five ways a module can reach the human, kept for the record of why the
shape is what it is.

| Channel | What it is | Precedent in the code |
|---|---|---|
| **A. Session-end report** | the session ends with a structured block; the strategy files it | `<evaluation_change_request>` → maintainer routing, with a cap and a freeze |
| **B. Blocking tool** | an MCP tool that holds inside the session until the human answers | none |
| **C. Non-blocking tool** | an MCP tool that posts to the hub and returns; the session is stopped | the bank gate's pull log: a gate appending to a campaign file |
| **D. Prevention only** | preflight scans and an environment inventory | preflight `Requirement` rows; the selector's groundedness criterion |
| **E. Notify-and-resume** | status file, `on_status`, a checkpoint with `last_stop`, `--resume` | observability layer, checkpoint schema 2 |

v3 is C as the writer (user decision 2026-09-04: with a tool, the call
itself tells Kapso to stop the session — nothing to parse, nothing that
depends on the agent ending its turn well), E as the stop-and-resume
machinery, and the CLI's own session resume as the continuation. B is
retired: it existed to keep context alive, and CLI resume keeps context
alive on disk without holding anything. A is not needed once C exists.
D stays the first line of defence, later.

## 4. The design

### 4.1 The hub's structure

One append-only JSONL file per campaign, `.kapso/hub.jsonl`, beside
`run_state.json`, `status.json` and `experiment_history.json`: locked
appends, gitignored with the rest of `.kapso/`. One JSON object per
line, each an **event** about an **item**. An item's state is the fold of
its events; nothing is ever rewritten.

| Event | Written by | Carries |
|---|---|---|
| `posted` | the session, through the `hub_post` tool | the whole need (fields below), the node, the CLI session id |
| `checked` | `kapso hub resolve`, or the orchestrator at `--resume` | exit code, the check's output tail, masked, duration |
| `met` | whoever ran the check that passed | — |
| `replied` | `kapso hub reply` | the human's note |
| `continued` | the orchestrator, when the session was resumed | the session id, the follow-up text |

A need (the `posted` event) carries:

| Field | What it is |
|---|---|
| `id` | campaign-local integer, the handle in every command |
| `key` | dedupe handle and history label: `env:OPENAI_API_KEY`, `data:raw/transactions.csv`, `tool:docker`, `access:hf:meta-llama/…` |
| `node`, `session` | the experiment node and the CLI session id, so the continuation knows what to resume |
| `for` | the idea in one line, so the human knows what their minutes buy |
| `hit` | the concrete error the session ran into |
| `fix` | what the human does — free text, copy-pasteable where it can be: a line for `.env`, a licence URL plus a login command, a path to drop a file at, an install command |
| `check` | a shell snippet; exit 0 means satisfied (§4.2) |
| `next_steps` | what the session would have done next, in its own words; fed back at the continuation |

States, from the fold: **open** (posted, nothing else), **answered**
(`met` or `replied`), **continued**. `kapso hub list` shows open items
first. No item ever carries a value; check output is masked before it is
written.

The file, for the example on the page:

```jsonl
{"ts":"2026-09-04T09:12:31Z","event":"posted","id":3,"node":3,"session":"550e8400-e29b-41d4-a716-446655440000","key":"env:OPENAI_API_KEY","for":"node 3 · re-rank candidates with text-embedding-3-large","hit":"openai.AuthenticationError at the embedding step — no key in the environment","fix":"add OPENAI_API_KEY=sk-... to /home/me/churn/.env","check":"python -c \"import openai; openai.OpenAI().models.list()\"","next_steps":"embed the candidate texts with text-embedding-3-large, re-rank, run kapso_evaluation/evaluate.py"}
{"ts":"2026-09-04T11:40:02Z","event":"checked","id":3,"exit":1,"seconds":0.8,"output":"OpenAIError: The api_key client option must be set"}
{"ts":"2026-09-04T11:41:15Z","event":"checked","id":3,"exit":0,"seconds":1.3,"output":""}
{"ts":"2026-09-04T11:41:15Z","event":"met","id":3}
{"ts":"2026-09-04T11:42:00Z","event":"continued","id":3,"node":3,"session":"550e8400-e29b-41d4-a716-446655440000"}
```

A reply instead of a fix:

```jsonl
{"ts":"2026-09-04T11:40:40Z","event":"replied","id":3,"note":"use the local bge-large model instead"}
```

Who reads it: `kapso hub` (the inbox), `kapso watch` (the open items on
the paused state), the orchestrator at `--resume` (which nodes can
continue, with what), the build prompt and the experiment-history render
(an open key means the resource is absent; a replied key carries the
note and is never asked for again).

### 4.2 The check

Every need carries a shell snippet the session writes, because the
session knows what it tried. It succeeds — exit code 0 — only when the
need is satisfied. Kapso runs it in the campaign workspace, in a fresh
process with the run's environment (the `.env` the run loads at start
included), capped at `blocked.check_timeout_seconds` so a hanging
command cannot hang `kapso hub resolve` or `--resume`. Its exit code
decides `met`; its output, masked, is stored and shown so the human sees
why it still fails.

It runs at most twice: once when the human runs `kapso hub resolve
<id>`, and once more at `--resume` before anything is spawned. Never on
an interval. It is the guard against situation 7: without it, a wrong
key would cost a whole resume, a failed session, a new post and a new
pause.

| Need | Check |
|---|---|
| a key in `.env` | `python -c "import openai; openai.OpenAI().models.list()"` |
| a gated model | `python -c "from huggingface_hub import model_info; model_info('meta-llama/Llama-3.1-8B-Instruct')"` |
| a dataset at a path | `test -s kapso_datasets/raw/transactions.csv` |
| a tool on the box | `docker info` |
| a service | `curl -sf http://localhost:6333/healthz` |
| a bucket permission | `python -c "import boto3; boto3.client('s3').head_bucket(Bucket='my-bucket')"` |

When the check is wrong rather than the fix — the key is there and the
snippet is bad — the human answers with `reply` instead, which
continues the session with the note and no check.

### 4.3 The writer: the `hub_post` tool

A bundled `hub` gate in `gated_mcp/presets.py`, given to implementation
sessions (ideation later), never to the feedback judge. Its one tool,
`hub_post(key, hit, fix, check, next_steps)`, appends the `posted` event
through the env-injected hub path (`KAPSO_HUB_PATH`, the bank gate's
pull-log pattern) and returns at once: `{"id": 3, "stopping": true}`.

The call is the signal. The adapter, which already polls the session
process every half second for its deadline, tails the hub file; on a
`posted` event for its node it gives the session `blocked.stop_grace_seconds`
to end its turn (the tool result tells the agent to stop, and the
prompt says the same), then SIGTERMs it — which both CLIs treat as a
resumable state. Either way the session close runs as today: the working
tree is committed and pushed to the node's branch. Nothing depends on
the agent's cooperation: the next steps are in the tool's arguments, the
work is in the branch.

Dedupe is server-side: a `hub_post` on a key that is open joins it
(same id back); on a key that was replied, the tool returns the reply
instead of posting, and the session carries on with it — a replied key
never stops a session again. The build prompt also renders the hub's
open and replied items, so a session rarely gets that far.

Sessions without MCP — the SDK-based adapters, codex ideation members —
have no tool in v3 and end with their report as today.

### 4.4 The cycle

1. **Post.** The session hits the wall and calls `hub_post`. The gate
   writes `posted`; the tool result says the session is stopping.
2. **Stop.** The adapter ends the session (grace, then SIGTERM); the
   session close commits and pushes the working tree; the strategy marks
   the node **suspended** with its hub item ids and CLI session id, runs
   no judge, returns.
3. **Pause.** When the iteration's lanes are done (a suspended lane
   returns early; the barrier waits only for lanes still building, and
   the finished lanes are judged as usual), the orchestrator saves the
   checkpoint with `last_stop: needs_input`, prints the ask in the
   preflight row format with the two commands to run next, writes the
   status file `done` with `stopped_reason: needs_input` and the open
   items, fires `on_status` once, and returns
   `SolveResult(stopped_reason="needs_input")`. The budget clock needs no
   special handling: the process exits, and elapsed time is only
   accumulated in-process, exactly like a budget stop today.
4. **Respond.** The person does the fix. `kapso hub resolve 3` runs the
   check once and records `checked` and, on exit 0, `met`; `kapso hub
   reply 3 "note"` records `replied`. Both optional: `--resume` runs the
   open checks itself.
5. **Resume.** `kapso evolve … --resume` validates the checkpoint as
   today, runs the check of every open item once, then for every
   suspended node whose items are all answered continues its session:
   recreate the session folder at its deterministic path from the
   node's branch, then the CLI's own resume with one follow-up message —
   "hub #3 OPENAI_API_KEY: met, check passed. Your next steps were: …
   Continue." or "hub #3: reply — 'use the local bge-large model
   instead'. Continue with that." — with the same MCP config, model,
   permissions and sandbox flags as the original launch. A suspended
   node with an item still open prints the ask again and pauses again,
   spawning nothing. The continued session takes the first iteration
   slot, ahead of any new ideation.
6. **Finish.** The continued session ends with its normal tags, the
   judge runs — this is the first time the judge sees the node, and it
   sees a completed one — and the node finalizes under its original id
   as the iteration it always was. The campaign proceeds.

A need nobody responds to leaves the campaign paused with a resumable
checkpoint. Modes on `blocked.policy: continue` (the benchmark harnesses,
where nobody is at the keyboard by design) do not expose the tool;
sessions there behave as today.

### 4.5 What the two CLIs give us (checked 2026-09-03/04 against the current docs)

| | Claude Code, `claude -p` | Codex, `codex exec` |
|---|---|---|
| Pin or capture the session id | `--session-id <uuid>` at launch — kapso mints the id, nothing to parse; also on the `system/init` and `result` events of `--output-format stream-json` | `--json` at launch; `{"type":"thread.started","thread_id":"…"}` is the first event; the adapter does not pass `--json` yet |
| Resume with a follow-up | `claude -p --resume <session-id> "<follow-up>"`; restores the full history including tool calls and results; findable from any directory (v2.1.223+) | `codex exec resume <SESSION_ID> "<follow-up>"` (or `--last`); rollout files persist unless `--ephemeral` |
| What must be repeated | `--mcp-config`, `--model`, `--dangerously-skip-permissions`, `--append-system-prompt`, output format: none of it is restored | the `-c` overrides (MCP servers, effort), `--sandbox`, `-m`, `--output-last-message`: per invocation |
| Working directory | transcripts under `~/.claude/projects/<cwd-derived name>/<session-id>.jsonl`; the agent's file paths are absolute, so the session folder must exist at the same path | rollouts under `$CODEX_HOME/sessions`; same requirement |
| A stopped session | SIGTERM leaves the turn unfinished and records no result; resuming continues that turn — the follow-up ordering after an interrupted turn is a live test (§8) | not documented |
| Retention | 30 days by default (`cleanupPeriodDays`); `--no-session-persistence` must never be set | not documented |
| The environment | a resumed session is a fresh process: it sees the `.env` the run reloads at start | same |

The session folder is `<workspace>/sessions/<branch>` — deterministic —
and `ExperimentSession` already removes and re-clones it at setup, so
recreating it from the branch before a resume is the existing setup step
with the branch checked out. Installed here: Claude Code 2.1.260, Codex
0.144.1; the adapter comments pin behaviours verified on 2.1.157.

### 4.6 The human's side

| Surface | Shows or does |
|---|---|
| terminal | at the pause: each need in the preflight row format (`[NEED]` / for / hit / fix / check), then the two commands: `kapso hub <campaign> resolve <id>` and the exact `kapso evolve … --resume` line; at resume: "hub #3 met — continuing node 3's session" or the ask again |
| `kapso hub <campaign>` | `list` (open first, age, last check result), `show <id>` (the ask, the next steps, the check's last output masked), `resolve <id>` (runs the check once), `reply <id> "note"` (continue with this note, no check — a decline, an alternative, or "done, your check is wrong") |
| `kapso watch` | `PAUSED · needs input · hub #3` on the done state, readable after the process is gone; `--json` for scripts |
| `on_status` | fires once at the pause with `stopped_reason: "needs_input"` and the open items |
| Python API | `solution.metadata["stopped_reason"] == "needs_input"`, `solution.needs`; `evolve(..., resume=True)` continues |
| end summary | the open needs, each with its node, fix and the resume line |
| experiment history / ideation | a suspended node renders with its open item ("absent"); a replied key renders with its note ("do not propose") |

### 4.7 Rules for modules

- Ask only for what a person must do. Installing a package, downloading
  public data, retrying a rate limit are the session's own job.
- A need is load-bearing or it is not asked for: W&B logging without a
  key is dropped and mentioned in the report.
- Never stub, mock, fabricate the resource, or search the machine for
  credentials. Asking must be the cheaper path.
- Put the next steps in the call; do nothing after it. The session is
  being stopped and its working tree committed.
- One item per key per campaign; the prompt shows what is open and what
  was replied; a replied key is never asked for again.
- A `check` is cheap, read-only, safe to run by hand, prints no secret,
  and is the only thing that turns a need into `met`.
- Transient versus human: rate limits retry; billing and auth states
  block.

### 4.8 Secrets

The hub holds needs and, masked, the output of their checks. A value
goes wherever the fix says — usually the `.env` file the run loaded,
whose path every ask prints — and a resumed campaign is a fresh process
that reads that file at start, so the value never passes through Kapso.
`config.yaml` holds no secrets (Rule 3), and neither does `.kapso/`.

## 5. Landing on today's code (v3)

- `execution/hub.py`: the record (locked append, fold, mask, the check
  runner with a per-run cap).
- `gated_mcp/gates/hub_gate.py` + a `GateDefinition` in `presets.py`
  with `KAPSO_HUB_PATH` as injected env; `hub_post`; server-side dedupe
  and the replied-key answer; added to the shipped modes'
  `implementation_gates`.
- Prompts: the one exception to "do not ask questions"; the rules of
  §4.7; the hub's open and replied items rendered into the build prompt.
- `coding_agents/base.py` + adapters: a `resume(session_id, follow_up)`
  method beside `run`; the poll loop tails the hub file and ends the
  session after `stop_grace_seconds`; Claude passes `--session-id` at
  launch and `--resume` on continuation; Codex passes `--json` at
  launch, records `thread_id`, runs `codex exec resume`; both repeat
  their launch flags.
- `generic/strategy.py` + `implementation.py`: mark the node suspended
  on a `posted` event (no judge); on the first `run()` after a resume,
  continue suspended nodes whose items are answered before any ideation.
- `search_strategies/base.py`: `suspended`, `hub_item_ids`,
  `cli_session_id` on `SearchNode`; round-tripped through `dump_state`
  and the experiment store.
- `experiment_workspace`: recreate a session folder from a branch at the
  deterministic path without cutting a new branch.
- `orchestrator.py`: the `needs_input` stop (`VALID_LAST_STOPS`,
  `stopped_reason`), the check pass at resume, the open items in the
  status file and the `on_status` payload.
- `kapso.py`: `solution.needs`.
- `cli.py`: `kapso hub` (`list`, `show`, `resolve`, `reply`); `watch`
  rendering of the paused state; the exact resume line in the pause
  message.
- `config.yaml` `defaults.blocked`: `policy` (pause | continue),
  `check_timeout_seconds`, `stop_grace_seconds`. Benchmark modes set
  `policy: continue`.
- Docs: `docs/evolve/` gains a page; `docs/reference/cli.mdx` and
  `configuration.mdx` gain the verb and the block.

## 6. Decisions

Settled (user, 2026-09-03/04): the hub is the record and the human
surface; a session that needs something posts it through the tool and is
stopped — the call is the signal; the campaign pauses when nothing else
runs; the person responds through the hub with `resolve` or `reply`;
`--resume` continues the very session with the response; no response
means no continuation; no ceilings, no polling; the check snippet is the
contract; implementation sessions only (ideation later); a replied key
never pauses the campaign again; a paused node is judged only when it
completes, because the judge only ever receives completed nodes; a
transcript that has expired is not handled now.

Open:

1. **Defaults.** `check_timeout_seconds` 30, `stop_grace_seconds` 120.
   Benchmark modes on `continue`.

## 7. Out of scope for v3

Questions and notices that do not stop a session; park and re-queue;
ideation asks; the selector's access criterion; the preflight sources and
the environment inventory; the hosted inbox and push notifications (the
hook and the record are built for them); asking through the feedback
judge; questions from the learning crews; an expired transcript.

## 8. Open concerns (v3)

Verify live, on the installed CLIs:

1. `claude -p --resume <id>` after the adapter's SIGTERM: the docs say
   the interrupted turn is continued; confirm the follow-up lands after
   the `hub_post` tool result, not before, and that a session that ended
   its turn cleanly within the grace behaves the same. The follow-up on
   stdin (the adapter never puts prompts in argv), `--mcp-config`
   re-passed, the session folder recreated first.
2. `codex exec resume <thread_id>` with `--json` at the original launch,
   the `-c` overrides and `--sandbox` repeated, `--output-last-message`
   on the resumed run, and what an interrupted turn does on resume.
3. What each CLI does when the working directory differs from the
   original.

Implementation care:

4. The session close must still commit on the stopped path; confirm the
   SIGTERM-then-close ordering leaves a clean commit on the branch.
5. Masking is heuristic; the check runs by hand through `kapso hub
   resolve` in the user's shell — `show` prints the check first.
6. Lock the appends: the gate, the CLI and the orchestrator all write.
7. The `.env` path: record which file `find_dotenv` loaded and print it
   in the fix; when none was found, name where to create it.
8. A node with several posts continues only when all are answered.
9. The status file's done state carries the items so `watch` can render
   them after the process is gone.
