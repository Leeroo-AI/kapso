# Evolve hub — the proxy between evolve modules and the human

**Status:** DESIGN v3 (2026-09-04) for review. Nothing built. Written
against the shipped code at a9d127ed (0.4.2), on branch `notif-evolve`.
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
no in-session holding.

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
and ends. The campaign stops as soon as nothing else is running. The
person does the fix, tells the hub, and resumes the campaign; the session
that stopped is continued with its full context and the person's
response. A need nobody responds to is never continued. That is the
whole rule; there is no attended-versus-unattended distinction any more,
because nothing ever holds.

| # | Situation | v3 | Later |
|---|---|---|---|
| 1 | Idea needs a key mid-build | records the need, ends; campaign pauses; resumed with full context once the check passes | |
| 2 | Nobody at the terminal | identical — a paused campaign costs nothing; `kapso watch` shows it, `on_status` fired once | |
| 3 | Other candidates could run | pauses anyway; v3 has no parking | park the node, run the runner-up, continue it when the check passes |
| 4 | The evaluation suite itself needs the key | the first session records it; one response covers every later session | preflight fails in seconds before any session |
| 5 | The goal names a provider with no key present | recorded when a session hits it | preflight advisory; an environment inventory steers ideation |
| 6 | A human action, not a value (a gated model) | the fix is numbered steps, the check is the access call; same cycle | |
| 7 | Provided but wrong, expired, out of credits | `kapso hub resolve` shows the check failing and why; fix, resolve again; `--resume` re-runs open checks and pauses again if they still fail | |
| 8 | No GPU on this machine | not a need — nobody provides it in minutes; the session reports it | an infeasible-here outcome; the inventory keeps ideation off it |
| 9 | Rate limit on a key that exists | retried inside the session; never a need | |
| 10 | The user declines, or hands back an alternative | `kapso hub decline 3 "note"`; `--resume` continues the session with the decline; a declined key never pauses the campaign again | declined keys join the environment inventory |

Principles that survive: every ask carries a fix for the human and a
check that decides whether it is satisfied; secrets never pass through
Kapso; faking is never the cheaper path; ask once.

## 3. Channels compared

Five ways a module can reach the human, kept for the record of why the
shape is what it is.

| Channel | What it is | Precedent in the code |
|---|---|---|
| **A. Session-end report** | the session ends with a structured `<blocked>` block; the strategy files it | `<evaluation_change_request>` → maintainer routing, with a cap and a freeze |
| **B. Blocking tool** | an MCP tool that holds inside the session until the human answers | none |
| **C. Non-blocking tool** | an MCP tool that posts to the hub and returns; the session ends | the bank gate's pull log: a gate appending to a campaign file |
| **D. Prevention only** | preflight scans and an environment inventory | preflight `Requirement` rows; the selector's groundedness criterion |
| **E. Notify-and-resume** | status file, `on_status`, a checkpoint with `last_stop`, `--resume` | observability layer, checkpoint schema 2 |

v3 is A (or C — §6 decision 1) as the writer, E as the stop-and-resume
machinery, and the CLI's own session resume as the continuation. B is
retired: it existed to keep context alive, and CLI resume keeps context
alive on disk without holding anything. D stays the first line of
defence, later.

## 4. The design

### 4.1 The record

One append-only JSONL file per campaign, `.kapso/hub.jsonl`, in the same
family as the checkpoint and the serving pull log: locked appends,
gitignored with the rest of `.kapso/`, read by `kapso hub`, `kapso
watch`, the orchestrator at resume, and the experiment-history render.
Events, not mutable rows: `posted`, `checked`, `met`, `declined`. An
item's state is the fold of its events.

A need carries:

| Field | What it is |
|---|---|
| `key` | dedupe handle and history label: `env:OPENAI_API_KEY`, `data:raw/transactions.csv`, `tool:docker`, `access:hf:meta-llama/…` |
| `for` | the node and the idea, so the human knows what their minutes buy |
| `hit` | the concrete error the session ran into |
| `fix` | what the human does — free text, copy-pasteable where it can be: a line for `.env`, a licence URL plus a login command, a path to drop a file at, an install command |
| `check` | a shell snippet; exit 0 means satisfied; cheap, read-only, prints no secret. Run once on `kapso hub resolve` and once more at `--resume`; never on an interval |

Every item names its node and the CLI session it came from. No item ever
carries a value. Check output is stored masked. `worth`, `alternatives`
and `critical` are not in v3: the campaign pauses on every need, so
there is nothing to rank against.

### 4.2 The writer

**The `<blocked>` block.** The implementation contract gains one
sanctioned exception to "do not ask questions": when a session hits
something only a person can provide, it commits its partial work, writes
its next steps to `PLAN.md` and `changes.log`, and ends with the usual
tags plus one `<blocked>` block per need — `key`, `hit`, `fix`, `check`.
The strategy files each block into the hub at extraction time, dedupes
by key against the campaign's open and declined items, and records the
node as **suspended** with its hub item ids and its CLI session id. No
judge runs on a suspended node: it has not finished.

Why a block and not a tool in v3: with no waiting there is nothing for a
tool to do that the session's final message cannot do. The block works
on every adapter (Claude, Codex, the SDK-based ones, codex ideation
members later); it needs no MCP gate, no per-CLI timeout configuration,
no new process. A non-blocking `hub_post` tool would give the session an
earlier, crash-safe record and an immediate "already open as #3" answer;
it is the natural addition once mid-session questions arrive (§7), and
the record is identical either way — §6 decision 1.

**Ask once.** The build prompt renders the hub's open and declined items
with their notes, so a session does not re-raise what the campaign
already knows. If a session ends `<blocked>` on a key that was declined
anyway, the strategy does not pause the campaign: it continues the
session at once (§4.3 step 4) with the decline as the response. A key
that is open is joined, never duplicated.

**The dead-session backstop** (later): a session killed at its deadline
gets its difficulties reconstructed from the stream; the reconstruction
can also recognise an authentication or permission signature and file
the need. Both CLIs can resume a SIGTERM-killed session, so the
continuation works for it too.

### 4.3 The cycle

1. **Block.** The session hits the wall, commits, writes its next steps,
   ends with `<blocked>`. The strategy files the item(s), marks the node
   suspended with its session id, runs no judge, returns.
2. **Pause.** When the iteration's lanes are done (a suspended lane
   returns early; the barrier waits only for lanes still building, and
   the finished lanes are judged as usual), the orchestrator saves the
   checkpoint with `last_stop: needs_input`, prints the ask in the
   preflight row format with the two commands to run next, writes the
   status file `done` with `stopped_reason: needs_input`, fires
   `on_status` once, and returns `SolveResult(stopped_reason="needs_input")`.
   The budget clock needs no special handling: the process exits, and
   elapsed time is only accumulated in-process, exactly like a budget
   stop today.
3. **Respond.** The person does the fix. `kapso hub resolve 3` runs the
   check once and records `met` or the masked failure; `kapso hub
   decline 3 [note]` records a decline. Both optional: `--resume` runs
   the open checks itself.
4. **Resume.** `kapso evolve … --resume` validates the checkpoint as
   today, runs the check of every open item once, then for every
   suspended node whose items all have a response (`met` or `declined`)
   continues its session: recreate the session folder at its
   deterministic path from the node's branch, then the CLI's own resume
   with a follow-up message — "hub #3 OPENAI_API_KEY: met, check passed
   (output …). Continue from your next steps." or "hub #3: declined —
   note: … Continue without it." — with the same MCP config, model,
   permissions and sandbox flags as the original launch. A suspended
   node whose items are still open prints the ask again and pauses
   again, without spawning anything. The continued session runs inside
   the resumed campaign's first iteration slot, ahead of any new
   ideation.
5. **Finish.** The continued session ends with its normal tags, the
   judge runs, the node finalizes under its original id and counts as
   the iteration it always was. The campaign proceeds.

A need nobody responds to leaves the campaign paused with a resumable
checkpoint, for as long as the CLI keeps the transcript (§4.4). Modes on
`blocked.policy: continue` (the benchmark harnesses) file the item, mark
the node failed, and keep going; nothing pauses and nothing is resumed.

### 4.4 What the two CLIs give us (checked 2026-09-03/04 against the current docs)

| | Claude Code, `claude -p` | Codex, `codex exec` |
|---|---|---|
| Pin or capture the session id | `--session-id <uuid>` at launch — kapso mints the id, nothing to parse; also present on the `system/init` and `result` events of `--output-format stream-json` | `--json` at launch; `{"type":"thread.started","thread_id":"…"}` is the first event; the adapter does not pass `--json` yet |
| Resume with a follow-up | `claude -p --resume <session-id> "<follow-up>"`; restores the full history including tool calls and results; findable from any directory (v2.1.223+) | `codex exec resume <SESSION_ID> "<follow-up>"` (or `--last`); rollout files persist unless `--ephemeral` |
| What must be repeated | `--mcp-config`, `--model`, `--dangerously-skip-permissions`, `--append-system-prompt`, output format: none of it is restored | the `-c` overrides (MCP servers, effort), `--sandbox`, `-m`, `--output-last-message`: per invocation |
| Working directory | transcripts live under `~/.claude/projects/<cwd-derived name>/<session-id>.jsonl`; the agent's file paths are absolute, so the session folder must exist at the same path | rollouts under `$CODEX_HOME/sessions`; same requirement on the working directory |
| Retention | 30 days by default (`cleanupPeriodDays`); `--no-session-persistence` must never be set | not documented; assume rollouts persist until removed |
| A killed session | SIGTERM leaves the turn unfinished; resuming continues that turn | not documented |
| The environment | a resumed session is a fresh process: it sees the `.env` the run reloads at start, so no reload line is needed | same |

The session folder is `<workspace>/sessions/<branch>` — deterministic —
and `ExperimentSession` already removes and re-clones it at setup, so
recreating it from the branch before a resume is the existing setup step
with the branch checked out. Anything the agent did not commit is lost,
which is why the block contract commits first (and the session close
commits anyway).

Installed here: Claude Code 2.1.260, Codex 0.144.1. The adapter comments
pin behaviours verified on 2.1.157; every resume claim above is a live
test on whatever the deployment pins (§8).

### 4.5 The human's side

| Surface | Shows or does |
|---|---|
| terminal | at the pause: each need in the preflight row format (`[NEED]` / for / hit / fix / check), then the two commands: `kapso hub <campaign> resolve <id>` and the exact `kapso evolve … --resume` line; at resume: "hub #3 met — continuing node 3's session" or the ask again |
| `kapso hub <campaign>` | `list` (open first, age, last check result), `show <id>` (the ask plus the check's last output, masked), `resolve <id>` (runs the check once), `decline <id> [note]`; later: `answer`, `note`, `seen` |
| `kapso watch` | `PAUSED · needs input · hub #3` on the done state; `--json` for scripts |
| `on_status` | fires at the pause with `stopped_reason: "needs_input"` and the open items; a Slack post is a few lines of caller code |
| Python API | `solution.metadata["stopped_reason"] == "needs_input"`, `solution.needs`; `evolve(..., resume=True)` continues |
| end summary | the open needs, each with its node, fix and the resume line |
| experiment history / ideation | a suspended node renders with its open item ("absent"); a declined key renders with its note ("do not propose") |

### 4.6 Rules for modules

- Ask only for what a person must do. Installing a package, downloading
  public data, retrying a rate limit are the session's own job.
- A need is load-bearing or it is not asked for: W&B logging without a
  key is dropped and mentioned in the report.
- Never stub, mock, fabricate the resource, or search the machine for
  credentials. Asking must be the cheaper path.
- Commit before ending blocked; write the next steps down. The
  continuation starts from them.
- One item per key per campaign; the prompt shows what is open and what
  was declined, and a declined key is never asked for again.
- A `check` is cheap, read-only, safe to run by hand, prints no secret,
  and is the only thing that turns a need into `met`.
- Transient versus human: rate limits retry; billing and auth states
  block.

### 4.7 Secrets

The hub holds needs and, masked, the output of their checks. A value
goes wherever the fix says — usually the `.env` file the run loaded,
whose path every ask prints — and a resumed campaign is a fresh process
that reads that file at start, so the value never passes through Kapso.
`config.yaml` holds no secrets (Rule 3), and neither does `.kapso/`.

## 5. Landing on today's code (v3)

- `execution/hub.py`: the record (locked append, fold, mask, the check
  runner with a per-run cap).
- Prompts: the one exception to "do not ask questions"; the `<blocked>`
  block in the session-end contract beside the five existing tags; the
  rules of §4.6; the hub's open and declined items rendered into the
  build prompt.
- `generic/strategy.py` + `implementation.py`: extract `<blocked>`, file
  it, mark the node suspended (no judge); on the first `run()` after a
  resume, continue suspended nodes whose items have a response before
  any ideation; the "declined key" immediate continuation.
- `search_strategies/base.py`: `suspended`, `hub_item_ids`,
  `cli_session_id` on `SearchNode`; round-tripped through `dump_state`
  and the experiment store.
- `coding_agents/base.py` + adapters: a `resume(session_id, follow_up)`
  method beside `run`; Claude passes `--session-id` at launch and
  `--resume` on continuation; Codex passes `--json` at launch, records
  `thread_id`, and runs `codex exec resume` on continuation; both repeat
  their launch flags.
- `experiment_workspace`: recreate a session folder from a branch at the
  deterministic path without creating a new branch.
- `orchestrator.py`: the `needs_input` stop (`VALID_LAST_STOPS`,
  `stopped_reason`), the check pass at resume, `on_status` payload with
  the open items.
- `kapso.py`: `solution.needs`.
- `cli.py`: `kapso hub` (`list`, `show`, `resolve`, `decline`); `watch`
  rendering of the paused state; the exact resume line in the pause
  message.
- `config.yaml` `defaults.blocked`: `policy` (pause | continue),
  `check_timeout_seconds`. Benchmark modes set `policy: continue`.
- Docs: `docs/evolve/` gains a page; `docs/reference/cli.mdx` and
  `configuration.mdx` gain the verb and the block.

## 6. Decisions

Settled (user, 2026-09-03/04): the hub is the record and the human
surface; a session that needs something records it and stops; the
campaign pauses when nothing else runs; the person responds through the
hub; `--resume` continues the very session with the response; no
response means no continuation; no ceilings, no polling; the check
snippet is the contract; implementation sessions only (ideation later);
a declined key never pauses the campaign again; a wait that ends with a
response is the same iteration it always was.

Open:

1. **The writer.** The `<blocked>` block at session end (recommended:
   every adapter, no gate, no per-CLI configuration, identical record)
   versus a non-blocking `hub_post` tool (earlier record, immediate
   dedupe answer, the natural home for later questions). Either can be
   swapped for the other without touching the record.
2. **The judge on a suspended node.** v2 ran it on a node whose wait had
   run out because that node was finished. A suspended node is not
   finished; recommendation: no judge until the continued session ends.
3. **When the transcript is gone.** Claude keeps transcripts 30 days.
   Recommendation: if the CLI reports no such session, continue with a
   fresh session on the same branch fed the item and the next steps the
   agent wrote — a documented default for a missing file, not a
   fallback around an error.
4. **Defaults.** `check_timeout_seconds` 30. Benchmark modes on
   `continue`.

## 7. Out of scope for v3

`hub_post` for questions and notices that do not end a session; park and
re-queue; ideation asks; the selector's access criterion; the preflight
sources and the environment inventory; the hosted inbox and push
notifications (the hook and the record are built for them); asking
through the feedback judge; questions from the learning crews.

## 8. Open concerns (v3)

Verify live, on the installed CLIs:

1. `claude -p --resume <id>` with `--session-id` minted at launch, the
   follow-up on stdin (the adapter never puts prompts in argv), the
   `--mcp-config` re-passed, and the session folder recreated first.
2. `codex exec resume <thread_id>` with `--json` at the original launch,
   the `-c` overrides and `--sandbox` repeated, `--output-last-message`
   on the resumed run.
3. What each CLI does when the working directory differs from the
   original, and when the transcript has been cleaned up.

Implementation care:

4. Uncommitted work at the block is lost; the contract commits, and the
   session close commits anyway — confirm the close still runs on the
   `<blocked>` path.
5. Masking is heuristic; the check runs by hand through `kapso hub
   resolve` in the user's shell — `show` prints the check first.
6. Fewer writers on the hub file now (the strategy and the CLI), still
   lock the appends.
7. The `.env` path: record which file `find_dotenv` loaded and print it
   in the fix; when none was found, name where to create it.
8. A node with several needs continues only when all have a response.
9. `on_status` fires once at the pause; the status file's done state
   must carry the items so `watch` can render them after the process is
   gone.
