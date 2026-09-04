# Evolve inbox — how a campaign asks a person for what it needs

**Status:** DESIGN v4 (2026-09-04) for review — the concluded shape after
the 2026-09-03/04 conversation. Nothing built. Written against the
shipped code at a9d127ed (0.4.2), on branch `notif-evolve`. Companion:
the user-flow page ("Waiting on You"). Earlier revisions of this file
(a hub with a blocking wait, then a check-and-poll loop) are in the
branch history; §9 records what was dropped and why.

**Decision driver (user, 2026-09-03):** when evolve is blocked on access
or credentials an idea needs, it should ask the user and, once the user
provides it, continue from where it left off — without losing the
coding session's context.

**The concluded shape (user, 2026-09-04):** a session that needs
something posts a request through a tool and is stopped; the campaign
pauses and exits; the person reads the request in an **inbox** and
replies; the reply resumes the campaign, and the very session is
continued through the coding-agent CLI's own resume with the reply as
new input. No checks run by Kapso — the coder verifies for itself and
posts again if still blocked. No polling, no daemon, no cron: the
person's reply is the only trigger. A switch turns the whole feature
off. v4 covers one implementation lane: a campaign whose mode sets
`node_expansion_value` above 1 runs with the inbox off (user decision
2026-09-04), and parallel lanes come in a later implementation.

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

A session that hits something only a person can provide posts a request
and is stopped. The campaign stops as soon as nothing else is running,
and exits. The person does the fix, replies in the inbox, and the reply
resumes the campaign; the session that stopped is continued with its
full context and the reply. A request nobody replies to is never
continued. There is no attended-versus-unattended distinction, because
nothing ever holds.

| # | Situation | v4 | Later |
|---|---|---|---|
| 1 | Idea needs a key mid-build | posts the request, is stopped; campaign pauses; `kapso inbox reply` resumes it with full context | |
| 2 | Nobody at the terminal | identical — a paused campaign costs nothing; `kapso inbox` shows it whenever someone looks | |
| 3 | Other candidates could run | pauses anyway; v4 has no parking | park the node, run the runner-up, continue it on reply |
| 4 | The evaluation suite itself needs the key | the first session posts it; one reply covers every later session | preflight fails in seconds before any session |
| 5 | The goal names a provider with no key present | posted when a session hits it | preflight advisory; an environment inventory steers ideation |
| 6 | A human action, not a value (a gated model) | the fix is numbered steps; the reply says it is done; same cycle | |
| 7 | Provided but wrong, expired, out of credits | the continued session fails again and posts again; the new request quotes the previous reply, so the loop is visible at once | |
| 8 | No GPU on this machine | not a request — nobody provides it in minutes; the session reports it | an infeasible-here outcome; the inventory keeps ideation off it |
| 9 | Rate limit on a key that exists | retried inside the session; never a request | |
| 10 | The user declines, or hands back an alternative | the reply says so; the continued session proceeds on it | replies join the environment inventory |

Principles that survive: every request carries a fix the person can act
on; secrets never pass through Kapso; faking is never the cheaper path;
the person's action is the only trigger.

## 3. Vocabulary

| Thing | Name |
|---|---|
| the place, the command, the file | inbox — `kapso inbox` — `.kapso/inbox.jsonl` |
| one item | a request, with a simple per-campaign id (`#1`, `#2`) |
| the tool the coder calls | `request_from_user` |
| the person's answer | a reply — one kind, free text; empty means done |
| the campaign's state, everywhere it is shown | "waiting on you" — `stopped_reason: waiting_for_user` |
| the switch | `inbox.enabled` |

"Hub" was dropped because a hub is a place things pass through, not a
place a person goes to act. An inbox says what to do with it.

## 4. The design

### 4.1 The inbox on disk

One append-only JSONL file per campaign, `.kapso/inbox.jsonl`, beside
`run_state.json`, `status.json` and `experiment_history.json`: locked
appends, gitignored with the rest of `.kapso/`. One JSON object per
line, each an event about a request. A request's state is the fold of
its events; nothing is ever rewritten.

| Event | Written by | Carries |
|---|---|---|
| `requested` | the session, through the tool | `id`, `key`, `for`, `hit`, `fix`, `next_steps`, `node`, `session`; when the same key was requested before, `previous_reply` |
| `replied` | `kapso inbox reply` | `id`, `note` |
| `continued` | the orchestrator, when the session was resumed | `id`, `node`, `session` |

| Field | What it is |
|---|---|
| `id` | campaign-local integer, the only handle the person uses |
| `key` | what is needed, as the coder names it: `env:OPENAI_API_KEY`, `data/transactions-2019.csv`, `tool:docker`, `access:hf:meta-llama/…` |
| `for` | the idea in one line, so the person knows what their minutes buy |
| `hit` | the concrete error the session ran into |
| `fix` | what the person does — free text, copy-pasteable where it can be: a line for `.env`, a licence URL plus a login command, a path to drop a file at, an install command |
| `next_steps` | what the session would have done next, in its own words; fed back to it at the continuation |
| `node`, `session` | the experiment node and the CLI session id, so the continuation knows what to resume |

States, from the fold: **open** (requested, no reply), **answered**
(replied), **continued**. No request ever carries a value.

```jsonl
{"ts":"2026-09-04T09:12:31Z","event":"requested","id":1,"node":3,"session":"550e8400-e29b-41d4-a716-446655440000","key":"env:OPENAI_API_KEY","for":"node 3 · re-rank candidates with text-embedding-3-large","hit":"openai.AuthenticationError at the embedding step — no key in the environment","fix":"add OPENAI_API_KEY=sk-... to /home/me/churn/.env","next_steps":"embed the candidate texts with text-embedding-3-large, re-rank, run kapso_evaluation/evaluate.py"}
{"ts":"2026-09-04T11:41:15Z","event":"replied","id":1,"note":"added the key"}
{"ts":"2026-09-04T11:41:16Z","event":"continued","id":1,"node":3,"session":"550e8400-e29b-41d4-a716-446655440000"}
```

Who reads it: `kapso inbox`, `kapso watch` (the open requests on the
paused state), the orchestrator at resume (which nodes can continue,
with what), and the build prompt and the experiment-history render (an
open key means the resource is absent; an answered key carries the note
and is not requested again unless the coder cannot proceed on it).

### 4.2 The tool: `request_from_user`

A bundled `inbox` gate in `gated_mcp/presets.py`, given to implementation
sessions (ideation later), never to the feedback judge. Its one tool,
`request_from_user(requests=[{key, hit, fix, next_steps}, …])`, takes a
list so a session that needs two things asks for both in one call,
appends one `requested` event per entry through the env-injected inbox
path (`KAPSO_INBOX_PATH`, the bank gate's pull-log pattern), and returns
at once: `{"ids": [1, 2], "stopping": true}`.

The call is the signal. The adapter, which already polls the session
process every half second for its deadline, tails the inbox file; on a
`requested` event for its node it gives the session
`inbox.stop_grace_seconds` to end its turn (the tool result tells the
agent to stop, and the prompt says the same), then SIGTERMs it — a state
both CLIs treat as resumable. Either way the session close runs as
today: the working tree is committed and pushed to the node's branch.
Nothing depends on the agent's cooperation: the next steps are in the
call, the work is in the branch.

When a key was requested before in this campaign, the new request
carries `previous_reply`, and the pause message shows it ("again — your
reply to #1 was: …"), so a loop is visible the moment it starts.

Sessions without MCP — the SDK-based adapters, codex ideation members —
have no tool in v4 and end with their report as today. With
`inbox.enabled: false` no session has the tool and nothing pauses. The
same holds when the mode's `node_expansion_value` is above 1: v4 handles
one implementation lane, so a campaign with parallel lanes runs with the
inbox off and says so once at launch.

### 4.3 The cycle

1. **Request.** The session hits the wall and calls `request_from_user`
   with what it needs and what it would do next. The gate writes the
   requests; the tool result says the session is stopping.
2. **Stop.** The adapter ends the session (grace, then SIGTERM); the
   session close commits and pushes the working tree; the strategy marks
   the node **suspended** with its request ids and CLI session id, runs
   no judge, returns.
3. **Pause.** The orchestrator saves the checkpoint with
   `last_stop: waiting_for_user`, prints the requests
   with the reply line, writes the status file `done` with
   `stopped_reason: waiting_for_user` and the open requests, fires
   `on_status` once, and returns
   `SolveResult(stopped_reason="waiting_for_user")`. The process exits.
   The budget clock needs nothing special: elapsed time is only
   accumulated in-process, exactly like a budget stop today.
4. **Reply.** The person does the fix and runs `kapso inbox reply <id>
   "note"`. The command appends `replied`, then — if the campaign lives
   on this machine, no process holds it, and at least one suspended node
   has all its requests answered — resumes the campaign in the
   foreground through the same path `kapso evolve --resume` uses, with
   the arguments from the launch record (§4.5). Otherwise it says what
   is still open, or why it cannot resume here.
5. **Resume.** For every suspended node whose requests are all answered:
   recreate the session folder at its deterministic path from the node's
   branch, then the CLI's own resume with one follow-up message —
   "Request #1 OPENAI_API_KEY — reply: added the key. Your next steps
   were: … Continue." — with the same MCP config, model, permissions and
   sandbox flags as the original launch. A campaign never starts new
   work while a request is open: it continues what it can, and when
   that is done it pauses again showing what is still waiting.
6. **Finish.** The continued session verifies for itself and carries on.
   If it still cannot proceed, it posts again (§4.2) and the cycle
   repeats. Otherwise it ends with its normal tags, the judge runs —
   the first and only time it sees this node, and it sees a completed
   one — and the node finalizes under its original id as the iteration
   it always was.

### 4.4 What the two CLIs give us (checked 2026-09-03/04 against the current docs)

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

### 4.5 Two small records that make the reply self-sufficient

**The launch record**, `.kapso/launch.json`, written by `kapso evolve`
at the first run: the resolved arguments the checkpoint does not hold —
goal source, output path, mode, coding agent, eval dir, data dir,
iterations and budgets, KG index, config path — so `kapso inbox reply`
can resume without the person retyping anything. A campaign started
from the Python API with a callback that cannot be serialized (an
`iteration_evaluator`) is written with `resumable_from_inbox: false`,
and the reply command records the note and says to resume from the
script with `resume=True`.

**The campaign registry**, one append-only file whose path is the
config key `inbox.registry`: `kapso evolve` adds a line at launch with
the campaign path and goal, so `kapso inbox` with no campaign can list
every campaign waiting on you. A line whose directory no longer exists
is skipped (a missing file is the documented default; a corrupt line
raises). Deleting the registry loses only the cross-campaign view;
`kapso inbox <path>` works on any campaign directly.

### 4.6 The person's side

Two commands, no flags.

```
kapso inbox                        # what is waiting on you
kapso inbox reply <id> "…"         # answer it; the campaign resumes when it can
```

Run inside a campaign directory (the nearest ancestor with
`.kapso/inbox.jsonl`), both act on that campaign. Run anywhere else,
`kapso inbox` lists every campaign in the registry with open requests,
and `kapso inbox reply` takes the campaign path first. With one request
open the id may be omitted. An empty reply means done.

```
$ kapso inbox
./campaign  churn model  waiting 2h

  #1  OPENAI_API_KEY
      for   node 3 · re-rank candidates with text-embedding-3-large
      hit   openai.AuthenticationError at the embedding step — no key in the environment
      fix   add OPENAI_API_KEY=sk-... to /home/me/churn/.env
      next  embed the candidate texts, re-rank, run kapso_evaluation/evaluate.py

  reply with   kapso inbox reply 1 "…"

$ kapso inbox reply 1 "added the key"
  #1 answered. Resuming ./campaign: continuing node 3's session on generic_exp_3. Ctrl-C stops it.
```

With two requests: `#1 answered. #2 still open, so node 3 waits; nothing
else to run.` — then `kapso inbox reply 2` resumes. The reply resumes in
the foreground; anyone who wants to walk away wraps it in `nohup` as
with any long run. What the command never does: run anything the person
did not ask for, repeat, or hide. If a live process holds the campaign
(status file pid and heartbeat), it says so and stops. If the campaign
is not on this machine, it records the reply and says so. If the resume
fails, it prints the error and the manual `kapso evolve --resume` line.

Replies are stored verbatim and handed to the coder as text: a note,
never a value. The fix says where a value goes.

| Surface | Shows or does |
|---|---|
| terminal at the pause | the requests (`#id` / for / hit / fix / next), then `reply with kapso inbox reply <id> "…"` and `any time kapso inbox`; the summary block says `WAITING ON YOU` |
| `kapso inbox` | the two commands above |
| `kapso watch` | `WAITING ON YOU · 1 request` on the done state, readable after the process is gone; `--json` for scripts |
| `on_status` | fires once at the pause with `stopped_reason: "waiting_for_user"` and the open requests |
| Python API | `solution.metadata["stopped_reason"] == "waiting_for_user"`, `solution.requests`; `Kapso.inbox(campaign)` and `Kapso.reply(campaign, id, note)` as thin wrappers over the same operations |
| experiment history / ideation | a suspended node renders with its open request ("absent"); an answered key renders with its reply |

### 4.7 Rules for the coder

- Ask only for what a person must do. Installing a package, downloading
  public data, retrying a rate limit are the session's own job.
- A request is load-bearing or it is not made: W&B logging without a
  key is dropped and mentioned in the report.
- Never stub, mock, fabricate the resource, or search the machine for
  credentials. Asking must be the cheaper path.
- Ask for everything you need in one call, with the next steps; do
  nothing after it. The session is being stopped and its working tree
  committed.
- After a reply, verify for yourself. If you still cannot proceed, ask
  again and say what you tried; the person sees their previous reply
  next to the new request.
- A reply that says the resource is not available is an instruction:
  proceed on it, do not ask for that key again.
- Transient versus human: rate limits retry; billing and auth states
  request.

### 4.8 Secrets

The inbox holds requests and replies as text. A value goes wherever the
fix says — usually the `.env` file the run loaded, whose path every
request prints — and a resumed campaign is a fresh process that reads
that file at start, so the value never passes through Kapso.
`config.yaml` holds no secrets (Rule 3), and neither does `.kapso/`.

## 5. Landing on today's code (v4)

- `execution/inbox.py`: the record (locked append, fold), the launch
  record, the registry.
- `gated_mcp/gates/inbox_gate.py` + a `GateDefinition` in `presets.py`
  with `KAPSO_INBOX_PATH` as injected env; `request_from_user` with a
  list argument and the `previous_reply` lookup; added to the shipped
  modes' `implementation_gates` when `inbox.enabled`.
- Prompts: the one exception to "do not ask questions"; the rules of
  §4.7; the inbox's open and answered requests rendered into the build
  prompt.
- `coding_agents/base.py` + adapters: a `resume(session_id, follow_up)`
  method beside `run`; the poll loop tails the inbox file and ends the
  session after `stop_grace_seconds`; Claude passes `--session-id` at
  launch and `--resume` on continuation; Codex passes `--json` at
  launch, records `thread_id`, runs `codex exec resume`; both repeat
  their launch flags.
- `generic/strategy.py` + `implementation.py`: mark the node suspended
  on a `requested` event (no judge); on the first `run()` after a
  resume, continue suspended nodes whose requests are answered, before
  any ideation; never ideate while a request is open.
- `search_strategies/base.py`: `suspended`, `request_ids`,
  `cli_session_id` on `SearchNode`; round-tripped through `dump_state`
  and the experiment store.
- `experiment_workspace`: recreate a session folder from a branch at the
  deterministic path without cutting a new branch.
- `orchestrator.py`: the `waiting_for_user` stop (`VALID_LAST_STOPS`,
  `stopped_reason`), the open requests in the status file and the
  `on_status` payload.
- `kapso.py`: `solution.requests`; `Kapso.inbox`, `Kapso.reply`; the
  launch record and registry writes at the start of `evolve`.
- `cli.py`: `kapso inbox` and `kapso inbox reply`; `watch` rendering of
  the paused state; the reply line in the pause message.
- `config.yaml` `defaults.inbox`: `enabled`, `stop_grace_seconds`,
  `registry`. GENERIC and MINIMAL inherit `enabled: true`; benchmark
  modes set `enabled: false`. At the start of `evolve`, an enabled inbox
  with `node_expansion_value` above 1 is turned off for that campaign
  with one printed line.
- Docs: `docs/evolve/` gains a page; `docs/reference/cli.mdx` and
  `configuration.mdx` gain the command and the block.

## 6. Decisions

Settled (user, 2026-09-03/04): the name is inbox; a session that needs
something requests it through the tool and is stopped — the call is the
signal; the campaign pauses and exits when nothing else runs; the person
replies with free text through `kapso inbox reply <id>`; the reply
resumes the campaign in the foreground, and the very session is
continued with the reply; no checks run by Kapso — the coder verifies
and asks again if still blocked; no polling, no daemon, no cron; a
simple per-campaign id per request; implementation sessions only
(ideation later); a paused node is judged only when it completes; a
campaign never starts new work while a request is open; `inbox.enabled`
turns the feature off; one implementation lane only — a mode with
`node_expansion_value` above 1 runs with the inbox off; an expired
transcript is not handled now.

Open:

1. **Defaults.** `stop_grace_seconds` 120; `inbox.registry` under the
   user's Kapso home.

## 7. Out of scope for v4

Parallel implementation lanes (`node_expansion_value` above 1: a
suspended lane returning early, the barrier waiting only for lanes still
building, the finished lanes judged as usual); requests that do not stop
a session (questions, notices); park and re-queue; ideation asks; the
selector's access criterion; the preflight
sources and the environment inventory; a local page (`kapso inbox
serve`) and the hosted inbox — both are further clients of the same two
operations, and only the hosted one needs something local to react to a
remote reply; asking through the feedback judge; questions from the
learning crews; an expired transcript.

## 8. Open concerns (v4)

Verify live, on the installed CLIs:

1. `claude -p --resume <id>` after the adapter's SIGTERM: the docs say
   the interrupted turn is continued; confirm the follow-up lands after
   the tool result, not before, and that a session that ended its turn
   cleanly within the grace behaves the same. The follow-up on stdin
   (the adapter never puts prompts in argv), `--mcp-config` re-passed,
   the session folder recreated first.
2. `codex exec resume <thread_id>` with `--json` at the original launch,
   the `-c` overrides and `--sandbox` repeated, `--output-last-message`
   on the resumed run, and what an interrupted turn does on resume.
3. What each CLI does when the working directory differs from the
   original.

Implementation care:

4. The session close must still commit on the stopped path; confirm the
   SIGTERM-then-close ordering leaves a clean commit on the branch.
5. Lock the appends: the gate, the CLI and the orchestrator all write.
6. The `.env` path: record which file `find_dotenv` loaded and print it
   in the fix; when none was found, name where to create it.
7. The status file's done state carries the requests so `watch` and
   `kapso inbox` can render them after the process is gone.
8. The launch record must hold everything `evolve` needs; the API path
   with a callback is the one it cannot.
9. The registry with concurrent campaigns: append-only, locked, stale
   lines skipped.

## 9. What was dropped, and why

- **A blocking in-session wait** (v2): kept context by holding a tool
  call open; needed ceilings, progress notifications, idle rules and an
  ordering of three clocks. Unnecessary once the CLI's own session
  resume keeps context on disk.
- **Checks run by Kapso** (v2/v3): a snippet per request, run on an
  interval or on demand. Delegated to the coder, which verifies after
  the resume and asks again if still blocked; Kapso never runs
  agent-written snippets outside a session.
- **`done` and `decline` as separate verbs**, and reply-by-key: one
  free-text reply per request id is enough; a decline is a reply.
- **A watcher, daemon or cron** to resume campaigns: in an open-source
  tool such a thing is what users break first. The reply command is the
  trigger.
- **"Hub"**: replaced by inbox.

## Appendix A — the prompt text

Three pieces. All three are rendered only when the inbox is on for the
campaign; with it off, the prompts are byte-identical to today.

### A.1 The implementation prompt section

Rendered as `{{inbox_section}}` in `implementation_claude_code.md` and
`coding_agent_implement.md`, placed after "Session Runtime Discipline"
(it is about the session's lifecycle). The tool also gets one line under
"Available Tools" pointing here. `{{inbox_state}}` inside it lists the
campaign's requests so far (§A.4); it is empty on a fresh campaign.

```markdown
## When you are blocked on something only a person can provide

{{inbox_state}}

Some blockers no amount of engineering removes: a credential that was
never provided, a licence someone must accept, a permission on a bucket,
a dataset that exists only on someone's machine, credits on an account.
For these — and only these — use the `request_from_user` tool.

- **What qualifies.** Something a PERSON must do that you cannot: provide
  a secret, accept terms, grant access, drop a file, pay, install
  software you lack permission for. Installing a package, downloading
  public data, retrying a rate limit, working around a flaky service, or
  choosing between reasonable designs is YOUR job — never a request.
- **Load-bearing only.** Ask when the <solution> cannot be implemented
  as specified without it. A missing key for optional logging or
  telemetry is dropped and mentioned in `technical_difficulties`, not
  requested.
- **Never fake it.** Do not stub or mock the resource, fabricate outputs
  (random embeddings, canned API responses), hard-code a placeholder that
  lets the evaluation pass, or search this machine for credentials.
  Asking is always the cheaper path; a faked result is worse than none.
- **One call, everything you need.** Before calling, list every blocker
  you can already see and put them all in ONE call. Each request carries
  `key` (what is needed: `env:OPENAI_API_KEY`,
  `access:hf:meta-llama/Llama-3.1-8B-Instruct`,
  `data/transactions-2019.csv`, `tool:docker`), `hit` (the exact error
  or symptom you saw), `fix` (what the person should do, copy-pasteable:
  the line to add to `.env`, the URL to accept terms at plus the login
  command, the path to drop the file at), and `next_steps` (what you
  will do once it is met, in your own words — you will be resumed with
  this).
- **The call stops your session.** Commit any uncommitted work BEFORE
  calling. After the call returns, do nothing else: the session is being
  ended and your working tree committed; write no further code and
  return no final tags. You will be resumed later, in this same
  conversation, with the person's reply.
- **Continuing after a reply.** The reply is the first thing you read
  when resumed. Verify for yourself that the blocker is gone — try the
  call, read the file, run the command. If it still fails, call
  `request_from_user` again and say what you tried; the person sees
  their previous reply next to your new request. If the reply says the
  resource is not available, that is an instruction: proceed on it — use
  the alternative it names, or drop that part — and do not ask for that
  key again.
- **Transient is not a blocker.** Rate limits, timeouts and flaky
  networks are retried with backoff inside your session. Authentication,
  authorization and billing errors are blockers.
```

The closing line of both prompts, "Do not ask any questions. Implement
everything as specified and run the evaluation.", becomes, when the
inbox is on: "Do not ask questions in your output — text outside the
final tags is never read. The one way to ask for something is the
`request_from_user` tool, and only for what a person must do." The
Final Checklist's item about returning the XML tags gains "— unless you
called `request_from_user`, in which case return nothing".

### A.2 The tool as the agent sees it

```python
Tool(
    name="request_from_user",
    description=(
        "Ask the person running this campaign for something only a person "
        "can provide — a credential, a licence acceptance, an access grant, "
        "a file, credits — when the solution cannot be implemented without "
        "it. Calling this STOPS your session: the campaign pauses until the "
        "person replies, then this same session is resumed with their "
        "reply. Put every blocker you can see into one call. Never use it "
        "for things you can do yourself (installs, downloads, retries, "
        "design choices), and never fake the resource instead of asking."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "requests": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "What is needed, short and stable: "
                                "env:OPENAI_API_KEY, "
                                "access:hf:meta-llama/Llama-3.1-8B-Instruct, "
                                "data/transactions-2019.csv, tool:docker"
                            ),
                        },
                        "hit": {
                            "type": "string",
                            "description": "The exact error or symptom you saw",
                        },
                        "fix": {
                            "type": "string",
                            "description": (
                                "What the person should do, copy-pasteable: the "
                                "line to add to .env, the URL to accept terms at "
                                "and the login command, the path to drop a file at"
                            ),
                        },
                        "next_steps": {
                            "type": "string",
                            "description": (
                                "What you will do once this is met, in your own "
                                "words — you will be resumed with this"
                            ),
                        },
                    },
                    "required": ["key", "hit", "fix", "next_steps"],
                },
            }
        },
        "required": ["requests"],
    },
)
```

The tool result, as text: `Recorded as request #1. Your session is being
stopped now — do nothing further. You will be resumed in this
conversation with the person's reply.` When a key was requested earlier
in the campaign, one more line: `Note: #1 for env:OPENAI_API_KEY was
answered before — "added the key". The person will see that reply next
to this request.`

### A.3 The follow-up the resumed session receives

Sent as the one user message of `claude -p --resume` / `codex exec
resume`, built by the orchestrator from the inbox record:

```text
Your session was stopped while waiting on the person running this
campaign. They have replied.

Request #1 — env:OPENAI_API_KEY
  you asked them to: add OPENAI_API_KEY=sk-... to /home/me/churn/.env
  their reply: "added the key"

Your next steps, as you recorded them:
  embed the candidate texts with text-embedding-3-large, re-rank, run
  kapso_evaluation/evaluate.py

Continue from there. This is a fresh process: the current .env is
loaded, and your working tree is the branch as you committed it.
Verify the blocker is gone before relying on it; if it is not, call
request_from_user again and say what you tried. If the reply says the
resource is not available, proceed without it as instructed and do not
ask for that key again. Everything else about this session is
unchanged — same branch, same directories, same final output format:
end with the XML result tags as required.
```

With two requests, one block per request. A reply that is empty renders
as `their reply: (done)`.

### A.4 The inbox state, rendered into the prompt

```markdown
### Requests already in this campaign's inbox
- #1 env:OPENAI_API_KEY — answered (node 3): "added the key"
- #2 data/transactions-2019.csv — open, no reply yet: treat as ABSENT and
  do not request it again
```

Empty on a fresh campaign, in which case the heading is omitted.
