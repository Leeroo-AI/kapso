# Evolve hub — the proxy between evolve modules and the human

**Status:** DESIGN v2 (2026-09-03; same-day revision narrowing v1 to
the in-session wait) for review. Nothing built. Written against the
shipped code at a9d127ed (0.4.2), on branch `notif-evolve`.
Companion: the user-flow simulation ("Waiting on You", ten situations
with terminal mock-ups) — §2 carries its condensed form.

**Decision driver (user, 2026-09-03):** when evolve is blocked on access
or credentials an idea needs, critical cases wait for the user and then
continue from where they left off. The ask should go through a tool given
to the ideation and implementation modules, posting to a **hub**; the
user works the hub one item at a time; the hub is the proxy between
evolve modules and the human. This document compares that with the other
channels we have and pins the hub's shape.

**Second direction (user, same day):** to lose no context in any module,
the session stays idle until it gets the result — a tool that checks the
hub until a maximum. v1 has only this wait option; a wait that runs out
ends in the report the session already gives today. §4.4 records what
the two coding-agent CLIs allow, checked against their current docs.

---

## 1. The problem

The implementation prompt ends with "Do not ask any questions. Implement
everything as specified." There is no sanctioned way for a session to say
it is blocked. Today a missing `OPENAI_API_KEY` ends as a null score or an
invalid evaluation with the real cause inside `technical_difficulties`;
ideation reads that text through the experiment-history tools but nothing
stops it re-proposing the idea, and nothing in the contract forbids the
agent from stubbing the call to get an evaluation out. Ideation has no
channel at all: a question such as "may I use a hosted LLM on this data?"
cannot be asked, so it is either assumed or avoided.

Preflight (`core/preflight.py`, `kapso doctor evolve`) covers what the
**config** needs — CLIs, the embedding key, gate credentials, backends —
in seconds. It cannot know what an **idea** will need.

## 2. The situations and the one rule

| # | Situation | Found by | Evolve does |
|---|---|---|---|
| 1 | Idea needs a key mid-build, attended, no equal substitute | agent, in session | waits in place, budget clock paused |
| 2 | Same, nobody at the terminal (nohup, VM, CI) | agent, in session | waits a bounded time, then pauses with a resumable checkpoint |
| 3 | Idea needs Kaggle credentials; other candidates runnable | agent, in session | parks the node, continues, re-queues when the need is met, lists it at the end |
| 4 | The evaluation suite itself needs the key | preflight | fails in seconds, standard report |
| 5 | The goal names a provider with no key present | preflight | advisory row; ideation told to avoid it |
| 6 | Human action, not a value (gated HF model) | agent, in session | numbered steps plus a probe; same rule |
| 7 | Provided but wrong, expired, out of credits | probe | masked probe error, keeps waiting; no session spent |
| 8 | No GPU on this machine | agent, in session | closed as infeasible here; ideation told; no waiting |
| 9 | Rate limit on a key that exists | agent, in session | retried inside the session; never an ask |
| 10 | User declines or hands back an alternative | user | node closed as declined; remembered; note passed to ideation |

The rule: **criticality comes from the campaign, never from the kind of
thing missing.** Critical means nothing else can run, or the goal or the
evaluation mandates it, or the idea was judged best with no substitute.
Critical waits; everything else parks and the campaign keeps scoring.
While waiting the time budget is paused, no session runs, nothing is
spent, the heartbeat keeps writing. Attended only adds keys to press.

Principles that survive every channel choice: ask at the cheapest moment
(preflight, then the selector, then mid-session); every ask carries a
fix for the human and a check that decides when to continue; secrets
never pass through Kapso (names of needs only — a value goes wherever
the fix says, usually the `.env` the run loaded, and the check verifies
it); works with nobody watching; faking is never the cheaper path; ask
once.

## 3. Channels compared

Five ways a module can reach the human. Three of them exist in some form.

| Channel | What it is | Precedent in the code |
|---|---|---|
| **A. Session-end report** | the session ends with a structured `<blocked>` block; the orchestrator parses it at the boundary | `<evaluation_change_request>` → maintainer routing, with a per-campaign cap and a late-budget freeze (`orchestrator._route_change_requests`) |
| **B. Blocking tool** | an MCP tool that waits inside the session until the human answers or a timeout expires | none — gates are read/append only today |
| **C. Hub** (the proposal) | a non-blocking MCP tool posts an item to a durable per-campaign inbox; the module states what it assumes and keeps working or ends; the orchestrator reads the hub at boundaries; the human works the inbox at their own pace, including replies back into the campaign | the bank gate's pull log: a gate appending JSONL to a campaign file through an env-injected path, read later by the runner |
| **D. Prevention only** | preflight scans plus an environment inventory handed to ideation; no runtime channel | preflight `Requirement` rows; the selector's groundedness and rule-safety criteria |
| **E. Notify-and-resume** | status file + `on_status` + pause with `last_stop` + `--resume`; no structured record of the need | observability layer, checkpoint schema 2 |

What matters, and how each channel does:

| Criterion | A report | B blocking tool | C hub | D prevention | E notify |
|---|---|---|---|---|---|
| Reaches ideation (a question *before* an idea is proposed) | awkward — ideation output is a solution text, no boundary waits on it | yes | **yes** — post, assume a default, the answer lands next round | partly (avoids what is absent; cannot ask permission) | no |
| Cheapest ask point (need declared at selection, before a build session) | no | no | **yes** — the selector can pick an idea conditionally; the boundary applies the rule before implementation | n/a | no |
| Works unattended | yes | no (times out, degrades to A) | **yes, by construction** | yes | yes |
| Session keeps working after raising the ask (does the parts that need nothing) | no — raising is ending | no — raising is waiting | **yes** | n/a | no |
| Keeps the session's context while waiting | no (branch + notes carry it) | **yes** | **yes** — v1's `hub_ask` blocks in-session; a wait that ran out keeps it through CLI session resume (§4.4) | n/a | no |
| Human sees everything in one place, at their own pace, in batch | no — asks are moments in a log | no — modal | **yes, the point of it** | end summary only | no |
| Two-way: hints, decisions, "never do X" into the campaign | no | one answer to one question | **yes** — answers and free-standing notes read at the next round | no | no |
| Coverage across agents | every session, any CLI | Claude + Codex builds; not codex ideation members (no MCP by design) | same as B — needs A as a second writer | all | all |
| Implementation risk | low (regex on output, like change requests) | high (MCP tool timeouts in both CLIs, session deadline and budget clamp while idle) | medium (a gate, a record, a CLI verb; no IPC — gates already write campaign files) | low | none |
| Over-asking risk | low — ending a session costs the agent | medium | **highest — posting is free**; needs rules, caps, dedupe | none | none |
| Minimalism | smallest | adds machinery, no record | one component that *replaces* the parked list, the declined memory and the needs field in the checkpoint | smallest | smallest |
| Platform fit (hosted inbox, push notifications) | no | no | **direct** — the inbox is the surface; the CLI is its first client | no | partial |

Verdict. The hub is the right **record and human surface**. It subsumes
A (the session-end report becomes a second writer into the same record,
which is needed anyway for codex ideation members and for sessions that
die without posting) and takes B as its v1 writer: the blocking wait is
one tool on the same record (§4.2), within the limits the two CLIs set
(§4.4). D stays the first line of defence — the cheapest ask is the one
never raised — and E is what the hub's notification side plugs into.

Two things the framing must not become. First, the hub is not modal: a
module that posts a `question` declares what it will assume and keeps
working; only a `need` with no alternative can hold the campaign, and only
under the wait rule of §2. Otherwise an unattended run turns into a stall
machine. Second, the hub is never where a value goes: it holds needs,
`.env` holds values, the probe verifies. A hosted inbox will tempt a
"paste your key here" box; if that ever exists, the value goes straight
into the run's process environment or a secrets manager, never into the
record.

## 4. The hub

### 4.1 The record

One append-only JSONL file per campaign, `.kapso/hub.jsonl`, in the same
family as the checkpoint and the serving pull log: atomic appends,
gitignored with the rest of `.kapso/`, read by `kapso hub`, `kapso watch`
and the orchestrator's boundary pass. Events, not mutable rows: `posted`,
`answered`, `resolved`, `declined`, `expired`; the current state of an
item is the fold of its events.

Three item types, a closed set:

| Type | Raised when | Carries | Resolved by |
|---|---|---|---|
| `need` | anything only a person can provide: a credential, a login, a licence click, a permission, credits, a dataset dropped at a path, a service started, a tool installed on this box, a bigger disk mounted | `key` (dedupe, e.g. `env:OPENAI_API_KEY`, `data:raw/transactions.csv`, `tool:docker`, `access:hf:meta-llama/…`), `for` (node or round + idea), `hit` (the concrete error), `worth` (why it is worth the human's time), `fix` (what the human does, free text, copy-pasteable where possible), `check` (a shell snippet; exit 0 means satisfied; cheap, read-only, safe to re-run). `alternatives` and `critical` arrive with the park policy later | the check passing at any re-check, or the human declining with an optional note |
| `question` | a decision the human should make: permission, scope, preference | the question, `options`, `assume` (what the module proceeds with if unanswered), `for` | the human answering; the answer becomes campaign context |
| `notice` | something left for the human, no action required by evolve: an orphaned better result on disk, a feature dropped for lack of a key (W&B logging), an infeasible-here idea | text, `for` | reading it (`seen`) |

Every item names its `module` (`ideation`, `implementation`, or
`orchestrator` for backstop-raised items) and its `node` or `round`.
No item ever carries a value. Probe output is stored with anything
secret-shaped masked.

### 4.2 Writers — v1 is the in-session wait

**`hub_ask`, the one v1 tool.** A bundled `hub` gate in
`gated_mcp/presets.py`, given to ideation and implementation sessions,
never to the feedback judge. The gate appends to `.kapso/hub.jsonl`
through an env-injected path (`KAPSO_HUB_PATH`), exactly the bank gate's
pull-log pattern. One call posts a `need` and blocks until one of three
results: `met` (the check passed), `declined` (the human declined through
`kapso hub`, with their note), or `timeout` (`blocked.wait_minutes`
elapsed). While blocked the gate runs the need's `check` every
`blocked.recheck_seconds` — in the workspace, under a per-run cap
(`blocked.check_timeout_seconds`), in an environment rebuilt for each
run from the process environment plus the `.env` the run loaded, so a
value added since launch is visible — and folds any hub events written
from outside. The check is the whole contract: exit 0 is the only thing
the gate reads, so one loop covers a key in `.env`, a licence accepted
upstream, a bucket permission, a file dropped at a path, a service
answering on a port, or `docker info` succeeding. The check's last
output is attached to the item, masked, so `kapso hub show` tells the
human why it still fails. On every re-check the gate emits an MCP
progress notification ("waiting on OPENAI_API_KEY · 12 min · next check
in 15 s") — this is what keeps the call alive on Claude Code (§4.4) and
gives the transcript a heartbeat. Nothing is spent while blocked: the
model is waiting on a tool result, not generating. A second `hub_ask` on
a key that already has an open item joins that item's wait (server-side
dedupe); `blocked.max_asks_per_session` caps the calls.

What the session does with the result. `met` → continue in place. The
result carries the check's passing output and, because the session's
process environment predates the wait, the one line that reloads the
`.env` the run loaded — Claude Code's Bash tool keeps a persistent
shell, so `set -a; . <path>/.env; set +a` once is enough; Codex runs each
command fresh, so it loads per command or through python-dotenv in the
code it writes. That line matters when the need was an environment
variable and is harmless otherwise. `declined` → the note is the
instruction; proceed on it or end with the report. `timeout` → commit,
write the next steps, end with the ordinary session-end report,
`technical_difficulties` naming the need.

**The session-end report.** The implementation contract gains one
sanctioned exception to "do not ask questions": a blocked session may
call `hub_ask`; when the wait runs out it commits the partial work,
writes the next steps, and ends with its report. Sessions without the
gate (codex ideation members run without MCP by design) end with the
report directly. The strategy files the need into the hub at extraction
time when no `hub_ask` was made, so the record is complete either way.

**The backstop.** A session that dies without either — killed at its
deadline, crashed — already gets its difficulties reconstructed from the
stream (`difficulties_fallback.md`). The reconstruction also classifies
an authentication or permission signature and files the `need` on the
session's behalf, module `orchestrator`.

### 4.3 The cycle in v1

1. The session hits the wall and calls `hub_ask(need)`. The gate appends
   `posted` and `wait_started` to `.kapso/hub.jsonl` and blocks.
2. While the call blocks, the adapter's poll loop — it already polls the
   process every half second for the deadline — tails the hub file. On
   `wait_started` it holds the session deadline and reports to the
   orchestrator through a callback, which pauses the ledger's clock and
   writes `waiting` to the status file: `kapso watch` shows the ask,
   `on_status` fires, the attended terminal prints the ask in the
   preflight row format above the session's own transcript. From another
   terminal, `kapso hub` can `resolve` (runs the check on demand) or
   `decline`. The heartbeat daemon keeps writing, so `elapsed_seconds`
   stays flat and `watch` never reports a stall.
3. `wait_ended` with the result → deadline and clock resume. `met`: the
   session continues; the orchestrator is not involved. `declined` or
   `timeout`: the session ends with its report; the node records as today
   (score null, difficulties) plus the hub item id; the orchestrator
   moves to the next iteration; the item stays open in the hub with its
   node.
4. Not in v1 — the continuation. When the check passes later, the
   orchestrator resumes the same CLI session with a follow-up
   ("OPENAI_API_KEY is now verified; continue"), full context restored by
   the CLI's own transcript (§4.4). Until then a need met after a
   timeout only means the next ideation round sees the resource as
   available and may propose the idea again.

With node expansion (K>1) a waiting lane holds only itself; the barrier
waits for it at most `wait_minutes`. Waiting minutes count against
neither the time budget nor the session deadline; a wait that ran out is
an ordinary failed iteration in v1 and is charged as one. Modes on
`blocked.policy: park` never expose the tool.

Questions and notices (`hub_post` without waiting), the park-and-re-queue
policy, and the runner-up at selection are the natural next steps on the
same record and are out of v1 (§7).

### 4.4 What the two CLIs allow (checked 2026-09-03 against the current docs)

| | Claude Code, `claude -p` | Codex, `codex exec` |
|---|---|---|
| Blocking MCP tool call, wall clock | `MCP_TOOL_TIMEOUT` (ms), default about 28 h; a per-server `timeout` (ms) in the `--mcp-config` entry overrides it → set to `wait_minutes` plus a margin | `mcp_servers.<id>.tool_timeout_sec`, **default 60 s** → the hub server's entry must override it to `wait_minutes` plus a margin, through the `-c mcp_servers.<id>.…` overrides the adapter already emits |
| Idle abort | `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` (ms): a call with no response and no progress notification for the window aborts; default 30 min for stdio servers (v2.1.203+) → the gate's progress notification on every re-check keeps the call alive; set the variable to the wait ceiling as well | no separate idle rule documented; `tool_timeout_sec` is the cap |
| Backgrounding of long calls | main-conversation calls over 2 min move to a background task (v2.1.212+) **except in non-interactive mode** unless `CLAUDE_AUTO_BACKGROUND_TASKS=1` → in kapso's `-p` sessions the call blocks the turn, as required | not applicable |
| Session persistence and resume | transcripts at `~/.claude/projects/<project>/<session-id>.jsonl`, 30-day `cleanupPeriodDays`; `claude -p --resume <session-id> "<follow-up>"` restores the full history including tool results; `--mcp-config`, `--model`, `--dangerously-skip-permissions` are not restored and must be passed again; a SIGTERM-killed run resumes "the turn that SIGTERM left unfinished"; `--no-session-persistence` must never be set | rollout files persist by default (`--ephemeral` disables); `codex exec resume <SESSION_ID> "<follow-up>"` or `--last`; the `-c` overrides are per invocation and must be repeated |
| Session id capture | `session_id` on the `system/init` and `result` events of `--output-format stream-json`; the adapter already streams but does not record it | `{"type":"thread.started","thread_id":"…"}` on `--json`; the adapter does not pass `--json` yet |
| kapso's own deadline | the adapter SIGTERMs at its deadline → hold the deadline while a `wait_started` is open | same |

The SDK-based adapters (gemini, openhands, aider) carry no MCP: no wait
there; they end with the report.

**Three clocks, one ordering.** The wait ceiling is bounded by whatever
ends the tool call first, so kapso orders the clocks so that the gate
is always the one that ends the wait:

1. `blocked.wait_minutes` — the gate's own ceiling; it returns a clean
   `timeout` result the session can act on;
2. the CLI's per-call wall clock — Claude's per-server `timeout`,
   Codex's `tool_timeout_sec` — set by kapso to the ceiling plus a
   margin, never below it;
3. kapso's session deadline — held for the whole wait, so the ceiling
   is never eaten by the session's remaining minutes; the campaign's
   budget clock pauses with it.

Claude's idle rule is reset by every progress notification, so it does
not bound the ceiling. Nothing model-side is open while waiting: the
assistant turn that issued the tool call has ended, and the next model
request goes out only when the tool result returns — no API timeout, no
tokens, no usage-window consumption. If a CLI cap fired first the
session would see a tool error instead of a `timeout` result and might
retry or improvise; the ordering above is what prevents that.

Why this settles the "which wait" question. Both mechanisms keep the
session's context: the in-session wait keeps it in memory for minutes;
CLI resume keeps it on disk for days and survives a kapso restart. v1
takes the in-session wait — the user's direction, and the simplest
orchestration (no continuation prompt, no re-queue). Recording the
session ids in v1 costs nothing and is exactly what v1.1's continuation
needs.

### 4.5 Readers

| Surface | Shows or does |
|---|---|
| terminal (attended) | each ask as it arrives, in the preflight row format (`[NEED]` / for / hit / worth it / fix / check), above the session's own transcript; the countdown; keys `Enter` (run the check now), `s` (decline, optional note), `q` (stop waiting now) |
| `kapso hub <campaign>` | v1: `list` (open first, waiting or not, age), `show <id>` (the ask plus the check's last output), `resolve <id>` (runs the check now), `decline <id> [note]`; later: `answer <id> <text>`, `note <text>` (a free-standing hint for the next round), `seen <id>` |
| `kapso watch` | `hub: 2 open (1 critical)` plus the newest item's fix line; `WAITING` and `PAUSED` states |
| `on_status` | the status dict gains `hub: {open, critical, newest}`; a Slack post is a few lines of caller code; the platform's push notifications hang off the same hook |
| end summary | needs still open at the end, each with its node and fix; later: parked items with their resume command, answered questions, notices |
| experiment history / ideation | a node whose wait ran out renders with its need through the experiment-history tools; later: declined keys join the environment inventory as *declined by user* so no candidate proposes them again |
| preflight / `doctor` (later) | the evaluation suite as a required source, the goal as an advisory source, and the environment inventory (names of credentials and logins present, hardware found) handed to ideation and the selector |

### 4.6 Rules for modules

Written into the ideation and implementation prompts, and enforced where
the code can:

- Post only what a person must do. Installing a package, downloading
  public data, retrying a rate limit are the session's own job.
- A `question` (later) always states its `assume`; the session proceeds
  on it.
- A `need` is load-bearing or it is a `notice`: W&B logging without a
  key is dropped and noticed, not asked for.
- Never stub, mock, fabricate the resource, or search the machine for
  credentials. Reporting the block must be the cheaper path.
- One item per key per campaign: a second `hub_ask` on an open key joins
  its wait (server-side dedupe).
- Per-session cap on asks (`blocked.max_asks_per_session`),
  config-sourced like the change-request cap.
- Transient versus human: rate limits retry; billing and auth states ask.
- A `check` is cheap (well under `blocked.check_timeout_seconds`),
  read-only, safe to run every few seconds, and prints no secret. It is
  the only thing that ends a wait with `met`.

### 4.7 Secrets

The hub holds needs and, masked, the output of their checks. A value
goes wherever the fix says — usually the `.env` file the run loaded,
whose path every ask prints — and the check runs with that file re-read,
so the value never has to pass through the gate. `config.yaml` holds no
secrets (Rule 3), and neither does `.kapso/`.

## 5. Landing on today's code (v1)

- `gated_mcp/gates/hub_gate.py` + a `GateDefinition` in `presets.py`
  with `KAPSO_HUB_PATH` as injected env; `hub_ask` posts, blocks, runs
  the `check` each interval in a rebuilt environment, folds events,
  emits progress notifications; added to the shipped modes'
  `ideation_gates` and `implementation_gates`.
- `execution/hub.py`: the record (append, fold, mask, the check runner
  with the rebuilt environment and the per-run cap).
- `coding_agents/adapters/claude_code_agent.py`: the hub server's entry
  in the written MCP config carries `timeout` (ms) derived from
  `blocked.wait_minutes`; the session env carries
  `CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT` at the same ceiling; the poll loop
  tails the hub file and holds the deadline across `wait_started` →
  `wait_ended`; `session_id` from `system/init` lands in the node's
  telemetry.
- `coding_agents/adapters/codex_agent.py`: `tool_timeout_sec` in the hub
  server's `-c` overrides; the same deadline hold; `--json` added and
  `thread_id` recorded (the final message still comes from
  `--output-last-message`).
- `search_strategies/base.py`: the hub item id and the CLI session id on
  `SearchNode`.
- `orchestrator.py`: a wait callback from the adapters → ledger pause,
  `waiting` in `EvolveStatus`, `on_status` payload with the open ask.
- `cli.py`: `kapso hub` (`list`, `show`, `resolve`, `decline`); `watch`
  rendering of `waiting` and open items.
- `config.yaml` `defaults.blocked`: `policy` (wait | park),
  `wait_minutes`, `recheck_seconds`, `check_timeout_seconds`,
  `max_asks_per_session`. Benchmark modes set `policy: park`.
- Prompts: the one exception to "do not ask questions"; the rules of
  §4.6; the load-the-`.env` line in the tool result.
- Docs: `docs/evolve/` gains a page; `docs/reference/cli.mdx` and
  `configuration.mdx` gain the verb and the block.

v1.1, on the same record: the continuation by CLI session resume at the
boundary; `hub_post` for questions and notices; park and re-queue; the
selector's access criterion and runner-up; the preflight sources and the
environment inventory.

## 6. Open decisions

1. **v1 scope** — settled (user, 2026-09-03): the in-session wait only;
   a wait that runs out ends in the existing report.
2. **The continuation (v1.1).** CLI session resume of the very session
   that timed out — recommended: context on disk for 30 days, survives a
   kapso restart, no idle process — versus a fresh session on the same
   branch with a written summary. Record `session_id` / `thread_id` in
   v1 either way.
3. **Ideation MCP for codex members.** Give the codex ideation runner
   the gate (the adapter can carry MCP) or keep them on the report.
   Recommendation: the report for v1.
4. **Where the value is loaded on `met`.** The tool result tells the
   session to load `.env` itself — recommended: the value never passes
   through the gate — versus the gate returning the value into the
   transcript (never: it would land in the stream artifact).
5. **Defaults.** `wait_minutes` 30, `recheck_seconds` 15,
   `check_timeout_seconds` 30, `max_asks_per_session` 3. Guesses to
   confirm.

## 7. Out of scope for v1

The continuation (§4.3 step 4); `hub_post` for questions and notices;
park and re-queue; the selector's access criterion and runner-up; the
preflight sources and the environment inventory; the hosted inbox and
push notifications (the hook and the record are built for them); asking
through the feedback judge (the judge stays tool-locked and card-blind
by design); questions from the learning crews.
