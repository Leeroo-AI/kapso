# Evolve hub — the proxy between evolve modules and the human

**Status:** DESIGN v1 (2026-09-03) for review. Nothing built. Written
against the shipped code at a9d127ed (0.4.2), on branch `notif-evolve`.
Companion: the user-flow simulation ("Waiting on You", ten situations
with terminal mock-ups) — §2 carries its condensed form.

**Decision driver (user, 2026-09-03):** when evolve is blocked on access
or credentials an idea needs, critical cases wait for the user and then
continue from where they left off. The ask should go through a tool given
to the ideation and implementation modules, posting to a **hub**; the
user works the hub one item at a time; the hub is the proxy between
evolve modules and the human. This document compares that with the other
channels we have and pins the hub's shape.

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
copy-pasteable fix and a probe that decides when to continue; secrets
never pass through Kapso (names of needs only — the value goes to the
`.env` the run loaded, the probe verifies it); works with nobody
watching; faking is never the cheaper path; ask once.

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
| Keeps the session's context while waiting | no (branch + notes carry it) | **yes** | no by default; `hub_wait` later would | n/a | no |
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
die without posting) and defers B (`hub_wait` would be one more tool on
the same record, only worth building for attended runs once the rest
works). D stays the first line of defence — the cheapest ask is the one
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
| `need` | something only a person can grant: a credential, a login, a licence click, a permission, credits, an install on this box | `key` (dedupe, e.g. `env:OPENAI_API_KEY`), `for` (node or round + idea), `hit` (the concrete error), `worth` (why it is worth the human's time), `fix` (copy-pasteable), `verify` (a command whose exit 0 means satisfied), `alternatives` (none / what), `critical` | the probe passing (auto, at any re-check), or the human declining with an optional note |
| `question` | a decision the human should make: permission, scope, preference | the question, `options`, `assume` (what the module proceeds with if unanswered), `for` | the human answering; the answer becomes campaign context |
| `notice` | something left for the human, no action required by evolve: an orphaned better result on disk, a feature dropped for lack of a key (W&B logging), an infeasible-here idea | text, `for` | reading it (`seen`) |

Every item names its `module` (`ideation`, `implementation`, or
`orchestrator` for backstop-raised items) and its `node` or `round`.
No item ever carries a value. Probe output is stored with anything
secret-shaped masked.

### 4.2 Writers

**The `hub` gate.** One more bundled gate in `gated_mcp/presets.py`, given
to ideation and implementation sessions (never to the feedback judge):
`hub_post(type, ...)` returns `{id, status: "open"}` immediately, and
`hub_open()` lists the campaign's open items so a session does not
re-raise a `need` the previous session already filed (dedupe by `key`
happens server-side too). The gate appends to `.kapso/hub.jsonl` through
an env-injected path (`KAPSO_HUB_PATH`), exactly the bank gate's pull-log
pattern. Both CLIs already carry MCP for build sessions (the Codex
adapter forwards `mcp_servers` overrides); codex ideation members run
without MCP by design and use the second writer.

**The session-end report.** The implementation contract gains one
sanctioned exception to "do not ask questions": when blocked, commit the
partial work, write the next steps, and end with a `<blocked>` block
carrying the same fields as a `need`. The strategy files it into the hub
at extraction time. This is the path for sessions without the gate and
the fallback when a session ends without having posted.

**The backstop.** A session that dies without either — killed at its
deadline, crashed — already gets its difficulties reconstructed from the
stream (`difficulties_fallback.md`). The reconstruction also classifies
an authentication or permission signature and files the `need` on the
session's behalf, module `orchestrator`.

### 4.3 The boundary pass

At every iteration boundary the orchestrator:

1. reloads `.env` from the file the run loaded at start, so a value
   added since is visible to probes and to the next session;
2. runs the probe of every open `need` — passing probes resolve their
   items and re-queue the nodes parked on them; a probe that runs on a
   present-but-wrong value records the masked error on the item;
3. applies the rule of §2 to nodes blocked on still-open needs: critical
   → `waiting` (the ledger's clock paused, the status file at
   `waiting`, re-check every `blocked.recheck_seconds`, up to
   `blocked.wait_minutes`, then a checkpoint with
   `last_stop: needs_input` and `stopped_reason: needs_input`); otherwise
   → parked, campaign continues;
4. hands answered `question`s and human `note`s to the next ideation
   round and the next build session as campaign context — the same
   channel `current_feedback` uses today;
5. on `--resume`, runs the probes first, then continues the same node on
   the same branch with a second session told which need is now met.

Ideation gets the cheapest ask point this way: a candidate that names a
need is filed at selection, and the boundary decides before any build
session whether to wait briefly (critical, attended) or run the next
candidate and park this one.

### 4.4 Readers

| Surface | Shows or does |
|---|---|
| terminal (attended) | each new item as it arrives, in the preflight row format (`[NEED]` / for / hit / worth it / fix / verify); a critical wait shows the countdown and takes `Enter` (check now), `s` (decline, optional note), `q` (pause now) |
| `kapso hub <campaign>` | `list` (open first, critical flagged, age), `show <id>`, `resolve <id>` (runs the probe), `decline <id> [note]`, `answer <id> <text>`, `note <text>` (a free-standing hint for the next round), `seen <id>` |
| `kapso watch` | `hub: 2 open (1 critical)` plus the newest item's fix line; `WAITING` and `PAUSED` states |
| `on_status` | the status dict gains `hub: {open, critical, newest}`; a Slack post is a few lines of caller code; the platform's push notifications hang off the same hook |
| end summary | "Parked — needs you" (fix + the one `--resume` command per item), "Questions you answered", "Notices" |
| experiment history / ideation | blocked nodes render as `BLOCKED: needs <key> (pending | declined)`; declined keys join the environment inventory as *declined by user* so no candidate proposes them again |
| preflight / `doctor` | the evaluation suite as a required source, the goal as an advisory source, and the environment inventory (names of credentials and logins present, hardware found) handed to ideation and the selector |

### 4.5 Rules for modules

Written into the ideation and implementation prompts, and enforced where
the code can:

- Post only what a person must do. Installing a package, downloading
  public data, retrying a rate limit are the session's own job.
- A `question` always states its `assume`; the session proceeds on it.
- A `need` is load-bearing or it is a `notice`: W&B logging without a
  key is dropped and noticed, not asked for.
- Never stub, mock, fabricate the resource, or search the machine for
  credentials. Reporting the block must be the cheaper path.
- One item per key per campaign (`hub_open()` first; server-side dedupe).
- Per-session cap on posts, config-sourced like the change-request cap.
- Transient versus human: rate limits retry; billing and auth states ask.

### 4.6 Secrets

The hub holds needs. Values go into the `.env` file the run loaded, whose
path every ask prints. Kapso re-reads that file at re-checks and hands
the environment to sessions as it does today. Probe output is masked
before it is stored or shown. `config.yaml` holds no secrets (Rule 3),
and neither does `.kapso/`.

## 5. Landing on today's code

- `gated_mcp/gates/hub_gate.py` + a `GateDefinition` in `presets.py`
  with `KAPSO_HUB_PATH` as injected env; added to the shipped modes'
  `ideation_gates` and `implementation_gates`.
- `execution/hub.py`: the record (append, fold, mask, probe runner).
- `search_strategies/base.py`: a `blocked` outcome on `SearchNode`
  beside `had_error` and `evaluation_valid`, carrying the hub item id.
- `generic/strategy.py` and `implementation.py`: file `<blocked>` into
  the hub at extraction; the resume-session prompt for a met need.
- `orchestrator.py`: the boundary pass (§4.3); `waiting` in
  `EvolveStatus`; `needs_input` joins `VALID_LAST_STOPS`; the ledger
  gains a pause; `stopped_reason: needs_input` through `SolveResult` to
  `SolutionResult.metadata` plus a `needs` list.
- `core/preflight.py`: evaluation-suite and goal sources; the
  environment inventory.
- `cli.py`: `kapso hub`; the attended wait loop; `watch` rendering.
- `config.yaml` `defaults.blocked`: `policy` (wait | park | fail),
  `wait_minutes`, `recheck_seconds`, `max_posts_per_session`. Benchmark
  modes set `policy: park`.
- Docs: `docs/evolve/` gains a page; `docs/reference/cli.mdx` and
  `configuration.mdx` gain the verb and the block.

## 6. Open decisions

1. **Ideation MCP for codex members.** Give the codex ideation runner
   the gate (the adapter can already carry MCP) or keep them on the
   session-end writer. Recommendation: keep them on the report for v1.
2. **Where `hub.jsonl` lives across campaigns.** Per campaign under
   `.kapso/` for v1; a user-level `kapso hub` that scans known campaigns
   later; the hosted inbox after that.
3. **`hub_wait`.** Defer. Attended runs get the terminal wait loop at the
   boundary; the in-session variant only matters once the record and the
   CLI exist.
4. **Defaults.** `wait_minutes` 30, `recheck_seconds` 15,
   `max_posts_per_session` 3. Guesses to confirm.
5. **Goal scanning.** Keep as advisory rows; evaluation-suite scanning
   stays required.

## 7. Out of scope for v1

The hosted inbox and push notifications (the hook and the record are
built for them); `hub_wait`; asking through the feedback judge (the judge
stays tool-locked and card-blind by design); questions from the learning
crews (a later profile of the same record).
