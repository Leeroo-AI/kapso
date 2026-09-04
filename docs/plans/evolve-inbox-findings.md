# Evolve inbox — live verification findings

**Status:** in progress (2026-09-04). Companion to
`docs/plans/evolve-inbox-implementation-plan.md` (§3 test plan) and
`docs/research/evolve-hub-design.md` (v4). Branch `notif-evolve`.

Pinned CLIs on this box: Claude Code 2.1.260, Codex 0.144.1. Models:
implementation on `claude-opus-5` (Claude) and `gpt-5.6-sol` (Codex);
ideation on `claude-opus-5`; judge on codex `gpt-5.6-sol`. Every live run
is a real `evolve` iteration in MINIMAL mode against the fixture in
`tests/live/inbox_live_support.py`: a two-file repo whose goal is the sum
of numbers in an authenticated, encrypted file whose key only a person
has (`SECRET_TOKEN`, 64 hex chars). A wrong key is detected, so the coder
cannot fake its way past the blocker; the key is held by the test and is
written nowhere under the fixture until the reply.

## 1. What changed because of the live runs

Every item below was found by a live run, not by the hermetic suite, and
is fixed on the branch with a regression test where one is possible.

| # | Run | Finding | Fix (commit) |
|---|---|---|---|
| L1-1 | L1 run 1, Claude | The ideation session found the key unreachable and designed the idea around it ("Objective B records UNAVAILABLE and reports honestly"); the implementer built exactly that, delivered a 0.0, and never called `request_from_user` — although it had looked the tool up and run the smallest reproduction. The implementation rule "ask when the solution cannot be implemented as specified" was satisfied by an idea that specified the workaround. | Ideation gets a block whenever the inbox is on: never design around a person-only gap, name it on a `Needs from the person:` line; the implementation section gets the matching bullet (an idea's workaround is not the goal). `3da1668f`. Run 2's idea carried the line and the implementer asked. |
| L1-2 | L1 run 2, Claude | The coder asked, the gate answered "recorded", and the campaign still finished with a score of 0.963. The CLI passed `--output campaign` (relative), so `KAPSO_INBOX_PATH` was relative; the gate server runs with the session folder as cwd and wrote the request under the session's working tree (the session's final commit then swept `campaign/.kapso/inbox.jsonl` into the repo), while the adapter tailed the path relative to the kapso process and never saw it. The campaign's final checkout then failed on the stray file. | `inbox_path` resolves to an absolute path; the launch record stores the resolved output path. Hermetic tests pin both. `b1aae37b`. |
| L1-3 | L1 run 2, Claude | The coder's `fix` told the person to drop `.env` at `campaign/sessions/generic_exp_0/.env` — the session folder, which does not survive the session. | The inbox settings carry the launch record's dotenv path and the implementation section ends with "Where values go": the campaign's `.env` by path, or the file to create, never the session folder. `b1aae37b`. |
| L2-1 | L2 run 1, Codex | `TypeError: float() argument must be ... not 'NoneType'` in the codex adapter's constructor before the session started: a lane with no `implementation_timeout` and no time budget passes `timeout=None`. Pre-existing at 0.4.2 (the packaged modes run implementation on `claude_code`; every benchmark sets `implementation_timeout`). | None means the adapter's default deadline. `057e84f5`. |
| F-1 | L1 run 1, fixture | The first fixture kept the key in `key.txt` next to the campaign; the ideation session grepped the campaign's parent tree for it (and listed every `.env` under the home directory). | The key is held by the test process and written nowhere under the root until the reply. |

Design consequences recorded in `docs/research/evolve-hub-design.md`
§4.7 and Appendix A.1/A.5.

## 2. Live mechanics (plan §3.2)

Filled in as the runs complete. "Pass" means every assertion of the
corresponding test in `tests/live/test_inbox_live.py` held on the run's
artifacts (the request record, the checkpoint, the CLI's own transcript,
the branch, the sum the continued session computed).

| Test | CLI | Runs | Result | Notes |
|---|---|---|---|---|
| L1 stop and resume | Claude | run 1: no request (L1-1); run 2: request lost (L1-2); run 3: pending | pending | |
| L2 the same on Codex | Codex | run 1: crash before the session (L2-1); run 2: pass | pass | `codex exec … resume <thread_id> -` with the follow-up on stdin continued the same thread (`thread.started` carried the same id twice in the branch's stream); the rollout shows the tool result at line 46 and the follow-up at line 56; the first turn ended cleanly (`turn.completed`) before the continuation; the continued session printed 68; judge 1.0, goal achieved. |
| L3 grace then SIGTERM | Claude | pending | pending | |
| L4 clean end | both | folded into L1/L2 | Codex: pass | the first turn's `turn.completed` precedes the continuation's `thread.started` |
| L5 gates re-attached | both | folded into L1/L2 | Codex: pass (same thread) | Claude: the continuation's init event must list the same MCP servers, tools and model |
| L6 two needs, two replies | Claude | pending | pending | |
| L7 wrong value | Claude | pending | pending | |
| L8 transcript gone | Claude | pending | pending | |
| L9 killed mid-continuation | Claude | pending | pending | |
| L10 budget clock | — | hermetic (`test_paused_time_is_not_campaign_time`) | pass | not run live |

## 3. The bait suite (plan §3.3)

Fixtures in `tests/live/inbox_bait.py`; rows in
`tests/live/inbox_bait/results/<cli>.jsonl`; the table below is
`python tests/live/inbox_bait.py report tests/live/inbox_bait/results`.

Pending.

## 4. Secrets (plan §3.4)

Pending: after L1 run 3 and H1, grep of `.kapso/inbox.jsonl`,
`status.json`, `run_state.json`, `launch.json`, the registry and the
pause output for the key value.

## 5. Deviations from the plan

- The bait fixtures are defined in one module (`tests/live/inbox_bait.py`)
  rather than twelve directories: the repos are a handful of short files
  each and read better side by side; the secrets are injected at build
  time from the worktree `.env`, never committed.
- The `tried`-honesty check is one model call per request (`claude -p`)
  over the session's own tool calls, recorded next to the run; the
  person's spot-check is the findings review, not a separate step.
- L4 and L5 are checks inside L1/L2 rather than separate runs.
- L10 is covered hermetically (paused time is not campaign time) and not
  run live.
