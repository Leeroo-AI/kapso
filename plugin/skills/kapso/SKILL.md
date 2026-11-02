---
name: kapso
description: Operate Kapso (PyPI leeroo-kapso), which turns coding-agent output into verified outcomes against a measurable objective, from a coding session — launch a build and hand it off, follow it with kapso watch, answer a campaign that is WAITING ON YOU, tell a slow run from a dead one and stop or resume it, learn from a finished campaign into your bank, serve a build or this session the experience it should start from, deploy the winner, and run the harness and experience verbs. Use when the user mentions Kapso or kapso build, watch, inbox, resume, stop, learn, experience, harness, deploy or doctor, asks how a campaign is doing, or says a run died. Do not use for ordinary code edits, for writing the objective and its evaluation (kapso-build does that), for reading a finished run's result (kapso-review does that), or inside a Kapso campaign session, which never launches another campaign.
---

# Kapso

Kapso runs experiment campaigns: propose a candidate, implement it on its own git
branch, score it with the judge, keep the best, repeat. These facts are written for
leeroo-kapso 0.8, the version this plugin pins. Prefer them and the docs links at
the end over reading the package source; a session that greps it runs out of turns.

## Facts that prevent the common mistakes

- Install with `pip install leeroo-kapso` (or `uv tool install leeroo-kapso`). The
  PyPI package named `kapso` is an unrelated WhatsApp tool that shadows the `kapso`
  command. Python 3.10+. The `kapso-setup` skill does the whole install.
- Every model call runs through a coding-agent CLI that must be logged in:
  `claude` (ideation, implementation, judging) and `codex` (research, utilities).
  There is no API-key fallback. `OPENAI_API_KEY` is optional and only ever embeds
  text (a corpus ingested with `kapso experience add`, semantic search over past
  experiments once `models.embedding` is set).
- `LEEROOPEDIA_API_KEY` in `.env` is the hosted knowledge base of ML and AI
  frameworks (nothing to install; a new account at
  https://app.leeroopedia.com/dashboard has $20 of free credit). A build reads it
  when it loads it: `--experience leeroopedia`.
- Secrets come from `.env` in the directory you run from; a shell export works only
  if it is actually exported. Config never holds secrets.
- The config is the packaged file plus one override file, `kapso-config.yaml` at the
  project root (the git top level), holding only the keys you change. Pass it with
  `--config kapso-config.yaml` on every `kapso` line below whenever the project has
  one; a build launched without it runs on the packaged defaults while `doctor`
  shows the override, and a resume that changes the config is refused by name.
  Never set Kapso behaviour through environment variables.
- Run `kapso doctor build --json --config kapso-config.yaml` (or `doctor learn|deploy|
  experience`) before a run: every row with `"required": true` must be `"ok": true`;
  the rest are optional. Add `--models` to live-probe every configured model with one
  token each.
- A campaign takes tens of minutes to hours. Launch it with `--detach`: the command
  returns at once with the pid, the log, the status path and the `watch` line, and
  the run outlives your tool call on every host. Hand off in your reply right away:
  the brief you passed, the metric and target you assumed and that the user's own can
  replace them, the `kapso watch <campaign> --follow` line, and that `WAITING ON YOU`
  is a pause answered through `kapso inbox`. The status file appears within seconds
  (phase `setup` first, minutes long), so do not poll the log or arm monitors unless
  asked. Never run `kapso build` without `--detach` from a tool call.
- When the host's shell tool cannot keep a child alive past the call (a sandbox
  that ends it) or Bash is restricted, launch through the session's
  `kapso_build_start` tool instead; `kapso_status` and `kapso_stop` are the watch
  and the stop the same way.
- A run that ends with `WAITING ON YOU` has paused, not failed (exit code 0). Do not
  restart it; reply through the inbox (below).
- A build is served exactly the experience it loads (`--experience`, the brief's
  `Experience` line) and nothing when it loads nothing. This session is served what
  the project's config names under `experience.session_sources`.
- You are a host session. A session Kapso itself started (a build's ideation,
  implementation or judge session, a deploy session) never launches a campaign, a
  learn or a deploy: that would be a campaign inside a campaign.

## Build: launch a campaign

The objective, its evaluation and the brief are the `kapso-build` skill's: it reads
the repository, writes `kapso-brief.md`, gets one approval and launches. When the
brief exists and the user asks for a launch, these are the lines:

```bash
kapso doctor build --json --config kapso-config.yaml
kapso build \
  --brief kapso-brief.md \
  --repo . \
  --evaluation evaluation \
  --data data \
  --output ../churn-campaign \
  --budget-minutes 60 --budget-experiments 10 \
  --experience mine \
  --config kapso-config.yaml \
  --detach
kapso watch ../churn-campaign --follow --config kapso-config.yaml
```

- `--detach` starts the campaign in a session of its own and prints its pid, the log
  (`../churn-campaign.log`) and the `watch` line; no `nohup`, `setsid` or `&` is
  needed, on any host. The launch is a tool call of its own: nothing chained in front.
- `--repo <path|github url>` seeds the campaign; a non-git directory is copied and
  committed as the baseline. The original is never modified. Every experiment lands
  on branch `generic_exp_N` inside `--output`, an empty directory outside the repo
  being seeded (a sibling), so the seed copy never contains the campaign.
- `--evaluation` is copied to `kapso_evaluation/` and integrity-protected: a
  candidate that edits it is rejected and unscored. `--data` is copied to
  `kapso_datasets/`. Keep the hold-out and its labels under the evaluation
  directory, never under `--data`. The seed copy includes everything in the
  directory, `.env` included.
- Bound the run: `--budget-experiments` (default 10), `--budget-minutes` (durable
  across resumes), `--budget-usd` (best effort; codex sessions report no cost). A
  flag wins over the brief's `Budget` line.
- `--experience mine,leeroopedia` names what the build is served; it wins over the
  brief's `Experience` line, and each source is loaded before the fork, so one that
  does not exist refuses here, not in the child's log. `--harness <ref>` names the
  domain layer the same way (a bundled name, a directory, a git reference).
- `-a` picks the host every role without a `cli` of its own runs on (`claude_code`
  as packaged, `codex`).
- The summary block ends `COMPLETED` with score and stop reason, or `WAITING ON YOU`.

Python, same thing:

```python
from kapso import Kapso, Brief, Budget, Solution
k = Kapso(config="kapso-config.yaml")
k.load_experience("mine")
solution = k.build(
    Brief.read("kapso-brief.md"), repo=".", evaluation="evaluation", data="data",
    output="./campaign", budget=Budget(minutes=60),
)
print(solution.explain())   # .score .goal_achieved .branch .campaign .experience .diff()
Solution.open("./campaign") # the same solution from any process, once the build returned
```

`build()` blocks for the whole campaign: from a tool call prefer the CLI with
`--detach`, or start the script in a session of its own.

## Follow a campaign

```bash
kapso watch ../churn-campaign --json --config kapso-config.yaml
kapso watch --config kapso-config.yaml                   # every campaign of this machine that is alive or waiting
```

Answer "how is it going" in three lines from `--json`: the state and phase with the
iteration, the best score and its branch against the brief's baseline and target, the
budget left — the launch's minutes and experiments less what the status shows used,
read from the launch record when the status does not carry them, never "not
reported"; add what the last attempt consulted when the user asks what it is doing.
Each line is a fact the user did not have, with its figure. A number you state — a
baseline, a target — names its source (the brief's line, the file it was measured
into), and when the records disagree (a README's claim against a measured baseline)
say so rather than choosing one in silence. A figure the user asserts against the
measured record ("the baseline is 0.90") gets which is measured and which is a claim,
with the file behind the measured one; the campaign stands on it, never conceded. `--follow` prints the lines
as they come; it is for the user's own terminal, not for a tool call. A `starting`
line before the status exists is a detached launch still seeding; `stopped before
writing a status` means the launch died, and the log says why.

## Inbox: the campaign is waiting on you

A session that needs something only a person can provide (a credential, access, a
file) records a request and the campaign pauses.

```bash
kapso inbox ./campaign --config kapso-config.yaml                                    # the open requests: key, hit, tried, fix, next
kapso inbox reply ./campaign 1 "added the key to .env" --detach --config kapso-config.yaml
```

- The id may be omitted when one request is open; an empty note means done.
- The reply resumes the very session that asked, in a session of its own with
  `--detach`; hand off with the `watch` line it prints.
- A credential-shaped reply is refused: put the value where `fix` says (usually
  `.env`) and reply with a note. Never carry a value through your reply.

## Slow, stalled, dead: diagnose, then stop or resume

Read `kapso watch <campaign> --json` before touching anything:

- `running` with a fresh heartbeat is a run at work; a long phase (an evaluation, a
  verification) is not a stall. Say where it is and leave it.
- `STALLED` is three missed heartbeats: the writer is alive but silent. Wait one more
  interval, then read the log's last lines.
- `DEAD` (the pid is gone) or `failed`: classify from the log's last lines before any
  action. A provider limit (`429`, quota, "usage limit") means wait and resume; an
  evaluator crash means the evaluation directory needs a fix before a resume; an
  environment error (a missing package, a missing key) is fixed first; a full disk
  is cleared first; a killed session (`Killed`, a signal) is resumed as it is. The
  answer that follows a fix names the fault, what was done about it (the space
  freed, the key written, the labels put back) and the resume itself.

```bash
kapso build --output ../churn-campaign --stop --config kapso-config.yaml      # end a running campaign; its checkpoint stays
kapso build --output ../churn-campaign --resume --detach --config kapso-config.yaml
```

- `--stop` ends the campaign's processes, marks the status `failed` with `stopped by
  the person`, and names the resume. A terminal status, or a writer already gone, is
  refused by state: that run is dead, not running, so resume it.
- A resume is a launch: `--detach` it and hand off with the `watch` line it prints.
  Do it in the same turn you diagnose the death in; a reply that only announces a
  resume leaves the campaign dead. The checkpoint carries the brief and the search
  state, the launch record every flag, so pass only what you mean to change (a bigger
  `--budget-minutes`, say). A resume that changes the host, the evaluation, config,
  harness or experience is refused by name; start a new campaign in a new output
  path instead. A campaign paused by the inbox resumes through `kapso inbox reply`,
  not `--resume`. Other `kapso build` processes belong to other campaigns; never
  stop or kill one that is not the campaign you were asked about.

## Learn: bank what a campaign taught

`learn` mines one finished campaign into your bank, a local git repo of
evidence-priced cards. It runs crews for an hour or more and needs both CLIs.

```bash
kapso learn ./campaign --detach --config kapso-config.yaml
kapso learn ./campaign --harness ./harness --detach --config kapso-config.yaml      # when the campaign's record names no harness
kapso watch learning/status --follow --config kapso-config.yaml
```

- The bank is created on first use at `learning.bank.local_path` (`~/.kapso/bank.git`),
  or by `kapso experience add mine`. Share it with `kapso experience share <git-url>`
  (create the repository yourself first); after that every learn pushes and
  `kapso experience update` pulls the team's lessons.
- The launch answer says what the user got, a line each with its figure: what the
  learn reads (the campaign, its attempts and best score), where it banks (the bank's
  path and cards now), how long it runs (an hour or more, the update crew most of it)
  and what lands (cards the next build or session is served) — never the pid alone.
- In the same reply as the launch, offer the one-line edit that serves the bank to
  every later session of this project once the learn has banked its cards:
  `session_sources: [mine:all]` under `experience:` in `kapso-config.yaml`. That
  serves sessions; a build loads what its brief or `--experience` names. Do not
  wait for the learn to end: hand off the `watch` line and stop.
- `kapso learn dev …` are the learning system's own tools, for people working on the
  learning system, not the everyday path.

## Experience: what a build, or this session, is served

```bash
kapso experience add mine --config kapso-config.yaml                                           # your bank, empty; learn fills it
kapso experience add leeroopedia --config kapso-config.yaml                                    # checks the key in .env
kapso experience add --repo https://github.com/org/repo --name org-repo --detach --config kapso-config.yaml   # a repository into an index (hours)
kapso experience add --research "gradient boosting for churn" --name churn-gbm --detach --config kapso-config.yaml
kapso experience list --config kapso-config.yaml                                               # mine, the hosted gate, banks you read, indexes
kapso experience share https://github.com/org/kapso-bank.git --config kapso-config.yaml
kapso experience update --config kapso-config.yaml                                             # pulls mine, re-fetches banks you read
kapso experience remove org-repo --config kapso-config.yaml                                    # an index, or a bank you read
```

- `share` answers with what happened, not the command's echo: the remote, the bank's
  head and card count pushed, the line a teammate runs to read it (`kapso experience
  add <url>`); `add` and `update` likewise say what the bank or index now holds.
- The forms: `mine` (your bank, the cards your harness's scope covers), `mine:all`
  (every card), `leeroopedia`, a bank path (`./team/bank.git`) or git reference
  (`acme/kapso-bank@v2`), an index name. A source that does not exist is refused
  with the `add` line that makes it; loading never ingests.
- Ingesting a repository or a question needs `kapso doctor experience` green
  (Weaviate on 8080, Neo4j on 7687, `OPENAI_API_KEY`, both CLIs); it runs for hours,
  so `--detach` it and follow with `kapso watch learning/status --follow`.
- This session's tools come from the plugin's `kapso` MCP server, which reads
  `experience.session_sources` from `kapso-config.yaml`; a session without the plugin
  gets the same with `claude mcp add kapso -- kapso-mcp --experience mine:all` or
  `codex mcp add kapso -- kapso-mcp --experience mine:all`. A source the config names
  that does not exist leaves the bank tools off, named in the server's instructions.

## Harness: the domain layer a build loads

Writing a harness from a repository, and improving one from a run, are the
`kapso-harness` skill's. The verbs around one:

```bash
kapso harness list --config kapso-config.yaml                       # bundled and fetched, with what each needs
kapso harness check ./trading --config kapso-config.yaml            # failures, warnings, what this machine lacks
kapso harness add acme/trading-harness@v1 --config kapso-config.yaml
kapso harness update acme/trading-harness@v1 --config kapso-config.yaml
kapso harness remove acme/trading-harness@v1 --config kapso-config.yaml
kapso harness export ./trading --host claude_code --to . --config kapso-config.yaml    # or --host codex: its skills, delegates and servers as this session's files
```

`check` passing means the harness is well formed; a `needs …` line is this
machine's, not the harness's. `export` writes the files an ordinary session
discovers and prints the include line for `CLAUDE.md` or `AGENTS.md`; hooks and
the module pages stay behind, and an existing destination refuses it by name.

## Review: what a finished run did

A finished run's questions (what won, why, is it safe to merge) go through the
`kapso-review` skill, which reads `kapso trace <campaign>` and the record. `kapso
trace ../churn-campaign` prints the attempts with their sessions, evaluations,
verdicts and judgments; `--json` is the same as data.

## Deploy

Deploy runs one session that adapts a copy of the solution to a target, deploys it
with the target's scripts, proves one call, and returns when the engine has checked
the result. It takes minutes (up to thirty, the session's deadline). Run it in the
foreground when your tool's timeout allows; when it does not (Claude Code's Bash tool
stops at ten minutes), run it once in the background writing to a log and wait for
the summary to appear there, never a second time. The record is written when the
deploy starts, so `--status` says what is deploying even while the call runs.

```bash
kapso doctor deploy --json --config kapso-config.yaml
kapso deploy ../churn-campaign --target local --config kapso-config.yaml       # auto|local|docker|modal|bentoml|langgraph, or a harness target
kapso deploy ../churn-campaign --status --config kapso-config.yaml             # every deployment of the campaign, with its state
kapso deploy ../churn-campaign --logs --number 1 --config kapso-config.yaml
kapso deploy ../churn-campaign --stop --number 1 --config kapso-config.yaml
kapso deploy ../churn-campaign --start --number 1 --config kapso-config.yaml
```

The summary names the deployment's number, how `run()` calls it, the call the
session proved (the example of what it takes), the engine's own answer to it, and
the next actions pinned to that number. To call what was deployed, reopen that
number from Python:

```python
from kapso import Software
software = Software.open("../churn-campaign", 1, "kapso-config.yaml")
print(software.proof["inputs"])                     # what run() takes: the call the session proved
print(software.run(software.proof["inputs"]))       # the same call; your own inputs take this shape
software.stop()
```

A scheduled target (the trading example's `paper-trading`) runs on its own; read
`--logs`, and `run()` runs one cycle on demand. The original solution is untouched:
the deployment lives under `<campaign>/.kapso/deployments/<n>/`. `--target auto` lets
the session choose among the targets this machine can serve and records why. The
local target runs the function in the engine's own interpreter: upgrading the
`leeroo-kapso` package replaces it, so the solution's requirements are reinstalled
there after an upgrade.

## Models and config

```bash
kapso doctor --show-config --config kapso-config.yaml      # the effective config: the packaged defaults with your file over them
kapso doctor learn --models --config kapso-config.yaml
```

The hosts table (`hosts.claude_code`, `hosts.codex`) names each CLI's model; the
three roles under `search_strategy.params` (`ideation`, `implementation`, `judge`)
and the `learning.*` crews take `cli`, `model` and `effort` of their own, else the
host's. Shipped: campaigns on `claude-opus-5`, learning crews on `claude-fable-5`,
codex roles on `gpt-5.6-sol`. To change a model, write only that key into
`kapso-config.yaml`; a slower model may need the role's `timeout` raised, but change
only what was asked.

## When Kapso is the wrong tool

A one-file change with an obvious fix is faster by hand. Reach for a campaign when
there is a judge to beat, many candidates worth trying, and the user accepts an
unattended run of an hour or more on their coding-agent subscription. Say which you
are doing and why; never launch a campaign the user did not ask for.

## Docs, served as Markdown

- CLI reference: https://docs.leeroo.com/docs/reference/cli.md
- Python API: https://docs.leeroo.com/docs/reference/kapso-api.md
- Configuration: https://docs.leeroo.com/docs/reference/configuration.md
- Inbox: https://docs.leeroo.com/docs/build/inbox.md
- Resuming runs: https://docs.leeroo.com/docs/build/resume-and-budgets.md
- The record: https://docs.leeroo.com/docs/build/trace.md
- Evaluation: https://docs.leeroo.com/docs/build/evaluation.md
- Learning: https://docs.leeroo.com/docs/learn/what-learn-produces.md
- Experience: https://docs.leeroo.com/docs/experience.md
- Harnesses: https://docs.leeroo.com/docs/harnesses.md
- Deploy: https://docs.leeroo.com/docs/deploy/targets.md
- Index of every page: https://docs.leeroo.com/llms.txt

The whole site is also an MCP server:
`claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp`.
