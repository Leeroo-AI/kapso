---
name: kapso
description: Operate Kapso (PyPI leeroo-kapso), the self-improving AI software factory, from a coding session — verify the install with kapso doctor, launch and follow an evolve campaign toward a scored goal, answer a campaign that is WAITING ON YOU, resume an interrupted run, learn from a finished campaign into the lesson bank, ingest repos and research into the knowledge graph, and deploy the winner. Use when the user mentions Kapso or kapso evolve, learn, research, deploy, doctor, watch, inbox, bank, or asks to push a measurable metric (accuracy, latency, score) in a project where kapso is installed. Do not use for ordinary code edits, or for hand-tuning a metric when Kapso is neither installed nor asked for.
---

# Kapso

Kapso runs experiment campaigns: propose a candidate, implement it on its own git
branch, score it with the judge, keep the best, repeat. Everything below is 0.4.x.
Prefer the facts here and the docs links at the end over reading the package source;
the source is large and a session that greps it runs out of turns before answering.

## Facts that prevent the common mistakes

- Install with `pip install leeroo-kapso`. The PyPI package named `kapso` is an
  unrelated WhatsApp tool that shadows the `kapso` command. Python 3.10+.
- Every model call runs through a coding-agent CLI that must be logged in:
  `claude` (ideation, implementation, judging) and `codex` (research, utilities).
  There is no API-key fallback. `OPENAI_API_KEY` is used only for embeddings.
- Secrets come from `.env` in the directory you run from (`find_dotenv(usecwd=True)`);
  a shell export works only if it is actually exported. Config never holds secrets.
- All knobs live in one YAML. To change anything, copy the packaged file, edit, and
  pass it with `--config` / `config_path=`. Never set Kapso behaviour through
  environment variables.
- Run `kapso doctor` (or `kapso doctor evolve|learn|research|learn_knowledge|deploy`)
  before any run. Every required row must read `[OK ]`; `[-- ]` rows are optional.
  Add `--models` to live-probe every configured model with one token each; a model
  the login cannot serve fails here in seconds instead of hours in. A usage cap on
  a model it can serve is not visible to the probe.
- A campaign takes tens of minutes to hours. Launch it in the background with its
  output in a log file, check once that the process is alive, and hand off in
  your reply right away: the goal you passed, the metric and target you assumed
  and that the user's own can replace them, the `kapso watch <campaign> --follow`
  command, and that `WAITING ON YOU` is a pause answered through `kapso inbox`.
  The status file appears only after the seed copy and repo-memory bootstrap,
  usually a few minutes in, so do not wait for it, poll the log, or arm
  monitors unless asked. Never run `kapso evolve` in a foreground tool call.
- A run that ends with `WAITING ON YOU` has paused, not failed (exit code 0).
  Do not restart it; reply through the inbox (below).

## Evolve: run a campaign

### The goal text

The judge stops the campaign only when the goal reads as fully achieved, and it
scores every experiment from what the evaluation prints. So the goal has to name
a success metric and a number, and it helps to name the judge: "accuracy above
0.85 as measured by eval/evaluate.py". Both are the user's to give and optional;
whatever the user gives goes into the goal verbatim.

First look for an evaluation the repo already has: a scoring script, a test
suite, a benchmark command. Then:

- The repo has one but the request names no metric or target: run it once for
  the baseline, write the goal with that metric and a target you propose, and
  tell the user in one line what you assumed and that their own success metric
  goes straight into the goal. Leave headroom below any ceiling you can see; a
  target the data cannot reach ends in the inbox, not in a result.
- The repo has none and the request names none: ask once, in one line, whether
  they have a script or command that scores this and what "good" means, and
  end your turn there. A script they have goes in with `--eval-dir`,
  protected. If they answer that they have none or do not know, launch with
  the best metric you can state in the goal and no `--eval-dir`: the campaign
  builds its own evaluation in `kapso_evaluation/` from the goal, and the
  judge checks that it is fair. Never write an evaluator on the user's behalf,
  and never ask twice.
- Put rules in the goal, not methods. The judge enforces data rules and
  prohibitions ("do not modify eval/evaluate.py", "do not fit on
  data/test.csv") as invariants for the whole campaign, and would freeze a
  method preference the same way.

### Launch

```bash
kapso doctor evolve
nohup kapso evolve \
  --goal "Get the test accuracy of the churn model in train.py above 0.85 as measured by eval/evaluate.py. The evaluator must not be modified." \
  --initial-repo . \
  --eval-dir eval \
  --data-dir data \
  --output ../churn-campaign \
  --time-budget-minutes 60 --iterations 10 \
  > ../churn-campaign.log 2>&1 &
kapso watch ../churn-campaign --follow
```

- `--initial-repo <path|github url>` seeds the campaign; a non-git directory is
  copied and committed as the baseline. The original is never modified. Every
  experiment lands on branch `generic_exp_N` inside `--output`, which must be an
  empty directory outside the repo being seeded (a sibling); an output inside
  the repo gets copied into itself.
- `--eval-dir` is copied to `kapso_evaluation/` and integrity-protected: any
  candidate that edits it is rejected and unscored. Use it whenever the user has a
  judge. `--data-dir` is copied to `kapso_datasets/`. The seed copy includes
  everything in the directory, `.env` included.
- Bound the run: `--iterations` (default 10), `--time-budget-minutes` (durable
  across resumes), `--cost-budget` (best effort; codex sessions report no cost).
- `-m GENERIC` (default, knowledge search on) or `-m MINIMAL`; `-a` picks the coding
  agent (`claude_code` default, `codex`, `gemini`, `openhands`, `oss_claude_code`).
- The summary block ends `COMPLETED` with score and stop reason, or `WAITING ON YOU`.

Python, same thing:

```python
from kapso import Kapso
solution = Kapso().evolve(
    goal="...", initial_repo=".", eval_dir="eval", data_dir="data",
    output_path="./campaign", time_budget_minutes=60,
)
print(solution.explain())   # .final_score .succeeded .code_path .requests
```

`evolve()` blocks for the whole campaign, so run scripts with nohup as well.

## Inbox: the campaign is waiting on you

A session that needs something only a person can provide (a credential, access, a
file) records a request and the campaign pauses.

```bash
kapso inbox ./campaign                 # the open requests: key, hit, tried, fix, next
kapso inbox reply ./campaign 1 "added the key to .env"
```

- The id may be omitted when one request is open; an empty note means done.
- The reply resumes the very session that asked, in the foreground of that command.
- A credential-shaped reply is refused: put the value where `fix` says (usually
  `.env`) and reply with a note.

## Resume an interrupted campaign

```bash
kapso watch ../churn-campaign                  # DEAD (pid gone) or STALLED means it died
kapso evolve --output ../churn-campaign --resume
```

The checkpoint in `.kapso/run_state.json` carries the goal and the search
state; the launch record `.kapso/launch.json` carries every flag of the launch,
and a resume reads both, so pass only what you mean to change (a bigger
`--time-budget-minutes`, say). A resume that changes the mode, coding agent,
eval-dir, config or knowledge index is refused by name; start a new campaign in
a new output path instead. A campaign paused by the inbox resumes through
`kapso inbox reply`, not `--resume`.

## Learn: bank what a campaign taught

`learn()` mines one finished campaign into the lesson bank, a local git repo of
evidence-priced cards. It runs crews for an hour or more and needs both CLIs.

```python
from kapso import Kapso
k = Kapso()
lesson = k.learn("./campaign")      # or learn(solution); a trajectory id also works
print(lesson.explain())             # cards created/updated, admitted, report paths
print(k.memory.explain())           # bank head, active cards, serving flag
```

- The bank is created on first use at `learning.bank.local_path`
  (`~/.kapso/bank.git`). Share it with `kapso bank connect <git-url>` or
  `kapso bank create org/name`; after that every `learn()` pushes.
- Serving is off by default. For the next campaign to read the bank, set
  `learning.serving.enabled: true` in your config and pass it. Without that,
  banked cards have no effect on `evolve()`.
- Follow a learn with `kapso watch learning/status`.
- The `kapso learn ...` CLI subcommands (import, mine, grade, update, develop,
  codify, gauntlet, behave) are the learner-development regime, not the everyday
  path. From a shell the everyday path is the two Python lines above.

## learn_knowledge: import outside knowledge

Repos and research become wiki pages in a knowledge graph. This is a different
memory from the lesson bank.

```python
from kapso import Kapso, Source
k = Kapso()
findings = k.research("gradient boosting for churn", mode=["idea", "implementation"], depth="deep")
k.learn_knowledge(Source.Repo("https://github.com/org/repo"), findings, wiki_dir="data/wikis")
index = k.index_kg(wiki_dir="data/wikis", save_to="data/indexes/churn.index")
Kapso(kg_index=index).evolve(goal="...")
```

Needs Weaviate on 8080 and Neo4j on 7687 (from a source checkout:
`bash scripts/start_infra.sh`). Ingest time scales with the material, not with
`depth`, and can run for hours.

## Research

```bash
kapso research --objective "..." --mode idea --mode implementation --depth deep -o findings.json
```

`mode` is `idea`, `implementation`, or `study` (repeat the flag); `depth` is `light`
or `deep`. In Python `research(objective, mode=[...], depth=...)` with keyword-only
`mode` and `depth`; pass results into a campaign with
`evolve(context=[findings.to_string()])`. Runs on codex with web search.

## Deploy

Deploy runs one claude session that adapts a copy of the solution
(`<path>_adapted_<strategy>`) for the target and returns when it is ready. It
takes a few minutes, not hours: run it in the foreground with a long tool
timeout and use the result, rather than backgrounding it and polling.

```bash
kapso doctor deploy
kapso deploy --solution-path ../churn-campaign --strategy local   # auto|local|docker|modal|bentoml|langgraph
```

To call what was deployed, use the Python API, which hands back the running
software:

```python
from kapso import Kapso, DeployStrategy, SolutionResult
solution = SolutionResult(goal="churn model", code_path="../churn-campaign")  # or the object evolve() returned
software = Kapso().deploy(solution, strategy=DeployStrategy.LOCAL)
print(software.run({"tenure_months": 3, "monthly_spend": 92.0, "support_tickets": 4, "logins_per_week": 1.0, "plan": "basic"}))
software.stop()
```

The original solution is untouched. `AUTO` lets a selector pick the target.

## Models and config

```bash
python -c "from kapso.kapso import DEFAULT_CONFIG_PATH; print(DEFAULT_CONFIG_PATH)"
cp "$(python -c 'from kapso.kapso import DEFAULT_CONFIG_PATH; print(DEFAULT_CONFIG_PATH)')" kapso-config.yaml
# edit, then:
kapso doctor learn --models --config kapso-config.yaml
```

Shipped models: campaigns on `claude-opus-5`, learning crews on `claude-fable-5`,
codex roles on `gpt-5.6-sol`. Swap a model by editing every occurrence in the block
you care about (`learning.*` for the crews, `modes.<MODE>.*` for campaigns). The
`timeout_minutes` caps were calibrated on the shipped models; tell the user a slower
model may need them raised, but change only what was asked.
`Kapso(config_path="kapso-config.yaml")` and `--config` on every CLI verb select
your file.

## When Kapso is the wrong tool

A one-file change with an obvious fix is faster by hand. Reach for a campaign when
there is a judge to beat, many candidates worth trying, and the user accepts an
unattended run of an hour or more on their coding-agent subscription. Say which you
are doing and why; never launch a campaign the user did not ask for.

## Docs, served as Markdown

- CLI reference: https://docs.leeroo.com/docs/reference/cli.md
- Python API: https://docs.leeroo.com/docs/reference/kapso-api.md
- Configuration: https://docs.leeroo.com/docs/reference/configuration.md
- Inbox: https://docs.leeroo.com/docs/evolve/inbox.md
- Resuming runs: https://docs.leeroo.com/docs/evolve/resuming-runs.md
- Evaluation integrity: https://docs.leeroo.com/docs/evolve/evaluation-integrity.md
- Trajectory learning: https://docs.leeroo.com/docs/learning/overview.md
- Knowledge graph: https://docs.leeroo.com/docs/knowledge/overview.md
- Deployment: https://docs.leeroo.com/docs/deployment/strategies.md
- Index of every page: https://docs.leeroo.com/llms.txt

The whole site is also an MCP server:
`claude mcp add --transport http kapso-docs https://docs.leeroo.com/mcp`.
