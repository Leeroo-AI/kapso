# Kapso 0.8.0 — release notes

Turn coding-agent output into verified outcomes, not review debt: plug in the harness for your domain, set the goal, every step traced.

This release lands the redesign as one change from the integration branch (`docs/plans/redesign/MASTER.md`). It renames the verbs, replaces the modes with one configuration, makes the harness the domain layer a build loads, makes experience something a build is served explicitly, turns deployment into target skills, records every session, ships a coding-agent plugin, ports the benchmarks' knowledge into two bundled harnesses, and reorganises the docs by what the reader is doing. The mapping below is complete for the surfaces that changed; a name on the left no longer exists. Every line was checked against `kapso --help`, each verb's help and the packaged configuration at the release candidate.

## The verbs

| Before | Now |
|---|---|
| `kapso evolve …` | `kapso build --brief kapso-brief.md [--evaluation DIR] [--data DIR] [--repo PATH] --output DIR [--detach] [--harness REF] [--experience REFS] [--budget-minutes N] [--budget-experiments N] [--budget-usd N] [--budget-reserve-minutes N] [--resume] [--stop]` |
| `kapso bank create`, `kapso bank connect`, `kapso learn init-bank` | `kapso experience add mine`, `kapso experience share <url>` |
| `kapso research "…"`, `kapso learn knowledge …`, `--knowledge`, `--kg-index`, `--no-kg` | `kapso experience add --research "…" --name <index>` / `kapso experience add --repo <url> --name <index>`; served to a build with `--experience <refs>` or the brief's `Experience` line; `kapso experience list`, `update`, `remove` |
| `kapso watch --tree` | `kapso trace <campaign>` (the attempts, the lessons, what was voided, what was delivered); `kapso watch <campaign> --follow` for a live view |
| (new, from `main`) | `kapso report <campaign> [-o FILE] [--top N]`: the campaign as shareable Markdown |
| `kapso deploy --strategy …`, `--interactive` | `kapso deploy <campaign> --target auto\|local\|docker\|modal\|bentoml\|langgraph\|<a harness's target> [--env K=V] [--option K=V] [--status] [--stop] [--start] [--logs] [--number N]` |
| `--mode`, `-m MODE`, the `modes:` block, `load_mode_config` | one packaged config plus one override file passed as `--config` |
| (new) | `kapso harness new\|check\|list\|add\|update\|remove\|export`; the brief's `Harness` line |
| (new) | `kapso inbox` / `kapso inbox reply [<campaign>] [<id>] "<note>"` for a build that is waiting on you; `kapso doctor` with `--show-config` |
| `kapso learn <subcommands>` | `kapso learn <campaign>` learns from a finished campaign; `kapso learn dev …` is the development pipeline |

## The Python API

| Before | Now |
|---|---|
| `Kapso(config_path=…)` | `Kapso(config=…)` |
| `kapso.evolve(...)` | `kapso.build(brief, evaluation=, data=, repo=, output=, budget=Budget(minutes=, experiments=, usd=, reserve_minutes=), harness=, experience=, resume=)` → `Solution` (`.score`, `.branch`, `.diff()`, `.explain()`, `Solution.open(path)`) |
| `kapso.research(...)`, `learn_knowledge(...)` | `kapso.add_experience(repo=\|paper=\|research=, name=)` then `kapso.load_experience(ref)` |
| `LessonResult` from `learn()` | `kapso.learn(solution)` → the lesson with its evidence (`.explain()`), banked under `mine` |
| the deploy strategies | `kapso.deploy(solution, target=)` → `Software` (`.run()`, `.proof`, `.record`, `.stop()`, `.start()`) |
| (new) | `kapso.load_harness(ref)`; `Brief.read(path)` |

## The configuration

- One packaged `config.yaml` with one override file per project; no modes and no `defaults:` layer (an unknown key is refused by name). `hosts:` carries each role's host and model; `harness:`, `experience:` and `deployment:` are new blocks; `deployment.target` names the default target; `report.leaderboard_size` sizes `kapso report`'s leaderboard.
- Retired: `serving.enabled` and `serving_scope` (what is loaded is what is served; a source's `scope` rides on its reference, `mine:all` lifts it), `strategies.yaml` and the deploy strategy classes (target skills instead), `budget.min_iteration_seconds_insured` and the handlers' insured floor, the `feedback_generator:` and `coding_agent:` blocks (routes per role under `hosts:`), `knowledge_search.use_knowledge_graph`.
- From `main` since the branch: `search_strategy.params.parent_policy: weighted` with `parent_sampling` (temperature, offspring penalty, seed); `repo_memory_update_payload_max_chars` (a changed file that is binary or larger than that budget on its own is listed by name and size in the update, never sent: the bundled example's dataset had put every update past Codex's input limit); one `expansion_lane_env` entry per lane.

## The records

- Every session of a campaign writes `session.json` (its module, pages, skills, hooks, runs with tools, hooks fired, cost and tokens); `harness.json` is gone. `.kapso/launch.json`, `status.json`, `solution.json` and the trace carry the harness and the experience a campaign ran with, every path resolved; a resume refuses a different list.

## The docs

- The site is organised by what the reader is doing: Build, Harnesses, Experience, Learn, Deploy, How it works, Benchmarks, Reference. Learn is two pages (`/docs/learn/what-learn-produces`, `/docs/learn/development-mode`); the six `/docs/learning/*` pages redirect. `/docs/agent-access` is part of `/docs/coding-agent-skills`; `/docs/deployment/*` → `/docs/deploy/targets`; `/docs/knowledge/*` → `/docs/experience`.
- Two bundled harness pages (`predictive-modelling` 0.3.3, `competitive-programming` 0.3.2); every benchmark page says who ran its numbers, on what and when; two claim lints hold every public sentence to the project's own rules.

## The plugin

- `claude plugin marketplace add Leeroo-AI/kapso && claude plugin install kapso@kapso` (Codex: `codex plugin marketplace add Leeroo-AI/kapso && codex plugin add kapso@kapso`): the commands `/kapso:setup`, `/kapso:build`, `/kapso:watch`, `/kapso:inbox`, `/kapso:resume`, `/kapso:review`, `/kapso:learn`, `/kapso:deploy`, `/kapso:harness`, `/kapso:experience`; six skills; the `kapso-mcp` server.

## The examples

- `full-loop` (build, the review, learn, build again served the lesson, deploy), `churn-first-build`, `fraud-no-evaluator`, `pipeline-dbt`, `trading` (on the trading harness under `examples/harnesses/`), `prompt-harness`, `agent-scaffold`, `gpu-kernels`, `torch-performance`; `examples/README.md` says which have a recorded result.

## What was measured, and what was not

- The semantic suite from the release wheel on fresh machines, run by us on 2026-10-30 (the reports `tests/semantic/reports/rel-*.md`, each read in `docs/plans/redesign/wp11-release.md`): S1 on both hosts FAIL (12/17 steps on Codex, 11/17 on Claude Code) with every failed step the attempt's clock, the persona or the rig and no wrong call of the package; S2 on Claude Code FAIL 17/25 (five answers over the suite's counts, the setup session's requests never marked continued, the second engineer's build refused on the bundled harness's requirements); S2 on Codex FAIL 17/24 (a plugin session that saw no kapso tools on the loaded machine, the trigger probe's Codex sandbox failing under its parallel launches, the hour's single attempt not fitting so nothing to deploy or learn at the steps' reads, the same two findings); S4 twice FAIL on the one bar no archived campaign can meet (a judge rejection beside the void), every reading step passing; S5 twice on Claude Code (the driver takes no host) 7/8 and 5/8, the tampered evaluator scoring no candidate in either and the integrity void with its hash evidence in the second. The gate found, and this release fixes, the repo-memory update that sent a campaign's dataset whole past Codex's input limit (the fix exercised live from the release wheel with the engine's own Codex session: three unreadable files withheld by name and size, two calls under the limit, the memory landed); it raised for the owner the bundled harnesses' requirements under a tool install, the setup session's inbox bookkeeping, the suite's counts against a plugin that waits, and a deadline-cut attempt's raw git error in its feedback. S6 (experience pays) from the suite's build; S3 (the harness ports) from the arms, each a quick run at the budget its record states beside the recorded one: the MLE quick arm (spooky-author-identification, three hours on one GPU) graded at a log loss of 0.2271 on the private answers, silver; the ALE quick arm (ahc008, three hours) accepted on every private case at a performance of 1363 against the recorded 1221; the RelBench six-hour arm (rel-trial/study-outcome, one A100, four lanes) at a test AUROC of 0.7555 on the lanes' registered finals, inside the recorded band, with no candidate delivered (an evaluator change request mid-iteration, which benchmark runs no longer file); every arm and swap on a pinned harness tree; the RelBench swap (rel-event/user-attendance, a second family and a minimised metric) at a test MAE of 0.2298; the ALE swap (ahc011) 38th of 699 on the private cases; the Kaggle swap (spooky-author-identification under the modelling harness) at a best public log loss of 0.30161 over four ledger submissions, with no candidate delivered (the provided evaluation read the CLI's status under the wrong spelling, fixed since).
- Not measured: a cold start on the IOAI practice task (the one arm replayed the account's own earlier kernel, no candidate delivered; a cold start needs an account that never entered the competition); a maximised-metric RelBench swap and a tabular Kaggle swap (the tasks run were a minimised-metric regression and a text classification); the MLE arm ran with the web on in implementation, which the modelling harness's configuration has since turned off, and its in-loop score is the candidate's own hold-out number, checked by the verifier and the final private grade; no arm reran for the review's prose fixes to the harnesses (0.3.3, 0.3.2).
