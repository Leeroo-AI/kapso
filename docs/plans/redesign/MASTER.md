# The redesign: master implementation plan

2026-10-24. The *what* is `docs/research/redesign-status.md`; this is the
*how*: the work packages, their dependencies, the tests each one carries
and the tests that wait for the end, the branch and release shape, and the
method every package follows. Each package gets a sub-plan in this
directory, written from `TEMPLATE.md` when the package is next, reviewed,
and only then built. `wp1-build.md` is the first and the model.

## 1. The shape of the work

**One integration branch, one release.** The redesign renames `evolve` to
`build`, retires modes and `serving.enabled`, and replaces `SolutionResult`.
Landed piecemeal on `main`, which releases often and pushes every commit,
that would break released users every week. So: an integration branch
`redesign`, cut from today's `kapso-plugin` branch; one branch per work
package merged into it by pull request with the gate green; `main` merged
into `redesign` weekly to absorb fixes; and `redesign` merged into `main`
once, as the release, when the full semantic suite passes on a fresh
machine. No compatibility shims at any point (CLAUDE.md rule 7); the
release notes carry the rename.

**Disjoint ownership so packages run in parallel.** The facade
(`src/kapso/kapso.py`, 1,933 lines) and the CLI (`src/kapso/cli.py`, 1,330
lines) would be touched by six packages. WP0 splits the CLI into one module
per verb and moves the handles out of the facade, so that after it each
package owns files no other package edits. The ownership column below is
the rule; a package that needs a file it does not own asks, in its
sub-plan, for an interface from the owner.

**Plan the next step in detail, the rest to its interfaces.** Sub-plans
for packages two steps ahead would be rewritten; what the master fixes
now is each package's scope, the interface it exposes, what it depends on
and what proves it done.

## 2. Work packages

| WP | Name | Scope in one line | Owns | Depends on | Size |
|---|---|---|---|---|---|
| 0 | Foundations | the `redesign` branch; the CLI split into `src/kapso/cli/<verb>.py` and the handles module; the fresh-machine provisioning script; the semantic suite's skeleton | `src/kapso/cli/` (after WP0, each verb module passes to the package that owns the verb; `cli/__init__.py` and `cli/common.py` stay with WP0, and adding a verb is one module plus one tuple entry; `cli/learn.py` passes to WP3), `scripts/redesign/`, `tests/semantic/` skeleton | none | S |
| 1 | `build` | `Brief`, `Budget`, `Solution`, `Software`, `Experience`; `Kapso.build()` replacing `evolve()`; `kapso build`; `Solution.open`; the rename in code, tests and skills | `src/kapso/handles.py`, the build section of the facade, `src/kapso/cli/build.py`, `skills/` | 0 | M |
| 2 | Harness package | reference grammar and store; `check`; content hash in the record; staging per module (pages, skills, servers, delegates, merged hooks); prepared resources; prerequisite refusal; `kapso harness new\|check\|list\|add\|update\|remove\|export`; `--harness` on the three verbs | `src/kapso/harness/`, `src/kapso/cli/harness.py`, the staging seams in sessions and prompt assembly | 1 | L |
| 3 | Experience | reference grammar and homes; `load_experience`, `add_experience`; `kapso experience add\|list\|share`; provenance intro; scope rule; `serving.enabled` removed; `kapso bank`, `learn init-bank`, `learn_knowledge`, `research`, `index_kg` retired into it | `src/kapso/experience/`, `src/kapso/cli/experience.py`, the serving layer | 1, 2 (scope) | M |
| 4 | Config without modes | packaged defaults plus one override file; `modes` and the `mode` argument gone; `design_axes` deleted with the planner reading the ideation page and the goal; `additional_context` and handler prose retired; `implementation_web` a hook; benchmark configs as override files; `doctor` prints the effective config | `src/kapso/core/config.py`, `config.yaml`, strategy parameter reading, `benchmarks/*/config.yaml` | 2 | M |
| 5 | Deploy | the `target` argument with the config default; the five bundled targets as skills with their scripts | `src/kapso/deployment/`, `src/kapso/cli/deploy.py` | 2 | M |
| 6 | Observability | `kapso trace` text and `--json`; `doctor --json`; the registry's liveness filter; the harness in the status header; hook decisions, delegate reports and verification evidence in the record from both CLIs' transcripts | `src/kapso/cli/trace.py`, `doctor.py`, `watch.py`, the record writers | 1, 2 | M |
| 7 | Plugin | the `plugin/` tree, manifests, launcher, marketplace files; the commands; the six skills with sidecars; session-start brief; delegation tools; skill lab re-run; the authoring interview | `plugin/`, `.claude-plugin/`, `.agents/plugins/` | 1, 2, 3, 6 | L |
| 8 | Harness ports | `predictive-modelling` from RelBench then MLE; `competitive-programming` from ALE then IOAI; banks; cross-checks (per `initial-harnesses-plan.md`; parked until the owner says go) | `src/kapso/harness/bundled/`, `benchmarks/` | 2, 3, 4 | XL |
| 9 | Semantic suite | fixtures (the examples), persona cards, the simulated-user driver, the judge, the report; S1 to S6; GCP provisioning | `tests/semantic/`, `scripts/redesign/` | 0; each S on its WP | L |
| 10 | Docs and examples | the new navigation and pages; examples as tasks on harnesses; the drift check and the two lints (per `docs-and-examples-redesign.md`) | `docs/`, `docs.json`, `examples/`, `scripts/` | each section on its WP | L, spread |
| 11 | Release | version, changelog and release notes, PyPI, the docs site, the marketplace tag; the full semantic suite as the gate | | all | S |

Sizes: S under a week, M one to two weeks, L two to three weeks, XL
longer and GPU-bound. Estimates, not commitments; each sub-plan refines
its own.

## 3. Dependencies and tracks

```
WP0 ─ WP1 ─┬─ WP2 ─┬─ WP3 ──────┐
           │       ├─ WP4       ├─ WP7 ─┐
           │       ├─ WP5       │       ├─ WP11
           ├─ WP6 ─┴────────────┘       │
           │                            │
           └─ WP9 (S1 at 1, author test at 2, S4/S5 at 6, S2 at 7, S6 at 3+8, S3 at 8)
           └─ WP10 (Start/Build at 1, Harnesses at 2, Experience at 3, Deploy/trace at 5+6, plugin page at 7)
WP8 ─ after 2, 3, 4; parked
```

The spine is WP0 then WP1, serial: everything records against a
`Solution`. After WP1 four tracks run in parallel: **A** WP2 then WP5 (and
WP8 when unparked); **B** WP3, whose grammar and homes start beside WP2
and whose scope rule waits for WP2's manifest; **C** WP6, then WP4 once
WP2's pages exist; **D** WP9 and WP10, which start on day one with their
skeletons and grow one section per landed package. WP7 starts when A, B
and C have delivered their first interfaces and finishes last before the
release. The critical path is WP0, WP1, WP2, WP7, S2, WP11; WP2 and WP7
are the long poles.

## 4. Tests: what each package carries, what waits for the end

**Syntactic, every package.** Unit tests for each new module; contract
tests for each schema and grammar (negative cases included: unknown key,
unresolved reference, secret literal, malformed hook); integration tests
with the provider boundary mocked, asserting the shapes a module produces;
goldens for staged trees, rendered traces and records. They join the
curated gate, which stays green at every merge, with the CI invocation
unchanged. No live-provider test inside the gate.

**Proof run, every package.** One live run at a tiny budget on the fresh
machine, exercising the package end to end: the churn fixture for WP1, the
trading example on both CLIs for WP2 (where the two unverified Codex
facts, MCP tool naming in hook matchers and `stop_hook_active`, get
settled), a served card for WP3, a deploy for WP5, a trace for WP6, the
plugin round for WP7. Its campaign directory is archived, one file per
run, and linked from the sub-plan.

**Semantic, early where it can be.** S1 on today's `main` before WP1
starts, as the baseline the redesign is measured against; S1 again after
WP1 and WP10's Start and Build pages; the guide's author test when WP2's
`check` exists; S4 and S5 when WP6 lands; S2 one scenario step per skill
as WP7 lands them; S6 when WP3 serves and a second build exists.

**Semantic, in full, at the end.** The release gate is S1 to S6 complete,
both hosts, on a fresh machine, from the wheel that will be released; S3's
non-regression runs with WP8 when it is unparked. Anything that fails there
reopens its package.

**Reviews.** Every sub-plan and every pull request gets a cross-model
review before merge (the house practice: Codex at high effort), with the
findings answered in the plan or the PR. Four checkpoints with the owner:
after WP1 (the handle shapes become final and the interface document is
rewritten), mid-WP2 (host facts verified), after WP7 (the plugin on both
hosts), and before the release.

## 5. The method, per package

1. Write the sub-plan from `TEMPLATE.md`: scope and non-scope, the
   interface it exposes and consumes, file ownership, the change list,
   the tests, the proof run, acceptance, risks, estimate. Review it. The
   owner says go.
2. Build on a worktree branch off `redesign`; commit per change with its
   tests and docs; the gate green locally before every push.
3. Run the proof run on the fresh machine; archive it.
4. Run the package's semantic hooks; record the numbers in the sub-plan.
5. Land the package's section of the docs and examples (WP10's slice).
6. Pull request into `redesign` with the review; merge when green.
7. Update the status table in this file.

## 6. Status

| WP | Sub-plan | State |
|---|---|---|
| 0 | `wp0-foundations.md` | built and reviewed; gate green; proof run on the fresh machine recorded in the sub-plan |
| 1 | `wp1-build.md` | built on `wp1-build`; gate green; docs and examples slice landed; naming review and pull-request review folded (fourteen renames, twenty fixes, see the sub-plan); three proof runs on the fresh machine: the first found the relative-output crash (fixed), the second showed ideation takes twelve to fourteen minutes so a fifteen-minute campaign delivers nothing (budgets and pages corrected), the third, at forty-five minutes on the final wheel, delivered a candidate at 0.8474 and found the record's feedback following the wrong node (fixed); merged into `redesign` by pull request #155 |
| 2 | `wp2-harness.md` | built on `wp2-harness` 2026-10-25: the harness package (manifest, check, reference grammar and store, handle and `load_harness`, staging per host, task statement and setup session, verification per attempt, the verbs on a finished build, `kapso harness`, the trading example, the docs); gate green; the author test passed by both model authors (the guide revised with their guesses); both pull-request reviews answered (four P1s and some forty P2/P3s fixed, see the sub-plan's Review); proof runs on two fresh machines delivered 1.93 and 1.99 with valid verdicts and one identity in every record, the first launches having found the setup session's UUID bug; a third campaign with implementation and verification on Codex recorded in the sub-plan; findings for WP3, WP4 and WP6 listed there; merged into `redesign` by pull request #156 |
| 3 | `wp3-experience.md` | built on `wp3-experience` 2026-10-25: the experience package (the reference grammar and homes, four kinds of source with their provenance, the store, the ingest), the facade's load, add, share, update and remove, serving over sources (one environment for every bank, the intro in its own prompt slot, scope as a filter on the harness coordinate, exposure credited by source), `kapso learn <campaign> [--harness]` with the crews staged in full, `kapso experience` and `--experience` on `kapso build` and `kapso-mcp`, the knowledge and research verbs gone, the docs; gate green; both pull-request reviews answered (four P1s and three highs, some thirty P2/P3s fixed, eleven decisions recorded); proof: a learn of WP2's trading campaign under its harness (fifteen cards, bank `79e93044` to `a9ddad3d`) and three builds, the trading example on a fresh machine with `--harness` and `--experience mine`, the churn example there under no harness, the hosted gate on the development box, every item of the proof list holding; the first launches found two engine bugs, fixed on the branch (the factory's missing keyword, a leftover `result.json` refused at launch); findings for WP4, WP6 and WP8 listed there; merged into `redesign` by pull request #157 |
| 6 | `wp6-observability.md` | built on `wp6-observability` 2026-10-25/26: every session's record with one run entry per host invocation from both hosts' transcripts (the hooks and their decisions, a blocking hook's denial included, the delegates and their reports, the tools and gates, the files written, the cost or the tokens, how it ended), the Codex paths outside the adapters recording theirs, the pull log by session; the record's lifecycle (the delivery inside the protected region, `done` after the record, failure owned from the first status write, a launch registered before its checks and told by its receipt); `kapso trace` text and `--json` with `Kapso.trace`, `kapso watch` completed (the header, the receipt, the no-path listing), `kapso doctor --json`, the engine's errors as their message on stderr, `Solution.session_failures`, the refused mining lead relaunched; the real orchestrator as a test fixture; the docs; gate green (1,617 passed); proof on a fresh machine (a trading build with the harness and the bank, a churn build with two injected failures, a launch dying at a check) and on the development box (the implementation on Codex), S4's two readings by Codex and Claude Code graded against the record with no wrong claim; three findings fixed on the branch, the rest listed for WP1, WP3 and WP4; the pull-request review answered (see the sub-plan's Review); merged into `redesign` by pull request #158 |
| 4 | `wp4-config.md` | built on `wp4-config` 2026-10-26: the config as two layers (the packaged file the schema at every depth, one override file, an unknown key refused by name), the hosts table and the three roles with `--coding-agent` as the host and the preflight's pairing, every build session's config through one precedence, the brief as the problem text and the launch record in the handle's words, the web as the engine's boundary per role with the harness hook as the domain rule, the budget's end (search and finishing rooms, the derived and bounded reserve, admission from the last round, the unjudged attempt, the attempt's commit), the judge's retry by reason, the engine's own sessions in the record, the harness directory protected by the engine's hook and refused by file name, the benchmark configs as override files with the runners' smokes, the design axes the planner's, the docs; ownership taken from WP1 (the budget's end, the launch record's words, the evaluation directory's rule stated), WP2 (the web switch as a hook, the engine hook, the hosted gate's approval), WP6 (the trace's `commit`, `unjudged` and `retry_reason`, the engine sessions and the run's deadline in the record, `doctor --json`'s config) and WP8 (the handlers' constructors); 349 files, +18,648 −5,738; gate green (1,705 passed); proof on a fresh machine (the doctor and the stale file, the hooks, the twelve- and eighteen-minute churn builds, the trading build to its goal, the hash check catching a shell write) and on the development box (every session on Codex, the judge on Claude, the hosted gate's approval), S4's two readings graded with no wrong claim, S1 22.2/26.9 min on Codex; six findings fixed on the branch, the rest listed there; merged into `redesign` by pull request #159 (merge 170a2962) |
| 5 | `wp5-deploy.md` | built on `wp5-deploy` 2026-10-26: `deploy(solution, target, env, options, coding_agent)` and `kapso deploy --target` with the lifecycle flags (`--status`, `--stop`, `--start`, `--logs`); the five bundled targets as skills with their scripts under `kapso/deployment/targets/`, staged into one deployment session with a harness's deployment module; the result file the engine checks (the target, the artifacts, the health script, the proof call run again) with one resume; the record under `<campaign>/.kapso/deployments/<n>/` with the deployment's own name and port, the states, the target's scripts copied beside it; `Software` with three run kinds (a worker process for a function) and the lifecycle through the scripts from any process; the `deployment` block as a spec over the hosts table; the trading example's `paper-trading` target; the two Deploy pages; gate green (1,745 passed); proof on a fresh machine (the churn build deployed to local, docker twice side by side, auto and the config's default; WP4's trading build restored and deployed to paper-trading; the refusals) and on the development box (the session on Codex; a Claude Code session following the engine skill end to end); six findings fixed on the branch, the rest listed there; merged into `redesign` by pull request #160 (merge 781673b1), its review's 60 findings fixed or answered in the sub-plan |
| 7 | `wp7-plugin.md` | built on `wp7-plugin` 2026-10-26/27: the plugin tree (three manifests, the launcher, the repository as its marketplace, one version pinned by test), six skills with Codex sidecars and ten commands, the engine seams (`kapso build --stop`, `--detach` on learn and ingest, `kapso harness export`, the `campaign` gate, `experience.session_sources`, `kapso-mcp` reading the project's config and composing its instructions, `--version`), the plugin kept out of campaign sessions on both hosts, the trigger suite of 111 cases run on both hosts, the plugin page and the README's step; the proof round on two fresh machines for Claude Code (23 of 24 steps, the learn's completion the one the learning system failed) and the development box for Codex (18 of 18), its records and reports cited in the sub-plan; nine findings fixed on the branch (the Codex adapter's isolation line that refused a config among them), the rest listed there for their packages; merged into `redesign` by pull request #162 (merge 5cca2905), its review's 60 findings fixed or answered in the sub-plan |
| 9 | `wp9-semantic-suite.md` | built on `wp9-semantic-suite` 2026-10-27 to 29: the drivers for S1, S2, S4, S5, S6 and the author test, the persona and its cards, the Gemini judge with its citation checks, the report with the verdict as the runner's exit status, the fresh-machine runner with its caps derived from the suite; the proof arms recorded in the sub-plan — S1 nine of twelve first-try against the released baseline's six (re-read under the packages' own verbs), the author test passing on both hosts after four guide rounds, S4 both readers six of six, S5's clean arm meeting every bar, S6 passing on the fixed wheel, the clean S2 runs on both hosts with every false row traced to a fix, a finding or a calibration (the second engineer's served build not established: three arms, each lost to the harness); the pull request's cross-model review, fifty-nine findings fixed or answered, three records re-derived and four runs repeated for them; the product findings under the sub-plan's "Findings for later packages" (Codex campaign sessions do not take up served experience; a build without a git identity dies after its receipt); merged into `redesign` by pull request #165 (merge 5647528b) |
| 8 | `wp8-harness-ports.md` | merged 2026-10-30 (PR #171 → `redesign` 5092a5bc): two bundled harnesses, four runners on `build()`, the IOAI book as a bank with its provenance table, the abstraction scan, the S3 record from the arms and swaps (every required step run; the IOAI delivery bar unmet and accepted by the owner), the author test green on both hosts, the two review halves (22 and 27 findings) answered |
| 10 | `wp10-docs-examples.md` | merged 2026-10-30 (PR #172 → `redesign` 6bafff13): the navigation by what the reader is doing, the front pages on the direction document, two claim lints, Learn in two pages, the bundled harness pages, the benchmark pages' provenance, the examples under the design's names with their driver test, the two review halves (32 and 23 findings) answered |
| 11 | `wp11-release.md` | at the owner's checkpoint 2026-10-30 (PR #173 → `redesign`): `main`'s twenty commits ported (nine conflicts resolved as ports), the version 0.8.0, the release notes with the rename mapping, eight gate reports from the release wheel on fresh machines (every one FAIL by the suite's bars, none on a wrong call of the package; the repo-memory update's withholding found by the gate, fixed and exercised live), the findings for the owner in the sub-plan; the merge to `main`, the tag and PyPI on the owner's go |

WP0 landed directly on `redesign` because it is the package that creates
the branch; every later package comes in by pull request.

## 7. Risks

| Risk | Mitigation |
|---|---|
| the rename breaks released users | one release from the integration branch, release notes with the mapping, no piecemeal landing |
| Codex hook facts unverified (MCP matcher names, `stop_hook_active`) | settled in WP2's proof run; the staging layer rewrites matchers per host if needed |
| the facade and CLI as merge hot spots | WP0's split and the ownership column; interfaces frozen in sub-plans before parallel work |
| benchmark configs and runners break when modes go | WP4 converts them with a runner smoke test per benchmark; WP8 is the full reproduction |
| model-token limits throttle proof and semantic runs | the token gates already in use (probe the CLI per model before launching; the 7-day rule) |
| the IOAI bank's provenance | WP8, after the backup and the licence pass; local source only until then |
| scope creep toward the live UI | out of the first release by decision; the master lists no package for it |
| estimates | each sub-plan refines its own; the status table shows slippage early |
