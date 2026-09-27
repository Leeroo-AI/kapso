# Leeroopedia learning pipeline: findings, 2026-09-25

What a live `learn_knowledge(Source.Repo(...))` run and an audit of the Leeroopedia
corpus (27,803 pages, built with this pipeline in February 2026) turned up. Each item
carries evidence, impact and a proposed fix. Status is **open** unless marked otherwise.

## How these were found

- **Live run.** `learn_knowledge(Source.Repo("https://github.com/lucidrains/speculative-decoding"))`
  on the shipped config (`learner.ingestor`: `claude-opus-5`, `oauth`, timeout 1800 s;
  extract-only; scratch `wiki_dir`; the repo-builder phase stubbed to write placeholder
  URLs instead of creating repositories). A 10-file repository.
- **Corpus audit.** Every page under `data/wikis` parsed with `parse_wiki_directory`,
  plus checks on the live wiki's API.
- **Code reading.** `knowledge_base/learners/` (pipeline, `RepoIngestor`, merger,
  validator) and `services/wiki_service/sync`.

## Live run, phase by phase

| Phase | Duration | CLI-reported cost | Outcome |
|---|---|---|---|
| 0 repo understanding | 189 s | $1.26 | 10/10 files mapped |
| anchoring | 1,163 s | $2.27 | 4 Workflow pages |
| anchoring_context | 400 s | $2.56 | index enriched |
| excavation_synthesis | **1,801 s** | **$0.00 (lost)** | **killed at the 1800 s deadline** after writing 21 Principle + 21 Implementation pages; pipeline logged "failed, continuing" |
| enrichment | 1,302 s | $5.35 | 9 Heuristic + 3 Environment pages |
| audit | 389 s | $2.52 | repaired the killed phase: 8 broken links removed, 44 index entries added |
| repo builder | stubbed | — | 4 placeholder URLs |
| orphan triage | code | — | 0 candidates in every bucket |
| orphan review | 29 s | $0.24 | nothing to review |
| orphan create → publish | see addendum | | |

Roughly 90 minutes and $14 of estimated API-equivalent cost for a 10-file repository,
producing 60 pages (4 W, 22 P, 22 I, 9 H, 3 E). The February batch averaged 126 minutes
and 110 pages per repository on `claude-opus-4-6` (189 repositories, 3 in parallel).

## Pipeline issues

### L1. The learner runs at whatever effort the machine's user settings say — **fixed**

**Fixed 2026-09-25.** `learner.ingestor.effort` and `learner.merger.effort` in `config.yaml`
(`xhigh`, the value every other Claude session in the config uses) are threaded into the
repo ingestor, the research ingestors and the merger. A learner built without params takes
the packaged config's values (`learners/defaults.py`), so there is one source for them.

`RepoIngestor._initialize_agent` and `KnowledgeMerger._initialize_agent` never pass
`effort`, so the session inherits `effortLevel` from the user's `~/.claude/settings.json`
(`xhigh` on the dev box). That is why anchoring took 19 minutes on ten files, and why the
same code behaves differently per machine. The config has no `effort` key for the learner.

Fix: `learner.ingestor.effort` / `learner.merger.effort` in `config.yaml`, threaded to
the adapter's `effort` (the backend-search service already does this for its own session).

### L2. The per-phase deadline is too small, and a deadline kill is treated as success — **fixed**

**Fixed 2026-09-25, by decision: there is no wall clock on a learning phase.**
`learner.*.timeout` is `null` by default (a phase runs until it finishes; a number of seconds
imposes a deadline), and a failed or killed phase now raises instead of being logged and
skipped.

`excavation_synthesis` was killed at 1800 s, half-written. `_run_phase` returned False,
`ingest()` logged "Phase failed, continuing to next phase" and went on; the final result
reported success with 60 pages. At the observed pace no real repository fits a phase in
30 minutes. Related: the killed session reports `$0.0000` and near-zero tokens, so a
timed-out run also under-reports its cost.

Fix: size the deadline per phase from config; make a killed phase a hard failure (Rule 2),
or record per-phase status on `PipelineResult` so `success` is honest.

### L3. Failures are soft everywhere — **fixed**

**Fixed 2026-09-25.** Every phase raises on failure (repo ingestor, research ingestors,
Phase 0 after its retries, the merge session); a prompt with a missing variable raises
instead of going out unformatted; `KnowledgePipeline.run` propagates ingest and merge
exceptions; the no-index merge's index build is mandatory; `PipelineResult.success` is
`not errors`. Pinned by `tests/test_learners_fail_loud.py`.

**2026-09-27:** a run stages under `<wiki_dir>/_staging/<repo>/<branch>`, clones into it,
marks each finished phase in `_phases/`, and a run of the same source continues from the
first unfinished phase (a completed run's directory is replaced). The orphan verification
is now the gate after the orphan audit: one targeted audit pass with its findings, then the
run fails. A missing wiki-structure file raises instead of sending a prompt that says none
is defined. Pinned by `tests/test_repo_ingestor_pipeline.py`.

- `KnowledgePipeline.run` swallows ingest exceptions into `result.errors`;
  `PipelineResult.success` is True whenever any page came out.
- `KnowledgeMerger._create_all_pages` wraps the index build in `try/except` and warns.
- `_build_phase_prompt` returns the **unformatted** template when a format variable is
  missing (a prompt would go out with literal `{repo_path}` placeholders), with a warning.
- Every phase in `ingest()` continues after a failure.

Fix: fail loud per CLAUDE.md Rule 2; the only sanctioned "continue" is a phase whose
absence the validator can prove harmless.

### L4. The validator checks links, not pages — **fixed**

**Fixed 2026-09-27.** `validate_wiki_directory` now rejects a page that has no wikitext
sections, no `== Overview ==` section or no text under it, cites a temporary clone path
(`kapso_repo_…`), or links to a page without a namespace; it reads index entries from their
`[→](./dir/Name.md)` links, so the `## Workflow:` sections and step tables no longer produce
false warnings. The prompts now teach namespaced links (`[[Principle:Name]]`): the live run's
pages had 29 plain links in 60 pages, which the earlier tally missed. The parser accepts the
older layouts (the first section after the metadata block, subsections included, stands in
for a missing Overview; a Description-only Overview reads as the description), which makes
649 of the 655 corpus pages embeddable without editing them; the remaining 6 are Markdown.
Correction to the table below: 6 pages are pure Markdown; the other 96 counted are wikitext
pages that quote Markdown headings inside their content. Pinned by
`tests/test_repo_ingestor_pipeline.py`. The corpus repair (temp paths, plain links, the 6
Markdown pages) is a separate data pass, recorded in the addendum.

`validate_wiki_directory` enforces link targets, Principle→Implementation, Workflow
GitHub URLs and index files. It never checks section structure, markup, or content.
In the corpus this let through:

| Defect | Pages | Consequence |
|---|---|---|
| no `== Overview ==` and no `=== Description ===` (older prompts wrote `== Summary ==` etc.) | 655 | `parse_wiki_directory` yields no text → the page is never embedded |
| Markdown headings (`## `) instead of wikitext | 102 | renders wrong on the wiki; 6 of them have no headings the parser can read at all |
| temporary clone path (`/tmp/kapso_repo_…/file.py`) cited as source location | 94 | dead references (three repositories: lmms_eval, autogen, river) |
| plain `[[Name]]` links without a namespace | 3,037 | resolve to the main namespace → red links on the wiki (confirmed on the live site) |

The current prompts produce none of these (the live run's 60 pages are all wikitext, all
have Overview + Description, none cite a temp path, none carry plain links), so these are
older-prompt artifacts — but nothing prevents a regression.

Fix: deterministic checks in the validator for required sections per page type (from
`wiki_structure/*/sections_definition.md`), wikitext-only headings, no `/tmp/` paths,
and namespaced links; a one-off repair pass over the 655 + 102 + 94 + 3,037 corpus pages.

### L5. A GitHub failure discards the whole extraction — **fixed**

**Fixed 2026-09-27.** Publishing is a config switch, `learner.ingestor.publish_workflows`
(off by default; `learn_knowledge(github_org=..., is_private=...)` turns it on for a call), it
runs after the deterministic validation, and the phase raises if any repository is not
created, so a publishing failure never discards validated pages. Off, a Workflow page keeps
its placeholder URL and the validator reports that as a warning. The token no longer passes
through our code or the prompt: `gh` reads `GH_TOKEN` itself, and the preflight requires it
exactly when publishing is on. Pinned by `tests/test_repo_ingestor_pipeline.py`.

The repo-builder phase runs before validation, and validation requires a real GitHub URL
on every Workflow (`[https://github.com/PENDING …]` fails the regex). If a single repo
creation fails, the run raises (`fail_on_validation_errors`) after up to two hours of
extraction. A missing `GITHUB_PAT` only logs a warning and lets the run proceed to that
failure. The agent also holds `gh`'s stored login as a fallback, so a run without a PAT
can create repositories under whatever account `gh` is logged into.

Fix: make repository publishing an explicit config switch; validate before building;
treat a missing URL as a warning when publishing is off.

### L6. Orphan phases run with nothing to do — **fixed**

**Fixed 2026-09-27.** Orphan mining is one method (`RepoIngestor._run_orphan_mining`):
review runs only when triage left files awaiting a decision, create and audit only when
files must get a page (`orphan_candidate_counts`). Pinned by
`tests/test_repo_ingestor_pipeline.py`.

Triage found 0 candidates, yet review, create and audit each start an Opus session.

Fix: skip the agent phases when every bucket is empty.

### L7. The learner executes third-party code with unrestricted Bash — **fixed**

**Fixed 2026-09-27.** Every extraction and merge session bans Bash, WebFetch and WebSearch
through `--disallowedTools` (`LEARNER_BANNED_TOOLS`), the only flag that removes a tool under
`--dangerously-skip-permissions`; the publishing session alone has Bash, for `git` and `gh`.
Pinned by `tests/test_learners_fail_loud.py` and `tests/test_repo_ingestor_pipeline.py`.

The ingestor session gets `Read, Write, Edit, Bash` under `--dangerously-skip-permissions`
inside the clone. During the run it `import`ed the repository's package (failing on
missing `beartype`/`einops`). A repository's README or code can steer the agent
(prompt injection) and anything it runs has the dev box's credentials.

Fix: drop `Bash` (the phases need `Read`/`Write`/`Edit`), or run learning in a sandbox
with no credentials.

### L8. Page count is not tied to repository size

Ten source files became 60 pages (22 Principles, 22 Implementations). Whether that is
knowledge or restatement needs a quality review; the merger's same-type similarity search
is the only dedup, and the February batch skipped the merge entirely.

### L9. Learned pages do not reach the served search index

The merger writes into the Weaviate collection named by `wiki_dir/.index`. The public
search API serves a different, separately built collection that nothing refreshes; a
learned page appears on the wiki (via the sync) but is not searchable through the API.

Fix: a scheduled incremental re-embed of pages missing from the served collection.

### L10. Smaller items — **mostly fixed 2026-09-27**

- `KnowledgePipeline(weaviate_collection=...)` is a dead parameter (the collection comes
  from the index file); the CLI still advertises `--collection`. **Fixed:** the pipeline
  and the merger take an `index_builder` (`Kapso.index_kg`, which `learn_knowledge` passes;
  the backends and the collection come from the config it reads); the pipeline's own CLI is
  gone and `python -m kapso.knowledge_base.learners` runs the facade.
- `KnowledgeMerger._initialize_agent` spawns the MCP server with a bare `python` from
  `PATH`. **Fixed:** `sys.executable`.
- `GITHUB_PAT` is read via `os.environ` in our code (CLAUDE.md Rule 3). **Fixed with L5:**
  `gh` reads `GH_TOKEN` itself; our code reads no environment variable.
- `learning.status_dir` (`learning/status`) is relative to the current directory, so a
  run from any directory leaves a `learning/` tree there. **Left as designed** (the
  observability design documents the cwd-relative default).
- A killed run leaves its clone behind (`/tmp/kapso_repo_*`; two also sit in `$HOME`):
  `finally` never runs on SIGTERM. **Fixed:** the clone lives in the run's staging directory,
  is removed when the run completes, and is kept for the resume when it does not; there is
  no temporary directory to leak.
- The merge path is untested at scale: one agent call with a 3600 s deadline for 100+
  pages; the February batch never merged. **Still open** (the deadline is gone with L2; the
  scale test is not done).
- `test_learn_batch.py` relies on `load_dotenv()` and defaults `--github-org leeroopedia`
  with public repositories. **Still open.**
- February's single batch failure was environmental: `git clone` hit a stale VS Code
  credential-helper socket on the dev box. **Mitigated:** the clone runs with
  `GIT_TERMINAL_PROMPT=0` and no stdin, so a credential prompt fails the clone at once
  instead of hanging, and the branch is the one asked for or the repository's default,
  never a silent fallback.

## Wiki service

### W1. Sync wedged for six months — **fixed** (`c01be909`)

`WikiPoller.mw` / `SyncEngine.mw` stored the MediaWiki client before `login()` had
succeeded; one unreachable wiki (2026-03-19) left a client with no API URL and every poll
for six months failed on it instead of logging in again (547k errors). Fixed by storing
the client only after login, with a regression test; image rebuilt, container recreated.

### W2. Plain links are red links on the wiki

See L4: 3,037 pages carry `[[Name]]` links that MediaWiki resolves to the main
namespace; neither `import_wikis.sh` nor the sync transforms rewrite them.

Fix: rewrite plain links to their namespace on import/sync (the type is knowable from
the page index files), and repair the source pages.

### W3. Open self-registration

`$wgGroupPermissions['*']['createaccount'] = true`; anonymous users cannot edit, but
accounts can be created freely and one account (2026-05-27, one edit to its own user page)
looks like spam.

Fix: require email confirmation or close registration if the wiki is read-only for
outsiders.

## Addendum: run completion

The first attempt was cut short by the harness wrapper around it (a 90-minute limit of
the test setup, not of the pipeline) during orphan create; the remaining steps were run
from the staging directory exactly as `ingest()` runs them.

| Phase | Duration | CLI-reported cost | Outcome |
|---|---|---|---|
| orphan create | 82 s | $0.57 | nothing to create |
| orphan verify | code | — | PASS |
| orphan audit | 327 s | $2.50 | no changes |
| deterministic validation | code | — | **0 errors**, 43 warnings |
| publish | code | — | 60 pages copied to `wiki_dir` |

Whole run: 9 agent sessions, about 100 minutes of session time, about $17 of
CLI-estimated cost, 60 pages from 10 source files.

**Page quality of the output (all 60 pages):** every page has an Overview and a
Description (median 189 / 1,542 characters), all headings are wikitext, no page cites a
temporary clone path, no page carries a plain `[[Name]]` link, and every Principle links
to an Implementation. The corpus defects in L4 are therefore artifacts of earlier prompt
versions, and the current prompts do not reproduce them.

**Two things the run still got wrong:**

- The excavation phase timed out (L2) and the audit phase had to repair the damage;
  the final result reported success.
- The 43 validator warnings are mostly false positives: `validate_page_indexes` does
  not parse the `## Workflow: <name>` entries that the anchoring phase writes into
  `_WorkflowIndex.md`, so it reports every Workflow page as missing from its index, plus
  one stale index entry (`Train_Both_Models_In_Parallel`, a workflow that was planned
  and then not written). Fix: one index-entry grammar shared by the writers and the
  validator, and a check that every index entry has a page.

## Addendum: corpus repair (2026-09-27)

A one-off pass over `data/wikis` (backed up before editing), after the validator and parser
changes: 2,972 pages changed. 8,905 plain `[[Name]]` links gained their namespace (the
page's type directory was unambiguous); 957 were left as they are (six ambiguous, the rest
not page links at all: Python lists written as `[[1, 2, 3]]`, or pages that never existed).
The 97 temporary clone paths in 94 pages became repository-relative paths. The six
Markdown pages were converted to wikitext (headings, fences, bold, code). The wiki sync
pushes the edits to the site at its one-edit-per-second rate. The public search API's
collection still holds the pre-repair text of these pages until it is refreshed (L9).
