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

### L1. The learner runs at whatever effort the machine's user settings say

`RepoIngestor._initialize_agent` and `KnowledgeMerger._initialize_agent` never pass
`effort`, so the session inherits `effortLevel` from the user's `~/.claude/settings.json`
(`xhigh` on the dev box). That is why anchoring took 19 minutes on ten files, and why the
same code behaves differently per machine. The config has no `effort` key for the learner.

Fix: `learner.ingestor.effort` / `learner.merger.effort` in `config.yaml`, threaded to
the adapter's `effort` (the backend-search service already does this for its own session).

### L2. The per-phase deadline is too small, and a deadline kill is treated as success

`excavation_synthesis` was killed at 1800 s, half-written. `_run_phase` returned False,
`ingest()` logged "Phase failed, continuing to next phase" and went on; the final result
reported success with 60 pages. At the observed pace no real repository fits a phase in
30 minutes. Related: the killed session reports `$0.0000` and near-zero tokens, so a
timed-out run also under-reports its cost.

Fix: size the deadline per phase from config; make a killed phase a hard failure (Rule 2),
or record per-phase status on `PipelineResult` so `success` is honest.

### L3. Failures are soft everywhere

- `KnowledgePipeline.run` swallows ingest exceptions into `result.errors`;
  `PipelineResult.success` is True whenever any page came out.
- `KnowledgeMerger._create_all_pages` wraps the index build in `try/except` and warns.
- `_build_phase_prompt` returns the **unformatted** template when a format variable is
  missing (a prompt would go out with literal `{repo_path}` placeholders), with a warning.
- Every phase in `ingest()` continues after a failure.

Fix: fail loud per CLAUDE.md Rule 2; the only sanctioned "continue" is a phase whose
absence the validator can prove harmless.

### L4. The validator checks links, not pages

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

### L5. A GitHub failure discards the whole extraction

The repo-builder phase runs before validation, and validation requires a real GitHub URL
on every Workflow (`[https://github.com/PENDING …]` fails the regex). If a single repo
creation fails, the run raises (`fail_on_validation_errors`) after up to two hours of
extraction. A missing `GITHUB_PAT` only logs a warning and lets the run proceed to that
failure. The agent also holds `gh`'s stored login as a fallback, so a run without a PAT
can create repositories under whatever account `gh` is logged into.

Fix: make repository publishing an explicit config switch; validate before building;
treat a missing URL as a warning when publishing is off.

### L6. Orphan phases run with nothing to do

Triage found 0 candidates, yet review, create and audit each start an Opus session.

Fix: skip the agent phases when every bucket is empty.

### L7. The learner executes third-party code with unrestricted Bash

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

### L10. Smaller items

- `KnowledgePipeline(weaviate_collection=...)` is a dead parameter (the collection comes
  from the index file); the CLI still advertises `--collection`.
- `KnowledgeMerger._initialize_agent` spawns the MCP server with a bare `python` from
  `PATH`.
- `GITHUB_PAT` is read via `os.environ` in our code (CLAUDE.md Rule 3).
- `learning.status_dir` (`learning/status`) is relative to the current directory, so a
  run from any directory leaves a `learning/` tree there.
- A killed run leaves its clone behind (`/tmp/kapso_repo_*`; two also sit in `$HOME`):
  `finally` never runs on SIGTERM.
- The merge path is untested at scale: one agent call with a 3600 s deadline for 100+
  pages; the February batch never merged.
- `test_learn_batch.py` relies on `load_dotenv()` and defaults `--github-org leeroopedia`
  with public repositories.
- February's single batch failure was environmental: `git clone` hit a stale VS Code
  credential-helper socket on the dev box.

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

Filled in after the remaining phases finished from the staging directory (see the
commit that updates this file).
