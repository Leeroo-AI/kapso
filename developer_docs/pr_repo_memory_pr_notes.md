# PR Notes: RepoMemory V2 (evidence-backed) + Observability + Ideation ReAct

This document captures **what changed** in this PR and **why**.
It is intended to be used as PR description material and as reviewer context.

## High-level goals

- Make RepoMemory **portable** and **git-consistent** (no absolute paths, no ignored/transient files in RepoMap).
- Make RepoMemory **auditable** (commit `changes.log`; persist which sections were consulted).
- Ensure RepoMemory reflects the repo's **latest committed state** (“latest commit semantics”).
- Externalize prompts so we can **tune them without code changes**.
- Add **engine-mediated** “ReAct-style” section retrieval for ideation, and a CLI tool for coding-agent section access (Claude Code).

## Tests run (pre-PR)

- `pytest -q --collect-only`
  - Ensures the entire repo’s test suite can be imported/collected without syntax/import errors.
  - This is important because we touched core orchestration code and added new prompt/template files.

- RepoMemory unit/integration (fast; no infra):
  - `tests/test_repo_map_git_consistency.py`
    - Verifies `build_repo_map()` is portable and git-consistent (e.g., filters out `.praxium/*`, `sessions/*`, and `changes.log`).
  - `tests/test_prompt_externalization.py`
    - Verifies all required prompt template files exist and can be loaded via `prompt_loader`.
  - `tests/test_repo_memory_book.py`
    - Tests RepoMemory V2 “book” APIs: Summary+TOC formatting, section retrieval, and evidence validation on book-shaped docs.
  - `tests/test_repo_memory_book_integration.py`
    - Integration tests for reading/migrating RepoMemory documents from worktrees and git branches.
  - `tests/test_repo_memory_cli_standalone.py`
    - Tests the standalone CLI (`tools/repo_memory_cli.py`) can list sections and render a section from a synthetic RepoMemory file.

- LLM-gated RepoMemory tests (requires API keys; costs money):
  - `tests/test_repo_memory.py::test_bootstrap_baseline_model_with_real_llm`
    - Runs baseline RepoMemory inference on a seeded repo and asserts the generated memory is evidence-backed.
  - `tests/test_repo_memory.py::test_update_after_experiment_with_real_llm`
    - Makes real code changes, runs RepoMemory update, and asserts experiment deltas + evidence validation still pass.
  - `tests/test_repo_memory_react_loop.py::test_ideation_react_loop_smoke`
    - Smoke test for the engine-mediated ideation ReAct loop JSON protocol (model can request sections then return a final solution).
  - `tests/test_repo_memory_latest_commit_semantics.py`
    - Ensures RepoMemory updates run after final code commits (latest-commit semantics) and commit hashes line up.

- RepoMemory E2E (LLM + coding agent; used to audit artifacts)
  - `python tests/test_repo_memory_e2e.py`
    - Runs a small seeded repo through the build loop and prints RepoMemory Book semantics + section usage for human inspection.
    - Validates “ideation sections consulted” and `changes.log` auditability in a realistic run.

- KG infra + cognitive (requires `./start_infra.sh` + indexed wiki)
  - `python tests/test_cognitive_real_kg.py`
    - Verifies KG connectivity and that workflow retrieval uses real KG data (no fallback).
  - `python tests/test_expert_full_e2e.py tier1_exact_workflow`
    - Full cognitive E2E: KG Tier 1 workflow retrieval + coding agent run + LLM judge evaluation.
    - Includes RepoMemory audit checks (RepoMap invariants + changes.log vs persisted consulted sections).

## Monitoring / test-only changes in this PR

- Several changes are **test-only** or **monitoring/auditability-only** (they do not affect runtime behavior for end users):
  - `tests/test_expert_full_e2e.py`: adds RepoMemory audit assertions (post-run invariants).
  - `tests/test_repo_memory_e2e.py`: improves printed diagnostics so humans can judge RepoMemory quality/usage.
  - `tests/test_claude_code_agent_bedrock.py`: makes the Bedrock smoke test opt-in and skip-by-default.
  - `tests/deployment_examples/*`: prevents accidental pytest execution of deployment scripts; keeps deployment smoke tests opt-in.
  - New tests under `tests/test_repo_memory_*` and `tests/test_repo_map_git_consistency.py` validate RepoMemory invariants and tooling.

## Files changed vs `main`

- `developer_docs/coding_agents.md`
  - Document RepoMemory Summary+TOC injection and Claude Code section retrieval via CLI.

- `src/execution/experiment_workspace/experiment_session.py`
  - Add `schedule_repo_memory_update()` and run RepoMemory update in `close_session()` **after** final commits and **before** push/cleanup.
  - Commit `.praxium/repo_memory.json` as the final metadata commit (latest-commit semantics).

- `src/execution/experiment_workspace/experiment_workspace.py`
  - Ensure `.gitignore` includes `!changes.log` after `*.log` so `changes.log` is committed for auditability.
  - Pass the LLM handle into `ExperimentSession` so RepoMemory updates can run at close.

- `src/execution/search_strategies/base.py`
  - Externalize coding prompts into templates.
  - Inject RepoMemory **Summary+TOC** (bounded) instead of a large “brief” blob.
  - Parse `changes.log` to persist `repo_memory_sections_consulted`.
  - Schedule RepoMemory updates instead of running them inline.
  - Provide Claude Code instructions to use the RepoMemory CLI for section retrieval.

- `src/execution/search_strategies/linear_search.py`
  - Switch ideation to the engine-mediated RepoMemory ReAct loop and record consulted sections.

- `src/execution/search_strategies/llm_tree_search.py`
  - Switch solution generation to the engine-mediated RepoMemory ReAct loop.
  - Record ideation-consulted section IDs on each node.

- `src/tinkerer.py`
  - When `output_path` is provided, use it as the experiment workspace directory so the returned `solution.code_path` points to a real git repo with `.praxium/repo_memory.json`.

- `src/memory/config.py`
  - Whitespace-only formatting change.

- `src/repo_memory/builders.py`
  - RepoMap fixes:
    - `repo_root` stored as `"."` (portable).
    - File enumeration uses git where possible (respects `.gitignore`).
    - Filters out `changes.log`, `.praxium/*`, `sessions/*`.
  - Inference robustness:
    - Reuse the same file payload during evidence-retry loops (so the model can fix its own citations).
    - Improve evidence matching to tolerate whitespace around punctuation (handles multi-line-to-single-line quote flattening without allowing invented tokens).
  - Planning robustness:
    - Always include key files + entrypoints in the “files to read” selection.

- `src/repo_memory/manager.py`
  - RepoMemory updates now fall back to **full rebuild with retry feedback** on evidence failure.
  - Comments updated to consistently refer to “RepoMemory V2” format.

- `tests/deployment_examples/input_repos/chatbot_adapted_langgraph/test_deployment.py`
  - Fix indentation so pytest can import/collect the file.
  - Still skip-by-default unless `PRAXIUM_RUN_DEPLOYMENT_TESTS=1`.
  - **Why in this PR**: on `main`, this file was an ad-hoc script under `tests/` that performed HTTP calls at import time.
    That is unsafe for pytest collection and can hang/fail CI. We converted it into an opt-in smoke test.

- `tests/deployment_examples/test_unified_deployment.py`
  - Rename `test_repo()` -> `run_repo()` to avoid accidental pytest collection.
  - Keep it as a script-style deployment runner.
  - **Why in this PR**: on `main`, the helper was named `test_repo()`, so pytest could treat it as a real test even though
    it depends on external deployment/runtime setup. Renaming makes it opt-in (script-run), not an accidental test gate.

- `tests/test_claude_code_agent_bedrock.py`
  - Make the Bedrock smoke test skip-by-default unless explicitly enabled (`PRAXIUM_RUN_BEDROCK_TESTS=1`) and credentials + `claude` CLI exist.

- `tests/test_expert_flow.py`
  - Whitespace-only formatting change.

- `tests/test_expert_full_e2e.py`
  - Add post-run RepoMemory audit checks:
    - RepoMap invariants (portable + excludes meta paths)
    - `changes.log` auditability and consistency with persisted `repo_memory_sections_consulted`

- `tests/test_repo_memory.py`
  - Adjust tests for the updated RepoMemory behavior (prompt externalization + evidence robustness).

- `tests/test_repo_memory_e2e.py`
  - Improve printed diagnostics:
    - show semantic claim statements
    - show ideation sections consulted
    - show coding-agent sections consulted

## New files added

- `developer_docs/coding_agents_tooling_research.md`
  - Research notes on coding agent tooling capabilities and next steps.

- `developer_docs/prompt_tuning.md`
  - How to tune externalized templates (including override via env var).

- `developer_docs/pr_repo_memory_pr_notes.md`
  - This PR-notes file.

- `paper/Praxium_paper_v1.md`, `paper/Praxium_paper_v1.tex`
  - Draft paper artifacts (documentation/research; no runtime impact).

- `src/core/prompt_loader.py`
  - Prompt template loader + simple renderer, with optional override directory.

- `src/execution/ideation/repo_memory_react.py` (+ `src/execution/ideation/__init__.py`)
  - Engine-mediated ReAct loop for ideation: model can request RepoMemory sections by ID before outputting a final solution.

- `src/execution/prompts/`
  - `coding_agent_implement.md`, `coding_agent_debug.md`, `ideation_solution_react.md`
  - Externalized templates for coding-agent prompting and ideation ReAct protocol.

- `src/repo_memory/cli.py`
  - RepoMemory CLI that imports `RepoMemoryManager` but keeps stdout clean (redirects import noise).

- `src/repo_memory/observation.py`
  - Observability helpers (extract consulted sections from `changes.log`, book stats).

- `src/repo_memory/prompts/`
  - `plan_files_to_read.md`, `infer_repo_model_initial.md`, `infer_repo_model_retry.md`, `infer_repo_model_update.md`
  - Externalized prompts for RepoMemory inference and update.

- `tests/test_prompt_externalization.py`
  - Confirms prompt template files exist and are loadable.

- `tests/test_repo_map_git_consistency.py`
  - Guards RepoMap portability and filtering of meta paths.

- `tests/test_repo_memory_book.py`, `tests/test_repo_memory_book_integration.py`
  - Tests around RepoMemory V2 “book” rendering, section access, and migration behavior.

- `tests/test_repo_memory_cli_standalone.py`
  - Tests the standalone CLI in `tools/`.

- `tests/test_repo_memory_latest_commit_semantics.py`
  - Ensures RepoMemory update commit timing matches “latest commit semantics”.

- `tests/test_repo_memory_react_loop.py`
  - Tests JSON protocol parsing and includes a smoke run of ideation ReAct.

- `tools/repo_memory_cli.py`
  - Standalone CLI for agents (no `import src`) that reads `.praxium/repo_memory.json` directly.

