# File: `utils/pr_slow_ci_models.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 175 |
| Functions | `get_new_python_files_between_commits`, `get_new_python_files`, `get_new_model`, `parse_message`, `get_models`, `check_model_names` |
| Imports | argparse, git, json, os, pathlib, re, string |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines which models require slow CI testing for pull requests based on changed files and PR comments.

**Mechanism:** Analyzes git diff to detect new modeling files matching `src/transformers/models/(.*)/modeling_.*\.py` pattern, parses PR comment body for `run-slow`/`run_slow`/`run slow` directives followed by comma-separated model names, validates model names against allowed characters, and maps them to test directories (`tests/models/{model}` or `tests/quantization/{model}`). Returns JSON list of test paths for GitHub Actions matrix.

**Significance:** Critical CI optimization enabling targeted slow test execution on PRs. Allows both automatic detection of new models and manual override via comments. Works in tandem with `get_pr_run_slow_jobs.py` to minimize unnecessary slow test runs while ensuring adequate coverage for changed code.
