# File: `utils/tests_fetcher.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1187 |
| Functions | `checkout_commit`, `clean_code`, `keep_doc_examples_only`, `get_all_tests`, `diff_is_docstring_only`, `diff_contains_doc_examples`, `get_impacted_files_from_tiny_model_summary`, `get_diff`, `... +17 more` |
| Imports | argparse, collections, contextlib, git, glob, important_files, json, os, pathlib, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Intelligently determines which tests need to run based on code changes in a PR or commit, dramatically reducing CI time by only running affected tests.

**Mechanism:** Operates in two stages: (1) Identifies modified Python files by comparing current commit with branching point or last commit, filtering out docstring-only changes, and (2) Builds a reverse dependency map by parsing import statements in all modules and tests to determine which tests depend on which modules. Recursively traverses dependencies to find all impacted tests. Can optionally filter to core models when too many models are affected. Also handles special cases like example tests and tiny model summary changes.

**Significance:** Core CI optimization infrastructure that makes the massive test suite practical by reducing test runs from all tests (thousands) to only relevant ones (often dozens). Saves significant compute resources and developer time by providing fast feedback on PRs. The dependency analysis is sophisticated enough to catch indirect impacts through import chains while being conservative enough to avoid missing necessary tests.
