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

**Purpose:** Intelligently determines which tests to run in CI by analyzing git diffs and code dependencies to minimize test execution time.

**Mechanism:** Uses git to identify modified Python files since branching point (filtering docstring-only changes), recursively builds dependency trees by parsing imports in all modules and tests, creates a reverse dependency map showing which tests depend on each module, and selects only impacted tests. Filters to core models when too many are affected, handles tiny model updates, parses commit messages for CI control flags, and outputs categorized test lists for different CI job types.

**Significance:** Core CI optimization system that dramatically reduces test execution time by running only relevant tests while maintaining safety through comprehensive dependency analysis. Critical for PR efficiency in a massive codebase with 100+ models, enabling fast feedback loops without sacrificing test coverage. Supports both PR and post-merge workflows.
