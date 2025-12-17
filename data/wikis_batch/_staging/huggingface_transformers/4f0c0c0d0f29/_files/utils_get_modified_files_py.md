# File: `utils/get_modified_files.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 36 |
| Imports | re, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Identifies modified Python files in specific directories since the branch forked from main, for targeted CI testing.

**Mechanism:** Uses `git merge-base main HEAD` to find the fork point SHA, then `git diff --diff-filter=d --name-only` to list modified files (excluding deleted ones). Filters results to .py files under directories passed as command-line arguments (e.g., utils, src, tests, examples) using regex matching. Outputs space-separated list without trailing newline for Makefile consumption.

**Significance:** CI optimization utility that enables targeted testing and linting by identifying only the files changed in a PR, reducing execution time for quality checks that don't need to run on the entire codebase. Commonly used by Makefile targets for selective code quality checks.
