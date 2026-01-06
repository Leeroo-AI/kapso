# File: `utils/get_modified_files.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 36 |
| Imports | re, subprocess, sys |

## Understanding

**Status:** ✅ Explored

**Purpose:** Identifies modified Python files in specific directories since branching from main. Outputs space-separated file list for use in Makefile commands and CI workflows.

**Mechanism:** Uses git merge-base to find fork point SHA, runs git diff with --diff-filter=d (exclude deleted) and --name-only to get changed files. Filters results using regex pattern built from command-line directory arguments to match only .py files in specified subdirectories. Prints result without trailing newline for shell command substitution.

**Significance:** Enables targeted CI/CD operations by identifying changed files. Used by Makefile and GitHub Actions to run quality checks, tests, or linting only on modified code rather than entire codebase. Critical optimization for large repository - running all checks on every commit would be prohibitively slow.
