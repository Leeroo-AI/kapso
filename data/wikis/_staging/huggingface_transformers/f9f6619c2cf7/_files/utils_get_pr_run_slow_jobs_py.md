# File: `utils/get_pr_run_slow_jobs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 133 |
| Functions | `get_jobs_to_run`, `parse_message`, `get_jobs`, `check_name` |
| Imports | argparse, json, re, string |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines which models or test categories should run slow CI tests based on pull request changes or comment triggers.

**Mechanism:** Analyzes files changed in a PR using regex patterns to extract model names from paths (modeling files, test files, quantizers), or parses PR comments for explicit model lists using the "run-slow" command syntax. Validates model names against repository directories and returns a filtered list of jobs to execute, with a maximum limit of 16 suggestions for automatic detection.

**Significance:** Essential CI automation tool that optimizes test execution by running slow tests only for affected models rather than the entire test suite. Supports both automatic detection from file changes and manual triggering via PR comments, improving CI efficiency while maintaining thorough testing coverage.
