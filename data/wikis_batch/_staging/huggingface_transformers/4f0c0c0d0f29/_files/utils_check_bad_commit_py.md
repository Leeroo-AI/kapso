# File: `utils/check_bad_commit.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 280 |
| Functions | `create_script`, `is_bad_commit`, `find_bad_commit`, `get_commit_info` |
| Imports | argparse, git, json, os, re, requests, subprocess |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Uses git bisect to identify the exact commit that introduced a test failure by binary searching through commit history.

**Mechanism:** Creates a temporary Python script that runs a specified test with pytest and flake-finder, executes git bisect between two commits to find the first "bad" commit where the test fails, and retrieves commit metadata (PR number, author, merged_by) via GitHub API. Includes flakiness detection by running tests multiple times and checking if failures are reproducible.

**Significance:** CI/CD debugging tool that automates root cause analysis when tests start failing, helping maintainers quickly identify problematic changes and assign responsibility for fixing issues.
