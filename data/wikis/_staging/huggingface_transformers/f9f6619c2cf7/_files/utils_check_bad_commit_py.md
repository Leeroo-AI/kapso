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

**Purpose:** Identifies the first commit that introduced a test failure using git bisect to pinpoint problematic commits in CI test failures.

**Mechanism:** The script employs git bisect to perform binary search through commit history between a known good commit and a bad commit. For each commit tested, it creates a temporary Python script that installs the package and runs the failing test with flake-finder (4 runs to detect flaky tests). The bisect process automatically navigates through commits until it identifies the first commit where the test fails. The script includes flaky test detection by verifying failures are reproducible and fetches commit metadata (PR number, author, merger) from GitHub's API for reporting.

**Significance:** This tool is crucial for debugging CI failures by automating the tedious process of manually searching through commits to find the source of a regression. It helps maintainers quickly identify which PR introduced a bug, enabling faster fixes and better accountability. The flaky test detection prevents false positives from being reported, making the debugging process more reliable.
