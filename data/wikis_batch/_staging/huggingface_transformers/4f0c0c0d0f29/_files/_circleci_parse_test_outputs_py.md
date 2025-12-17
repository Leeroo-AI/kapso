# File: `.circleci/parse_test_outputs.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `parse_pytest_output`, `parse_pytest_failure_output`, `parse_pytest_errors_output`, `main` |
| Imports | argparse, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses pytest output files to extract and summarize skipped tests, failed tests, and errors for CI reporting.

**Mechanism:** Uses regular expressions to match pytest output patterns: SKIPPED lines (test file path, reason), FAILED lines (test name, error type, reason), and ERROR lines (test name, error type, reason). Groups results by reason/error type, counts occurrences, and displays sorted summaries. Exits with status code 1 if failures or errors are found, enabling CI failure detection.

**Significance:** Essential CI utility for making test results human-readable and actionable. Called by CircleCI jobs (in create_circleci_config.py steps) to display test failure reasons, skipped test summaries, and error reports after pytest runs, helping developers quickly identify and diagnose test issues.
