# File: `utils/compare_test_runs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 91 |
| Functions | `normalize_test_line`, `parse_summary_file`, `compare_job_sets` |
| Imports | re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compares test results between two CI runs to identify newly appeared or disappeared test failures.

**Mechanism:** Normalizes test status lines (SKIPPED, XFAIL, ERROR, FAILED) by stripping location details and extra metadata using regex patterns. Parses summary files into sets of normalized test lines, then computes set differences to find tests added or removed between runs. Generates diff reports showing which tests appeared (+) or disappeared (-) in the current run.

**Significance:** CI diff analysis tool that helps identify test regressions or fixes by comparing successive test runs, making it easier to track when specific test failures were introduced or resolved.
