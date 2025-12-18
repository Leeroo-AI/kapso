# File: `utils/compare_test_runs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 91 |
| Functions | `normalize_test_line`, `parse_summary_file`, `compare_job_sets` |
| Imports | re |

## Understanding

**Status:** ✅ Explored

**Purpose:** Compares test results between two CI runs to identify newly appearing or disappearing test failures, skips, and errors.

**Mechanism:** Parses pytest summary files from two test runs, normalizing test lines (FAILED, SKIPPED, ERROR, etc.) to consistent formats. Computes set differences to identify tests that appeared in the current run but not the previous (additions) or vice versa (removals), generating a formatted diff report showing per-job changes.

**Significance:** Helps identify test regressions and improvements between commits or branches. Crucial for tracking test stability in CI/CD, allowing developers to quickly pinpoint which tests broke or were fixed in a particular change without manually comparing full test logs.
