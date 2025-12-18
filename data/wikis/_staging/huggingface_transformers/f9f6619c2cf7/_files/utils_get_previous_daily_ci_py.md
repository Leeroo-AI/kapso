# File: `utils/get_previous_daily_ci.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 159 |
| Functions | `get_daily_ci_runs`, `get_last_daily_ci_run`, `get_last_daily_ci_workflow_run_id`, `get_last_daily_ci_run_commit`, `get_last_daily_ci_artifacts`, `get_last_daily_ci_reports` |
| Imports | get_ci_error_statistics, os, requests, zipfile |

## Understanding

**Status:** ✅ Explored

**Purpose:** Retrieves artifacts and metadata from previous daily CI workflow runs for comparison and trend analysis.

**Mechanism:** Queries GitHub Actions API to find completed scheduled workflow runs on the main branch, fetches their artifacts (test reports, failure logs), downloads and extracts zip files, and returns structured data including workflow run IDs, commit SHAs, and artifact contents. Supports filtering by workflow ID or commit SHA for specific run retrieval.

**Significance:** Critical for CI reporting infrastructure that enables comparison between current and previous test runs to detect new failures, regressions, or improvements. Provides the foundation for trend analysis and helps identify whether failures are new or pre-existing in the scheduled daily CI runs.
