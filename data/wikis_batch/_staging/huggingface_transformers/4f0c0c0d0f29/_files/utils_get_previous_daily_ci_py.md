# File: `utils/get_previous_daily_ci.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 159 |
| Functions | `get_daily_ci_runs`, `get_last_daily_ci_run`, `get_last_daily_ci_workflow_run_id`, `get_last_daily_ci_run_commit`, `get_last_daily_ci_artifacts`, `get_last_daily_ci_reports` |
| Imports | get_ci_error_statistics, os, requests, zipfile |

## Understanding

**Status:** ✅ Explored

**Purpose:** Retrieves artifacts and reports from previous daily CI workflow runs using GitHub Actions API.

**Mechanism:** Queries GitHub API for scheduled workflow runs on the main branch, filters by workflow ID and commit SHA, downloads artifacts as zip files, extracts their contents, and returns structured data. Key functions handle workflow run lookup (`get_daily_ci_runs`), artifact retrieval (`get_last_daily_ci_artifacts`), and report extraction (`get_last_daily_ci_reports`).

**Significance:** Essential for CI comparison and regression tracking, enabling workflows to compare current test results against previous runs to identify new failures, track test stability trends, and generate meaningful diff reports for Slack notifications.
