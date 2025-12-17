# File: `utils/process_bad_commit_report.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 141 |
| Imports | collections, copy, get_previous_daily_ci, huggingface_hub, json, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Processes test failure reports to attribute failures to commit authors and generate accountability notifications.

**Mechanism:** Reads `new_failures_with_bad_commit.json` (from `utils/check_bad_commit.py`), augments with job links from CI artifacts, counts failures grouped by model and author, assigns to team members or mergers based on a hardcoded team list, generates `new_failures_with_bad_commit_grouped_by_authors.json`, uploads both files to Hub dataset, and formats Slack-ready message with `GH_` mentions. Output shows model failure counts per author.

**Significance:** Accountability and ownership tool for CI failures, enabling maintainers to quickly identify who introduced regressions and notify the responsible parties. Particularly useful for scheduled CI runs that catch issues merged to main. Part of the bad commit detection workflow used by `.github/workflows/check_failed_model_tests.yml`.
