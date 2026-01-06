# File: `utils/process_bad_commit_report.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 141 |
| Imports | collections, copy, get_previous_daily_ci, huggingface_hub, json, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Processes new test failure reports from the daily CI, enriches them with job links, groups failures by author, uploads organized reports to HuggingFace Hub, and generates formatted Slack notification messages with author mentions.

**Mechanism:** The script reads `new_failures_with_bad_commit.json` produced by check_bad_commit.py, augments each failure entry with corresponding GitHub Actions job URLs from job_links.json, attributes failures to team members by matching authors and merge committers, uses Counter to aggregate failure counts by model per author, creates both full and author-grouped versions of the failure report, uploads both JSON files to a HuggingFace dataset repository organized by date and run ID, and formats output with "GH_" prefixed usernames for Slack mentions.

**Significance:** This tool is critical for accountability and rapid failure response in the daily CI workflow. By automatically identifying who introduced failing tests and notifying them via Slack, it enables quick triage and fixes, preventing test failures from accumulating and maintaining high code quality standards across the large contributor base.
