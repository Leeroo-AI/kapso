# File: `scripts/log_reports.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 144 |
| Functions | `main` |
| Imports | argparse, datetime, json, os, pathlib, tabulate |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** CI reporting script that parses pytest JSON logs, formats test results into tables, and posts formatted failure reports to a designated Slack channel for monitoring nightly and scheduled test runs.

**Mechanism:** Scans for .log files containing JSON-formatted pytest output (using --report-log). Parses each line to extract test node IDs, durations, and outcomes. Separates failed and passed tests, counts failures per log file, and uses tabulate to format results into ASCII tables. Constructs Slack message blocks with headers, test location breakdowns, and action buttons linking to GitHub Actions. Truncates messages exceeding 2900 characters to respect Slack's 3001-char limit. Posts via slack_sdk.WebClient when TEST_TYPE environment variable is set.

**Significance:** Essential CI observability tool that enables the PEFT team to monitor test health across nightly builds and different test configurations. By pushing formatted reports to Slack with direct links to failing tests, it reduces response time to regressions and makes test failures visible to the entire team without requiring manual GitHub Actions monitoring.
