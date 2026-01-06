# File: `scripts/log_reports.py`

**Category:** CI reporting script

| Property | Value |
|----------|-------|
| Lines | 145 |
| Functions | `main` |
| Imports | argparse, datetime.date, json, os, pathlib, tabulate, slack_sdk.WebClient (conditional) |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Parses pytest JSON log files from CI test runs and posts formatted test results to a Slack channel, highlighting failures and providing links to GitHub Actions.

**Mechanism:**
- Scans current directory for all `.log` files containing pytest JSON output
- Parses each log file line-by-line:
  - Extracts test nodeid (test identifier)
  - Captures test duration and outcome (passed/failed)
  - Builds lists of failed and passed tests by test group
  - Detects empty log files (indicates test infrastructure issues)
- Deletes log files after processing

For failed tests:
- Groups failures by log file (test suite)
- Formats failure tables using tabulate with grid layout
- Truncates to MAX_LEN_MESSAGE (2900 chars) to respect Slack API limits
- Creates rich Slack message blocks with:
  - Header with test type (from TEST_TYPE env var)
  - Formatted failure tables with test locations
  - Warnings for empty log files
  - Action button linking to GitHub Actions run
  - Date/context footer

For successful runs:
- Posts simple success message with emoji

Environment variables:
- `TEST_TYPE`: Type of test run (e.g., "nightly", "integration")
- `SLACK_API_TOKEN`: Slack bot token for posting messages
- `GITHUB_RUN_ID`: GitHub Actions run ID for generating action links

Command-line arguments:
- `--slack_channel_name`: Target Slack channel (default: "peft-ci-daily")

**Significance:** Essential CI observability tool that provides automated test result notifications to development team. Enables quick identification of test failures without manually checking GitHub Actions. The formatted tables and direct links streamline debugging workflows. Particularly important for scheduled/nightly test runs.
