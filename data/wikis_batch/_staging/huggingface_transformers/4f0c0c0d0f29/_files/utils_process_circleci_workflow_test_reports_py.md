# File: `utils/process_circleci_workflow_test_reports.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 146 |
| Imports | argparse, collections, json, os, re, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Aggregates and analyzes test results from CircleCI workflow jobs to generate comprehensive test summaries and failure reports.

**Mechanism:** Fetches job artifacts from CircleCI API using a workflow ID, processes test summary files and failure logs, normalizes test names to group related failures, and generates three JSON outputs: per-job summaries, a workflow-wide test summary mapping tests to job statuses, and a detailed failure analysis organized by test name and model name with error frequencies.

**Significance:** Critical CI tooling that provides structured test result analysis across parallel CircleCI jobs, enabling developers to quickly identify failing tests, understand error patterns, and track test stability across different job configurations. Supports debugging by aggregating failures from distributed test runs.
