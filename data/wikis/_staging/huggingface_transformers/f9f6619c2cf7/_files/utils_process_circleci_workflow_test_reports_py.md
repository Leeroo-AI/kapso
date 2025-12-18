# File: `utils/process_circleci_workflow_test_reports.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 146 |
| Imports | argparse, collections, json, os, re, requests |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Fetches test artifacts from CircleCI workflow jobs, parses test results and failure details, and generates comprehensive JSON summaries organized by test, job, and model for analysis and reporting.

**Mechanism:** The script uses the CircleCI API with authentication tokens to retrieve job information and artifacts for a given workflow ID, downloads summary_short.txt and failures_line.txt artifacts from test/example/pipeline jobs across parallel nodes, parses pytest output to extract passed/failed test names and error messages, normalizes test names by removing parameterization suffixes and numeric variations, aggregates failures by test name and model name using Counter objects, and outputs multiple JSON files including per-job summaries, workflow-wide test status mappings, and detailed failure analyses with error counts and affected test variants.

**Significance:** This tool provides essential post-mortem analysis for CircleCI test runs, enabling teams to identify patterns in test failures across jobs and models. It's particularly valuable for large-scale parallel test execution where understanding failure distribution and commonalities across nodes helps prioritize fixes and identify systemic issues versus flaky tests.
