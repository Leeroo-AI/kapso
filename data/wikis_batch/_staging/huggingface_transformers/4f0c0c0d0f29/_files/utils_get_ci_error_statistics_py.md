# File: `utils/get_ci_error_statistics.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 305 |
| Functions | `get_jobs`, `get_job_links`, `get_artifacts_links`, `download_artifact`, `get_errors_from_single_artifact`, `get_all_errors`, `reduce_by_error`, `get_model`, `... +3 more` |
| Imports | argparse, collections, json, math, os, requests, time, traceback, zipfile |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes GitHub Actions CI failures by aggregating error statistics from test artifacts across an entire workflow run.

**Mechanism:** Queries GitHub API to get all jobs and artifacts from a workflow run. Downloads artifact zip files containing `failures_line.txt`, `summary_short.txt`, and `job_name.txt`. Parses error lines and failed test names, cross-references with job links. Aggregates errors using Counter to find most common failures. Provides two reduction views: by error type (showing all affected tests) and by model (showing error counts per model). Generates GitHub-formatted markdown tables for reporting.

**Significance:** Critical CI debugging tool that identifies patterns in test failures across hundreds of parallel jobs, helping maintainers quickly pinpoint widespread issues versus isolated failures. Used for daily CI health monitoring and prioritizing fixes.
