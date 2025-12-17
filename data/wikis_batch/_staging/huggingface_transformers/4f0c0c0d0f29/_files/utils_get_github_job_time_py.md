# File: `utils/get_github_job_time.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `extract_time_from_single_job`, `get_job_time` |
| Imports | argparse, dateutil, math, requests, traceback |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Analyzes GitHub Actions workflow job durations to identify performance bottlenecks in CI pipelines.

**Mechanism:** Queries GitHub API for all jobs in a workflow run (handling pagination for 100+ jobs). Extracts `started_at` and `completed_at` timestamps, calculates duration in minutes using dateutil.parser for proper timezone handling. Returns dictionary mapping job names to time info, sorted by duration descending to highlight longest-running jobs.

**Significance:** CI performance monitoring tool that helps identify slow test jobs requiring optimization. Useful for capacity planning, detecting performance regressions in test suites, and prioritizing parallelization efforts to reduce overall CI time.
