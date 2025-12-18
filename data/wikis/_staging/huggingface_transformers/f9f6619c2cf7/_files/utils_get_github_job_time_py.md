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

**Purpose:** Extracts execution time metrics from GitHub Actions workflow jobs. Calculates duration in minutes for each job to identify performance bottlenecks and long-running tests.

**Mechanism:** Queries GitHub API with workflow run ID to fetch all jobs (handles pagination for 100+ jobs per page). Parses started_at and completed_at timestamps using dateutil.parser, calculates duration in minutes, returns dictionary mapping job names to timing information sorted by duration descending.

**Significance:** CI performance monitoring tool crucial for optimizing the massive test suite. Helps identify slow tests that impact developer productivity and CI costs. With hundreds of test jobs per workflow, knowing which jobs take longest guides parallelization strategies, resource allocation decisions, and test optimization efforts.
