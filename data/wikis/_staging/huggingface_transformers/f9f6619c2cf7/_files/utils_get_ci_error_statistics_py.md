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

**Purpose:** Analyzes test failures from GitHub Actions workflows by downloading artifacts, parsing error reports, and generating statistical summaries. Creates actionable reports grouped by error type and model.

**Mechanism:** Uses GitHub API to fetch job links and artifact download URLs from a workflow run. Downloads all test artifact zip files, extracts failures_line.txt, summary_short.txt, and job_name.txt. Parses error messages and failed test names, counts occurrences, groups by error type and by model architecture. Generates markdown tables showing top errors and most problematic models with links to specific jobs.

**Significance:** Essential debugging and quality monitoring tool for large CI infrastructure. With hundreds of test jobs per workflow run, manual analysis is impractical. This automates failure triage, identifies systemic issues vs isolated failures, highlights which models need attention, and creates GitHub-ready reports for team communication and prioritization.
