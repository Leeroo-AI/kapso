# File: `utils/get_pr_run_slow_jobs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 133 |
| Functions | `get_jobs_to_run`, `parse_message`, `get_jobs`, `check_name` |
| Imports | argparse, json, re, string |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines which model/quantization tests to run in slow CI based on PR file changes or comment directives.

**Mechanism:** Parses PR file changes using regex patterns to identify modified modeling files under `src/transformers/models/` or `tests/models/`, or extracts model names from PR comments with `run-slow`/`run_slow` prefixes. Maps these to test directories and outputs a filtered list respecting a maximum of 16 jobs.

**Significance:** Critical CI optimization tool that prevents running all slow tests on every PR by intelligently selecting only relevant test suites based on code changes or manual specification, significantly reducing CI runtime and resource usage.
