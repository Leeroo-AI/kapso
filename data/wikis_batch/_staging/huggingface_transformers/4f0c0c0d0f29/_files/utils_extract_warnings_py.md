# File: `utils/extract_warnings.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 134 |
| Functions | `extract_warnings_from_single_artifact`, `extract_warnings` |
| Imports | argparse, get_ci_error_statistics, json, os, time, transformers, zipfile |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Extracts and filters specific warning types from GitHub Actions CI artifacts for monitoring code quality.

**Mechanism:** Downloads CI artifacts containing `warnings.txt` files via GitHub API (using `get_artifacts_links` and `download_artifact` from get_ci_error_statistics). Parses warning blocks by detecting indented lines under "warnings summary" sections, filters to target warning types (DeprecationWarning, UserWarning, FutureWarning by default), and deduplicates into a set. Outputs selected warnings as JSON for analysis or tracking.

**Significance:** CI monitoring tool that helps track deprecation warnings and other code quality signals across test runs, enabling proactive identification of deprecated API usage before they become errors in future library versions.
