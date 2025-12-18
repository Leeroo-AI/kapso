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

**Purpose:** Extracts and filters specific Python warnings from GitHub Actions workflow artifacts. Collects DeprecationWarning, UserWarning, and FutureWarning messages from CI test runs for analysis and tracking.

**Mechanism:** Downloads artifacts from a workflow run using GitHub API (or reads pre-downloaded artifacts from actions/download-artifact), parses warnings.txt files from zip archives, filters warnings by type using target list, deduplicates warnings, and outputs sorted JSON file with selected warnings. Reuses get_artifacts_links and download_artifact functions from get_ci_error_statistics module.

**Significance:** Proactive code quality monitoring tool. Helps maintainers identify and address deprecation warnings before they become breaking changes. By aggregating warnings across all CI jobs, it provides comprehensive view of technical debt and upcoming compatibility issues across the massive codebase.
