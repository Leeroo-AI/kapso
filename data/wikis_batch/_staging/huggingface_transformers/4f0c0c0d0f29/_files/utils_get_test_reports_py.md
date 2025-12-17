# File: `utils/get_test_reports.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 271 |
| Functions | `is_valid_test_dir`, `run_pytest`, `handle_suite` |
| Imports | argparse, contextlib, important_files, os, pathlib, subprocess, tempfile, torch |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manually runs test suites locally with the same configuration as CI, supporting parallel execution and filtering.

**Mechanism:** Provides `handle_suite()` to execute pytest on test directories with configurable options (GPU/CPU filtering, slow tests, temporary cache, resume functionality, parallel process distribution). Uses `IMPORTANT_MODELS` subset for focused testing and generates structured test reports with machine-type prefixes. Supports test isolation via temporary HuggingFace Hub cache directories.

**Significance:** Enables developers to reproduce CI test conditions locally for debugging, allows parallel test execution across multiple processes for faster local testing, and provides a consistent interface for running model tests that mirrors the GitHub Actions workflow behavior.
