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

**Purpose:** Executes the transformers test suite locally with the same configuration and options used in CI workflows.

**Mechanism:** Runs pytest on test subdirectories with proper filtering (device tests, slow tests), supports parallel execution across multiple processes using interleaved distribution, manages temporary cache directories per test to isolate runs, and provides resume functionality to continue interrupted test runs. Automatically detects machine type (CPU/single-GPU/multi-GPU) and generates structured test reports.

**Significance:** Essential development tool that allows contributors to reproduce CI test behavior locally before pushing changes. Reduces CI usage and iteration time by enabling local verification of test suites, particularly useful for testing specific models or subsets using the important models list or custom selections.
