# File: `utils/split_doctest_jobs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Imports | argparse, collections, pathlib, tests_fetcher |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Organizes doctest files into job groups for parallel execution in GitHub Actions CI workflows.

**Mechanism:** Calls tests_fetcher.get_all_doctest_files() to get eligible files, groups them by directory path, isolates model_doc and tasks files for independent testing, and outputs either a directory-to-files mapping or splits the directory list into N chunks for matrix-based job distribution (bypassing GitHub's 256 job limit).

**Significance:** CI optimization utility that enables parallel doctest execution across hundreds of documentation and source files. Special handling of model_doc and tasks ensures independent job execution for better failure isolation, while the splitting capability allows scaling beyond GitHub Actions matrix limitations.
