# File: `utils/split_doctest_jobs.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 98 |
| Imports | argparse, collections, pathlib, tests_fetcher |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Organizes doctest files into groups for parallel execution in GitHub Actions CI workflows, bypassing the 256-job matrix limit.

**Mechanism:** Fetches all doctest files using `tests_fetcher.get_all_doctest_files()`, groups them by directory path, but keeps files in `docs/source/en/model_doc` and `docs/source/en/tasks` as individual entries rather than grouping them together. Excludes files in `src/` directory due to current test failures. Can output either a dictionary mapping directory paths to files or split the keys into N lists for matrix generation.

**Significance:** Essential CI infrastructure utility that enables efficient parallel doctest execution. By splitting documentation tests into independent jobs, it allows the test suite to scale beyond GitHub Actions' default matrix limitations while running model documentation tests independently to isolate failures and speed up overall test runs.
