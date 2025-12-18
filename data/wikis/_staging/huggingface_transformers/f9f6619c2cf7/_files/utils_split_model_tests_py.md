# File: `utils/split_model_tests.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 88 |
| Imports | argparse, ast, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Partitions model test directories into multiple slices for parallel execution in GitHub Actions CI workflows.

**Mechanism:** Lists all directories under `tests/models/` and all test directories directly under `tests/`, combines them with model directories prioritized first, then splits the complete list into N equal (or nearly equal) slices based on the `--num_splits` argument. Can optionally filter to only specific subdirectories via `--subdirs` argument, handling both with and without the `models/` prefix.

**Significance:** Critical CI infrastructure that enables parallelization of the extensive model test suite by distributing tests across multiple GitHub Actions jobs. This overcomes the 256-job matrix limit and significantly reduces overall CI runtime by running different model tests concurrently, essential for maintaining fast feedback cycles in a repository with hundreds of models.
