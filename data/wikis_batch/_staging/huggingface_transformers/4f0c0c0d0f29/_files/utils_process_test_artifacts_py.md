# File: `utils/process_test_artifacts.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 75 |
| Functions | `count_lines`, `compute_parallel_nodes`, `process_artifacts` |
| Imports | json, math, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Calculates optimal parallelism levels for CircleCI test jobs based on test list sizes.

**Mechanism:** Reads test artifact URLs from a JSON file, counts lines in corresponding test list files, computes the number of parallel nodes needed using a heuristic (5 tests per node on average, max 8 nodes), and outputs a transformed JSON mapping each test job to its artifact URL and recommended parallelism level.

**Significance:** Essential CI optimization utility that dynamically adjusts parallelism to balance resource usage with test execution time. Prevents over-provisioning for small test suites and ensures adequate parallelization for large ones, making CI runs more efficient and cost-effective.
