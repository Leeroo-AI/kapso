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

**Purpose:** Calculates optimal CircleCI parallelism levels for test jobs by analyzing test list artifacts and computing the ideal number of parallel nodes needed based on test counts.

**Mechanism:** The script reads a JSON artifact file containing URLs to test list files, counts the number of tests in each list by reading line counts from the test_preparation directory, applies a heuristic formula (tests divided by AVERAGE_TESTS_PER_NODES=5) capped at MAX_PARALLEL_NODES=8 to determine parallelism, and generates a transformed JSON output mapping each test job to its artifact URL and computed parallelism value for use in CircleCI configuration generation.

**Significance:** This tool optimizes CI resource utilization and job duration by dynamically scaling parallelism based on actual test counts rather than using fixed values. It ensures faster feedback for large test suites while avoiding over-provisioning nodes for small test sets, directly impacting CI costs and developer iteration speed.
