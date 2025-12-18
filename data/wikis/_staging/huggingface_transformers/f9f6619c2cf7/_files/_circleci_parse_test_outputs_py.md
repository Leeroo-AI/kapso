# File: `.circleci/parse_test_outputs.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 71 |
| Functions | `parse_pytest_output`, `parse_pytest_failure_output`, `parse_pytest_errors_output`, `main` |
| Imports | argparse, re |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Parses and aggregates pytest output to provide human-readable summaries of test failures, errors, and skipped tests. This utility helps developers quickly understand CI test results by grouping similar issues together.

**Mechanism:** The script uses regular expressions to parse pytest output lines looking for SKIPPED, FAILED, and ERROR patterns. For each category, it extracts the test file/location, error type, and reason, then aggregates them by reason to show counts and examples. The script provides three parsing functions: `parse_pytest_output` for skipped tests, `parse_pytest_failure_output` for failed tests, and `parse_pytest_errors_output` for errors. Results are sorted by frequency and printed with counts. The failure and error parsers exit with status code 1 if any issues are found, enabling CI failure detection.

**Significance:** This utility is essential for making large test suite results comprehensible in CircleCI. When hundreds or thousands of tests run in parallel across multiple containers, raw pytest output can be overwhelming. By aggregating similar failures and presenting summaries, this tool helps developers quickly identify systemic issues (e.g., "120 tests failed due to connection timeout") versus isolated problems, significantly reducing debugging time in CI environments.
