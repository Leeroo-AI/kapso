# File: `.circleci/create_circleci_config.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 412 |
| Classes | `EmptyJob`, `CircleCIJob` |
| Functions | `create_circleci_config` |
| Imports | argparse, copy, dataclasses, os, typing, yaml |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Dynamically generates CircleCI configuration YAML files based on available test suites. This script creates the CI pipeline configuration that orchestrates parallel test execution across multiple Docker containers with custom environments and test splitting logic.

**Mechanism:** The script defines job configurations through the `CircleCIJob` dataclass and `EmptyJob` class, each specifying Docker images, environment variables, pytest options, parallelism settings, and installation steps. It reads test list files from the `test_preparation` directory to determine which jobs to include, then generates a complete CircleCI workflow with parameters for test lists and parallelism levels. The configuration includes sophisticated features like test splitting across parallel nodes, flaky test retry patterns, hub cache downloading, and test output parsing. Job definitions cover different test categories (torch, tokenization, processors, pipelines, examples, hub tests, exotic models, repo utils, and documentation tests) with appropriate Docker images and resource classes.

**Significance:** This is a critical CI infrastructure component that enables the transformers repository to run thousands of tests efficiently. By dynamically generating the CircleCI config based on which tests need to run, it provides flexibility in test execution while maintaining consistent test environments. The script's handling of parallelism, test splitting, and retry logic for flaky tests ensures robust CI runs despite the repository's large test suite and potential network/infrastructure issues.
