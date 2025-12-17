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

**Purpose:** Dynamically generates CircleCI configuration YAML files for running distributed test jobs across multiple parallel workers.

**Mechanism:** Defines job templates through the CircleCIJob dataclass (with parameters for Docker images, parallelism, pytest options, markers, etc.) and EmptyJob class. The create_circleci_config function reads test list files from test_preparation directory, instantiates job configurations (torch, tokenization, processor, pipeline tests, etc.), and generates a complete CircleCI workflow YAML with pipeline parameters, environment variables, and job dependencies. Includes sophisticated features like flaky test reruns with pattern matching, test splitting across parallel nodes, Hub cache downloading, and artifact collection.

**Significance:** Critical CI/CD infrastructure file that orchestrates the entire test suite execution strategy for the Transformers library. Enables efficient parallel testing across different job types (torch tests, examples, pipelines, hub tests, etc.) with customizable resource allocation and Docker environments, significantly reducing CI runtime.
