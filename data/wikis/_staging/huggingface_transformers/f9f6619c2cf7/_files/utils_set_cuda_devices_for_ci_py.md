# File: `utils/set_cuda_devices_for_ci.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 26 |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Determines the CUDA_VISIBLE_DEVICES setting for CI test runs by applying model-specific GPU allocation rules to handle memory-intensive tests on AWS runners.

**Mechanism:** The script accepts a --test_folder argument specifying which model is being tested, checks if the model requires special GPU allocation (currently only "models/cohere" needs all 4 GPUs for test_eager_matches_sdpa_generate due to high memory requirements), falls back to existing CUDA_VISIBLE_DEVICES environment variable if set, defaults to single GPU "0" for standard tests, and prints the computed value for capture by GitHub Actions workflows.

**Significance:** This simple but critical tool prevents out-of-memory failures in CI by ensuring memory-intensive model tests get sufficient GPU resources while conserving GPUs for parallel test execution in standard cases. It provides a centralized, maintainable way to encode model-specific resource requirements without hardcoding them in workflow files, supporting the efficient utilization of AWS CI runner hardware.
