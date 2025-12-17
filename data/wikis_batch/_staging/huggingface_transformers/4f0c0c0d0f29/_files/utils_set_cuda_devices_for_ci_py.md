# File: `utils/set_cuda_devices_for_ci.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 26 |
| Imports | argparse, os |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Configures CUDA device visibility for CI test runs based on model-specific GPU memory requirements.

**Mechanism:** Takes a test folder path as input, checks if the model requires special handling (e.g., Cohere's test_eager_matches_sdpa_generate needs multiple GPUs due to high memory usage), and outputs the appropriate CUDA_VISIBLE_DEVICES value (defaults to "0", Cohere gets "0,1,2,3", or uses existing environment variable).

**Significance:** Critical CI resource management utility that prevents out-of-memory errors by allocating multiple GPUs for memory-intensive model tests while conserving resources for standard tests. Specifically tuned for AWS CI runners to optimize GPU utilization and test reliability.
