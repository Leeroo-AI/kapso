# File: `utils/print_env.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 76 |
| Imports | os, sys, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Diagnostic script that prints environment information including Python, transformers, accelerator, and framework versions.

**Mechanism:** Prints Python version, transformers version, and conditionally imports torch/deepspeed/torchcodec to display versions and device counts. Detects accelerator type (CUDA/XPU/HPU) via `is_torch_hpu_available()` and `is_torch_xpu_available()` checks. For CUDA: shows CUDA version, CuDNN version, GPU count, NCCL version. For XPU: SYCL version and device count. For HPU: HPU version and device count. Also attempts to detect FFmpeg version via torchcodec.

**Significance:** Essential debugging tool for CI logs and issue reports, providing a standardized snapshot of the execution environment. Helps maintainers quickly identify version mismatches, hardware availability issues, and configuration problems when investigating test failures or user-reported bugs.
