# File: `unsloth/device_type.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 127 |
| Functions | `is_hip`, `get_device_type`, `get_device_count` |
| Imports | functools, inspect, torch, unsloth_zoo |

## Understanding

**Status:** ✅ Explored

**Purpose:** Device detection and compatibility checking for GPU hardware

**Mechanism:** Provides functions to detect device type (CUDA/HIP/XPU) with fallback logic, checks GPU availability via torch.cuda/torch.xpu/torch.accelerator, determines device count, and assesses bitsandbytes compatibility for AMD GPUs including warp size and block size considerations

**Significance:** Essential hardware abstraction layer that enables Unsloth to support multiple GPU vendors (NVIDIA CUDA, AMD ROCm, Intel XPU) while managing platform-specific quantization constraints
