# File: `unsloth/kernels/moe/grouped_gemm/kernels/tuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 277 |
| Classes | `DeviceProperties`, `KernelConfig`, `KernelConfigForward`, `KernelConfigBackward_dW`, `KernelConfigBackward_dX`, `KernelResult`, `TritonTuningContext` |
| Functions | `get_device_properties`, `get_kernel_configs`, `prune_kernel_configs_fwd`, `prune_kernel_configs_backward_dX`, `prune_kernel_configs_backward_dW` |
| Imports | collections, dataclasses, grouped_gemm, itertools, pandas, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Manual kernel tuning infrastructure and configuration management

**Mechanism:** Defines dataclasses for kernel configurations (KernelConfig, KernelConfigForward, KernelConfigBackward_dX/dW) specifying block sizes, warps, stages, TMA settings, and permutation flags. Provides KernelResult for storing benchmark results, TritonTuningContext for error handling, and functions to generate/prune configuration spaces. Manages device properties and converts results to DataFrames.

**Significance:** Essential infrastructure for both manual tuning and autotuning. Standardizes kernel parameter representation, enables systematic performance testing, and provides utilities for analyzing results. The pruning functions prevent testing invalid configurations, significantly speeding up the tuning process.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
