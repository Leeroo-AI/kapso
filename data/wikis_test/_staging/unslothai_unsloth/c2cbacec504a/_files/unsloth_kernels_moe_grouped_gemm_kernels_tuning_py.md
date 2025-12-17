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

**Purpose:** Data structures and utilities for kernel configuration management and device property detection.

**Mechanism:** Defines KernelConfig dataclasses (forward, backward_dX, backward_dW), config pruning rules, device properties caching, TritonTuningContext for error handling during autotuning.

**Significance:** Infrastructure for kernel parameter management and validation - ensures consistent configuration handling across benchmarks and tests.
