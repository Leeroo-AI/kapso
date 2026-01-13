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

**Purpose:** Manual tuning utilities providing kernel configuration dataclasses, device property queries, and a context manager for handling Triton compilation errors.

**Mechanism:** Defines `KernelConfig` base dataclass with block sizes, warps, stages, and permutation options. Specialized configs `KernelConfigForward`, `KernelConfigBackward_dW`, and `KernelConfigBackward_dX` add TMA load flags specific to each pass direction. `DeviceProperties` caches GPU SM count, registers, shared memory, and warp size via `get_device_properties()`. The `KernelResult` dataclass stores benchmark timing and provides `to_dataframe`/`to_csv` for result export. `TritonTuningContext` is a context manager that catches `OutOfResources` exceptions during manual tuning sweeps. Pruning functions filter invalid TMA+permute combinations.

**Significance:** Provides the configuration infrastructure for both autotuning and manual kernel parameter exploration, enabling systematic performance optimization of grouped GEMM kernels.
