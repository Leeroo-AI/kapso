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

**Purpose:** Data structures and utilities for manual kernel configuration and performance tuning.

**Mechanism:** Defines dataclasses KernelConfigForward, KernelConfigBackward_dW, KernelConfigBackward_dX encoding all tuning parameters (block sizes, warps, stages, TMA flags, permutation flags). Provides KernelResult for storing benchmark data and DeviceProperties for GPU capabilities. Includes pruning functions to filter invalid config combinations and TritonTuningContext for handling kernel compilation failures gracefully during manual tuning sweeps.

**Significance:** Enables reproducible performance experiments and production deployments with fixed kernel configurations. While autotuning is convenient for research, production systems often use validated manual configs to avoid runtime overhead and ensure deterministic behavior. The dataclass approach makes configs serializable and easy to share across teams.
