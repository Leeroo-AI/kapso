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

**Purpose:** Provides dataclasses and utilities for manual kernel tuning, configuration management, and results tracking.

**Mechanism:**
- **Device Properties**: Caches GPU properties (SM count, registers, shared memory, warp size)
- **Kernel Config Classes**: Dataclasses for forward/backward configs with block sizes, warps, stages, TMA flags, permutation flags
- **Kernel Result**: Stores benchmark results (torch_time, triton_time, speedup) with conversion to DataFrame/CSV
- **Pruning Functions**: Validates and filters kernel configs (similar to autotuning.py but for manual tuning)
- **TritonTuningContext**: Context manager for handling OutOfResources errors during kernel testing

**Significance:** Infrastructure for systematic kernel parameter exploration and performance analysis. Complements autotuning.py by providing structured configuration management for manual tuning experiments. The result tracking enables data-driven kernel optimization decisions.