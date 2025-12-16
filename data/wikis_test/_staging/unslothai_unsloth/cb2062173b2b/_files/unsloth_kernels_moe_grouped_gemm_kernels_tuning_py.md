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

**Purpose:** Manual kernel tuning infrastructure providing configuration dataclasses, result tracking, and pruning utilities for grouped GEMM kernels.

**Mechanism:**
- **Configuration Classes:**
  - `KernelConfig`: Base dataclass with BLOCK_SIZE_M/N/K, num_warps, num_stages, permute_x/y flags
  - `KernelConfigForward`: Extends with use_tma_load_w/x for forward pass
  - `KernelConfigBackward_dW`: Extends with use_tma_load_dy/x for weight gradient
  - `KernelConfigBackward_dX`: Extends with use_tma_load_dy/w for input gradient
- **Result Tracking:**
  - `KernelResult`: Stores torch_time, triton_time, speedup, and kernel_config
  - Provides to_dataframe(), to_csv(), print_table() for result analysis
- **Tuning Utilities:**
  - `get_kernel_configs()`: Generates and prunes configs for all modes (forward, dW, dX)
  - Mode-specific pruning functions enforce valid combinations (e.g., no TMA with permute)
  - `TritonTuningContext`: Context manager that catches OutOfResources errors during kernel execution
- **Device Properties:**
  - `get_device_properties()`: Caches GPU specs (NUM_SM, NUM_REGS, SIZE_SMEM, WARP_SIZE)

**Significance:** Enables manual kernel tuning workflow as an alternative to Triton autotuning. Provides structured way to specify kernel configurations, track benchmark results, and analyze performance characteristics. The pruning logic ensures only valid configurations are tested, while result tracking facilitates systematic performance optimization.
