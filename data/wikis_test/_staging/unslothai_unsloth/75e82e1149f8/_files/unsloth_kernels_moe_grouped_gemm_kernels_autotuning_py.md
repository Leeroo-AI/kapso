# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `maybe_disable_tma`, `prune_kernel_configs_fwd`, `prune_dX_configs`, `prune_kernel_configs_backward_dW` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates and prunes Triton kernel configurations for autotuning grouped GEMM operations in MoE layers.

**Mechanism:**
- Defines default ranges for block sizes (M, N, K), warps, stages, and TMA settings
- Creates config combinations via Cartesian product for forward, dW, and dX kernels
- Prunes invalid configurations based on:
  - Shared memory capacity (estimates based on block sizes and stages)
  - Invalid permutation combinations (permute_x with TMA load x, etc.)
  - Token distribution (block size vs tokens per expert)
  - GPU capability (TMA only on sm90+)
- Returns triton.Config objects for autotuner

**Significance:** Critical for automated kernel parameter optimization. Generates the search space for autotuning while eliminating configurations that would fail or perform poorly. Reduces tuning time by pruning ~50-80% of invalid configs before benchmarking.