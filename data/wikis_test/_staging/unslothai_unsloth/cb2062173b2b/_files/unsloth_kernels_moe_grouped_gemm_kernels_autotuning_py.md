# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `... +4 more` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Autotuning configuration generation and pruning logic for Triton grouped GEMM kernels.

**Mechanism:**
- `get_forward_configs()` / `get_dX_kernel_configs()` / `get_dW_kernel_configs()`: Generate triton.Config objects from Cartesian products of block sizes (M, N, K), warps, stages, and TMA options
- Default parameters: BLOCK_M=[64,128], BLOCK_N=[64,128,256], BLOCK_K=[64,128,256], warps=[4,8], stages=[3,4,5]
- `estimate_smem_reqs()` / `exceeds_smem_capacity()`: Predict shared memory usage to prune infeasible configurations
- `common_prune_criteria()`: Filters configs that exceed SMEM, use oversized blocks for small inputs, or have invalid permutation combinations
- `prune_kernel_configs_fwd()` / `prune_dX_configs()` / `prune_kernel_configs_backward_dW()`: Mode-specific pruning that enforces constraints like "TMA_LOAD_X incompatible with PERMUTE_X"
- `maybe_disable_tma()`: Disables TMA on GPUs < sm_90

**Significance:** Critical for Triton's autotuning system to efficiently explore the kernel configuration space. By intelligently pruning invalid or infeasible configurations, reduces autotuning time while ensuring only valid combinations are tested. The pruning rules encode hardware constraints (SMEM limits, TMA availability) and algorithmic requirements (permutation/TMA incompatibilities).
