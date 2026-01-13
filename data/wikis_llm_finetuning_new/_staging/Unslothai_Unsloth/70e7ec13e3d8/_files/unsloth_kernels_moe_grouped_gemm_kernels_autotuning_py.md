# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `... +4 more` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Autotuning utilities that generate Triton kernel configurations and prune invalid combinations for optimal grouped GEMM performance.

**Mechanism:** Defines default block sizes (M: 64/128, N: 64/128/256, K: 64/128/256), warp counts (4/8), and pipeline stages (3/4/5). Provides `get_forward_configs`, `get_dX_kernel_configs`, and `get_dW_kernel_configs` to generate `triton.Config` objects with all parameter combinations. The `common_prune_criteria` function eliminates configs exceeding shared memory capacity via `estimate_smem_reqs`. Mode-specific pruning functions (`prune_kernel_configs_fwd`, `prune_dX_configs`, `prune_kernel_configs_backward_dW`) disable incompatible TMA options when permutation is enabled. The `maybe_disable_tma` helper deactivates TMA on pre-SM90 GPUs.

**Significance:** Critical for achieving optimal kernel performance by automatically selecting the best configuration from a large parameter search space while avoiding invalid or resource-exceeding combinations.
