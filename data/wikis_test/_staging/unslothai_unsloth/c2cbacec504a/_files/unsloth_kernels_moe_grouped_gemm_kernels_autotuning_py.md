# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `... +4 more` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Defines kernel configuration generation and pruning logic for Triton autotuning.

**Mechanism:** Generates config combinations (block sizes, warps, stages, TMA flags), estimates shared memory requirements, prunes configs that exceed capacity or violate MoE constraints, dynamically disables unsupported features.

**Significance:** Critical for kernel autotuning - ensures only valid configs are tested, preventing wasted compilation time and runtime errors.
