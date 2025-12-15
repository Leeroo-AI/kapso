# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `... +4 more` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generates and prunes kernel configurations for Triton autotuning

**Mechanism:** Creates Triton Config objects for forward/backward kernels with different block sizes, warps, stages, and TMA settings. Implements pruning logic to filter invalid/suboptimal configurations based on shared memory constraints, problem sizes, and hardware capabilities. Manages default parameter ranges for systematic exploration.

**Significance:** Crucial for kernel optimization. Autotuning explores the configuration space to find the fastest kernel parameters for each problem size. Pruning reduces search space by eliminating configurations that exceed hardware limits or are incompatible with permutation strategies.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
