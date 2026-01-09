# File: `unsloth/kernels/moe/grouped_gemm/kernels/autotuning.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 396 |
| Functions | `val_to_list`, `convert_args_to_list`, `get_forward_configs`, `get_dX_kernel_configs`, `get_dW_kernel_configs`, `estimate_smem_reqs`, `exceeds_smem_capacity`, `common_prune_criteria`, `... +4 more` |
| Imports | itertools, logging, torch, triton, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Autotuning configuration generation and pruning for grouped GEMM Triton kernels.

**Mechanism:** Defines default ranges for block sizes (M/N/K), warp counts, and pipeline stages. Provides get_forward_configs(), get_dX_kernel_configs(), and get_dW_kernel_configs() to generate triton.Config objects from parameter combinations. Implements pruning functions that eliminate invalid configurations based on: shared memory capacity, permutation/TMA incompatibilities, token distribution across experts, and GPU compute capability.

**Significance:** Enables efficient autotuning by intelligently reducing the search space from thousands to hundreds of viable configurations. The pruning heuristics encode domain knowledge about MoE workload characteristics and hardware constraints, making autotuning practical for production use. Essential for achieving optimal performance across diverse model sizes and GPU architectures.
