# File: `unsloth/kernels/moe/grouped_gemm/interface.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 968 |
| Classes | `GroupedGemm` |
| Functions | `supports_tma`, `get_per_device_per_stream_alloc_fn`, `log_kernel_info`, `grouped_gemm_forward`, `grouped_gemm_dX`, `grouped_gemm_dW`, `check_valid_config_fwd`, `check_valid_config_bwd_dW`, `... +2 more` |
| Imports | dataclasses, grouped_gemm, logging, torch, triton, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main interface for grouped GEMM operations in MoE layers

**Mechanism:** Implements forward, backward (dX, dW) passes for grouped matrix multiplications with optional token permutation fusion. Provides `GroupedGemm` autograd function and validation functions. Supports TMA (Tensor Memory Accelerator) for efficient memory access on Hopper+ GPUs. Handles expert-specific matrix operations where each expert processes variable-sized token groups.

**Significance:** Core API for MoE expert computation. Enables efficient batched processing of multiple experts by grouping tokens assigned to each expert, with optional fusions (permute_x, permute_y, fuse_mul_post) that eliminate separate kernels for common operations. Critical for MoE performance optimization.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
