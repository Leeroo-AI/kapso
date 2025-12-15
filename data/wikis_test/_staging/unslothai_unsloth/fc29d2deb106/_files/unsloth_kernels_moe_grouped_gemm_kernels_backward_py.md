# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernels for grouped GEMM backward passes (dX and dW)

**Mechanism:** Implements two autotuned Triton kernels: `_grouped_gemm_dX_kernel` computes input gradients by multiplying grad_output with transposed weights; `_grouped_gemm_dW_kernel` computes weight gradients by multiplying transposed grad_output with inputs. Both handle per-expert processing with variable token counts, optional permutations, and TMA memory access. Use persistent thread blocks that process multiple tiles per expert.

**Significance:** Core backward pass implementation for MoE training. Enables efficient gradient computation with the same fusion opportunities as forward pass (permute_x/y). Critical for training performance as backward typically dominates wall time.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
