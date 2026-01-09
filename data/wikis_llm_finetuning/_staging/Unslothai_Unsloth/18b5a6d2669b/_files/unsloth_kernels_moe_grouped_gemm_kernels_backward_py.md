# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementations for grouped GEMM backward passes (dX and dW gradients).

**Mechanism:** Implements _grouped_gemm_dX_kernel (gradient wrt inputs: dX = dY @ W.T) and _grouped_gemm_dW_kernel (gradient wrt weights: dW = X.T @ dY). Both kernels iterate over expert groups, loading tiles with TMA or global memory, computing matrix products with tensor cores, and accumulating results. Supports permutation fusion where dX can scatter gradients back to token order, and dW can gather inputs from permuted order. Uses @triton.autotune decorator for performance optimization.

**Significance:** Completes the autograd implementation for MoE training. These kernels must match forward pass fusion decisions (permute_x/permute_y) to ensure correct gradient flow. The dW kernel is particularly challenging as it accumulates gradients across dynamically-sized expert groups while maintaining numerical accuracy. Critical for enabling efficient MoE fine-tuning in Unsloth.
