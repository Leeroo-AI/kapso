# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementations for backward pass gradients in grouped GEMM: `_grouped_gemm_dX_kernel` (gradient w.r.t. input) and `_grouped_gemm_dW_kernel` (gradient w.r.t. weights).

**Mechanism:** The dX kernel computes dY @ W for each expert group, iterating over experts and processing tiles assigned to the current thread block. The dW kernel computes dY^T @ X, accumulating gradients for each expert's weight matrix. Both kernels support TMA loads via `tl._experimental_make_tensor_descriptor`, permutation handling (PERMUTE_X for first GEMM, PERMUTE_Y for second GEMM), and configurable block sizes. Uses persistent kernel pattern with tile-based work distribution across SMs. Autotuned versions wrap the base kernels with `triton.autotune` decorators keyed on problem dimensions and permutation settings.

**Significance:** Essential for training MoE models - enables efficient gradient computation through the grouped GEMM operations used in expert MLPs, with optimizations matching the forward pass.
