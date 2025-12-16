# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Triton kernel implementations for backward pass gradients in grouped GEMM MoE operations.

**Mechanism:**
- `_grouped_gemm_dX_kernel()`: Computes gradient w.r.t. inputs (dX = dY @ W^T) with shape [NUM_TOKENS*TOPK, K]
  - Iterates over experts, processing tiles of dY and W to accumulate dX
  - Handles PERMUTE_X (store in permuted order from expert→token) and PERMUTE_Y (load in permuted order from token→expert)
  - Supports TMA loading of dY and W, TMA storing of dX
  - Uses gather_indices for permuted memory access patterns
- `_grouped_gemm_dW_kernel()`: Computes gradient w.r.t. weights (dW = X^T @ dY) with shape [NUM_EXPERTS, N, K]
  - Iterates over output tiles (N×K) for each expert, accumulating contributions from all tokens assigned to that expert
  - Handles PERMUTE_X (load X in permuted token→expert order) and PERMUTE_Y (load dY in permuted token→expert order)
  - Supports TMA loading of X and dY, TMA storing of dW
- Both kernels decorated with `@triton.autotune` using mode-specific config generators and pruning functions
- Grid parallelization across SMs, with work distribution based on tile counts

**Significance:** Core backward pass kernels enabling gradient-based training of MoE models. The dX kernel accumulates gradients for tokens routed to multiple experts (TOPK), while dW kernel aggregates gradients across all tokens assigned to each expert. Support for permutations and TMA loads/stores is essential for efficient memory access patterns in MoE architectures.
