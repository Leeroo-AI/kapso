# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Functions | `_grouped_gemm_dX_kernel`, `_grouped_gemm_dW_kernel` (Triton JIT kernels) |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Triton kernels for computing gradients (dX and dW) in the backward pass of grouped GEMM operations for MoE layers.

**Mechanism:**

**dX kernel** (gradient w.r.t. input):
- Computes dX = dY @ W^T for each expert group
- Iterates over experts, processing tiles in parallel across SMs
- Supports permute_x (gather on store) and permute_y (gather on load)
- Uses TMA for efficient loading of dY and W tensors (optional)
- Accumulates in fp32, converts to output dtype

**dW kernel** (gradient w.r.t. weights):
- Computes dW = X^T @ dY for each expert
- Processes output tiles (N×K) in parallel, accumulates across M (tokens)
- Handles permute_x (token→expert on X load) and permute_y (token→expert on dY load)
- Supports TMA loads for X and dY, TMA store for dW
- Each expert gets accumulated separately

Both kernels support autotuning via @triton.autotune decorators with configs from autotuning.py.

**Significance:** Core backward pass implementation enabling efficient training of MoE models. The fused permutations eliminate separate gather/scatter kernels, reducing memory bandwidth. Critical for achieving competitive training performance compared to standard implementations.