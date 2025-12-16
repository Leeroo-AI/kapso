# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Backward pass GEMM kernels

**Mechanism:** Triton JIT kernels for computing gradients wrt inputs and weights

**Significance:** Enables backpropagation through grouped GEMM
