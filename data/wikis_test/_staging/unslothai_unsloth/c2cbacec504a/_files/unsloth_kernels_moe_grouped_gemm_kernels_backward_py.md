# File: `unsloth/kernels/moe/grouped_gemm/kernels/backward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 502 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Triton kernels for backward pass (gradient computation) for both dX and dW.

**Mechanism:** Two Triton JIT kernels (_grouped_gemm_dX_kernel, _grouped_gemm_dW_kernel) with TMA support, permutation fusion, loop unrolling optimizations; autotuned versions available for different hardware configurations.

**Significance:** Core backward pass computation - critical for training performance of MoE models.
