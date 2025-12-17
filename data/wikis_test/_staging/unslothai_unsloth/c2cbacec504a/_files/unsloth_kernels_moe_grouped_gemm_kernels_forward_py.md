# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements Triton kernel for forward pass (matrix multiplication) with MoE-specific fusions.

**Mechanism:** Single Triton JIT kernel (_grouped_gemm_forward_kernel) supporting permute_x (load permutation), permute_y (store permutation), fuse_mul_post (weight multiplication), TMA loads, and autotuning.

**Significance:** Core forward pass computation - most performance-critical component for MoE inference and training.
