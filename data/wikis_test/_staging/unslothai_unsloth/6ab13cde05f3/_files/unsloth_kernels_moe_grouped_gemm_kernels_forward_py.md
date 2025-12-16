# File: `unsloth/kernels/moe/grouped_gemm/kernels/forward.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 265 |
| Imports | grouped_gemm, torch, triton |

## Understanding

**Status:** ✅ Explored

**Purpose:** Forward pass GEMM kernel

**Mechanism:** Triton kernel executing grouped matrix multiplications with expert-wise partitioning

**Significance:** Core computation kernel for MoE experts
