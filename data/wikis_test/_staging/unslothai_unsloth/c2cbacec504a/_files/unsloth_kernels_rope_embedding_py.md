# File: `unsloth/kernels/rope_embedding.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 465 |
| Classes | `Fast_RoPE_Embedding`, `Fast_RoPE_Embedding_QK`, `Slow_RoPE_Embedding` |
| Functions | `fast_rope_embedding`, `inplace_rope_embedding` |
| Imports | device_type, torch, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements rotary position embedding (RoPE) with optimized Triton kernels supporting individual Q,K or joint Q,K processing and rope_indices for flexible position mapping.

**Mechanism:** Kernel applies 2D rotation: [q0,q1]' = [cos,sin; -sin,cos] @ [q0,q1] (Q1*cos-Q2*sin, Q2*cos+Q1*sin) for half_head_dim pairs. Two paths: (1) Fast_RoPE_Embedding processes Q and K separately with broadcasting via ROPE_GROUP_SIZE for head parallelization, (2) Fast_RoPE_Embedding_QK processes Q and K jointly with rope_indices support for dynamic position computation. Backward negates sin for inverse rotation.

**Significance:** RoPE is critical for position encoding performance; kernel fusion provides 10% speedup vs PyTorch, significant for long-sequence inference where RoPE dominates compute.
