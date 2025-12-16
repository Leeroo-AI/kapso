# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Low-level MoE operations

**Mechanism:** Permutation, routing, and torch GEMM implementations

**Significance:** Reference implementations for validation
