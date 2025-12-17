# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Utility functions for MoE operations (permutation, unpermutation, routing, torch grouped GEMM).

**Mechanism:** Implements tensor permutation by expert indices, unpermutation to restore token order, routing index generation, PyTorch-native grouped GEMM reference implementation.

**Significance:** Core MoE operations used by reference implementations and tests - provides ground truth for kernel validation.
