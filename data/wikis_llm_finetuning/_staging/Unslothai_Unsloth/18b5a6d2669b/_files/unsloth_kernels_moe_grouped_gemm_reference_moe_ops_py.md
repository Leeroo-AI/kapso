# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Core MoE operations: routing, permutation, unpermutation, and torch-native grouped GEMM reference.

**Mechanism:** calculate_topk() implements router logic with softmax/sigmoid activation and optional renormalization. get_routing_indices() computes per-expert token counts via histogram and generates gather indices (sort by expert assignment). permute() scatters tokens from token order to expert-grouped order using gather indices. unpermute() reverses this via index_copy_. torch_grouped_gemm() provides a simple PyTorch reference that loops over experts doing individual matmuls.

**Significance:** Building blocks for MoE implementations and correctness testing. The torch-native grouped GEMM serves as ground truth for validating Triton kernels. The routing utilities are reused across all MoE implementations (HF, torch, Triton). Understanding these operations is essential for debugging permutation-related issues in the fused kernels.
