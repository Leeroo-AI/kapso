# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides core torch-native implementations of MoE primitive operations used as reference implementations for testing optimized Triton kernels.

**Mechanism:** Implements five fundamental MoE operations:
- `permute()`: Reorders tokens from token order to expert-grouped order using gather indices, with optimization for topk=1 case
- `unpermute()`: Restores tokens to original token order using index_copy_ for scatter operation
- `calculate_topk()`: Computes top-k expert selections with configurable activation (sigmoid/softmax), supporting pre-activation or post-activation topk and optional renormalization
- `get_routing_indices()`: Uses histogram and argsort to compute token counts per expert and gather indices for permutation, with optional scatter indices for reverse operation
- `torch_grouped_gemm()`: Pure PyTorch implementation of grouped GEMM that processes each expert's tokens separately in a loop, performing matrix multiplication with optional weight transpose

These operations are intentionally simple and readable to serve as ground truth for correctness testing.

**Significance:** Critical reference implementations that define correct behavior for all MoE operations. Used throughout the test suite to validate Triton kernel outputs. The torch_grouped_gemm function is particularly important as it defines the expected behavior that optimized kernels must match.
