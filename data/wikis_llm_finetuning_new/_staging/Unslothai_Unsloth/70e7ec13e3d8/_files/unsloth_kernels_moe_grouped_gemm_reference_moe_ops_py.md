# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Common MoE operations providing token permutation, routing index computation, and a reference torch-native grouped GEMM implementation.

**Mechanism:** `permute` reorders tokens from natural order to expert-grouped order using gather indices divided by topk. `unpermute` restores original order via `index_copy_`. `calculate_topk` handles sigmoid vs softmax routing with configurable pre/post activation. `get_routing_indices` uses `torch.histc` for expert token counts and `torch.argsort` for gather indices. `torch_grouped_gemm` iterates over experts, extracting per-expert input slices and computing X @ W^T using standard torch matmul.

**Significance:** Foundation utilities used by all MoE reference implementations - provides the baseline operations that Triton kernels must match for correctness while serving as fallback implementations.
