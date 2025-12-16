# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_ops.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 151 |
| Functions | `permute`, `unpermute`, `calculate_topk`, `get_routing_indices`, `torch_grouped_gemm` |
| Imports | torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides fundamental MoE operations used as building blocks and reference implementations for testing.

**Mechanism:**

**Token routing operations**:
- `permute`: Gathers tokens from token order to expert-grouped order using indices (supports topk>1)
- `unpermute`: Scatters tokens back from expert-grouped order to token order
- `get_routing_indices`: Computes token counts per expert and gather indices from selected_experts via histc and argsort

**Router operations**:
- `calculate_topk`: Selects top-k experts with sigmoid or softmax activation, optional renormalization

**Core GEMM**:
- `torch_grouped_gemm`: Reference implementation using simple Python loop over experts, performs Y = X @ W^T for each group

**Significance:** Essential utility functions for MoE layers. The permute/unpermute operations are critical for token routing. The reference torch_grouped_gemm serves as ground truth for validating optimized Triton kernels. These operations are simple, readable implementations that prioritize correctness over performance.