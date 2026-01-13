# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Alternative Triton-based Qwen3 MoE block implementation that inherits from the torch-native reference and overrides with grouped GEMM kernel calls.

**Mechanism:** `Qwen3MoeFusedGroupedGEMMBlock` extends `Qwen3MoeGroupedGEMMBlock` from moe_ops, storing permute_x, permute_y, autotune flags, and kernel configurations. The `from_hf` class method extracts weights from HF's `Qwen3MoeSparseMoeBlock`. Forward pass conditionally applies permutation based on permute_x flag, calls `grouped_gemm` for both gate_up_proj and down_proj, applies activation, and merges topk weights. Supports dW_only and dX_only modes for gradient computation debugging.

**Significance:** Cleaner alternative implementation demonstrating how to swap torch-native grouped GEMM with Triton kernels while reusing the reference block's routing and weight extraction logic.
