# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Llama4 MoE block implementations

**Mechanism:** Standard HF, torch-native grouped GEMM, and fused Triton implementations

**Significance:** Validates grouped GEMM correctness against HF reference
