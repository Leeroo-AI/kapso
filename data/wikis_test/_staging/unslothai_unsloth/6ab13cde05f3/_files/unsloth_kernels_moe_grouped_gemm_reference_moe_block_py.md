# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Base MoE block utilities

**Mechanism:** Tensor permutation/unpermutation, topk calculation, routing index computation

**Significance:** Core MoE algorithmic primitives
