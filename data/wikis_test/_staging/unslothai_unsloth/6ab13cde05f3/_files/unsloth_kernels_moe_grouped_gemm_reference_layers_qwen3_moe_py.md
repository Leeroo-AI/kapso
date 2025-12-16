# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Qwen3 MoE block implementations

**Mechanism:** Torch-native and Triton fused versions with weight extraction and routing

**Significance:** Reference implementation for Qwen3 MoE validation
