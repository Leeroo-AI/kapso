# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference implementations of Qwen3 MoE block using grouped GEMM for testing and benchmarking.

**Mechanism:** Qwen3MoeGroupedGEMMBlock (PyTorch reference) and Qwen3MoeFusedGroupedGEMMBlock (Triton version) with similar structure to Llama4 implementation; handles router, token permutation, expert computation.

**Significance:** Reference implementation for Qwen3 MoE - enables testing and validating Qwen3 model optimizations.
