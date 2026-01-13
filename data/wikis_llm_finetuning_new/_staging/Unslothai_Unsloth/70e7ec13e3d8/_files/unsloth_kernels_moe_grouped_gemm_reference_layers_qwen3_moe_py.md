# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference and optimized implementations of Qwen3's sparse MoE block, providing both torch-native grouped GEMM and Triton kernel variants.

**Mechanism:** `Qwen3MoeGroupedGEMMBlock` extracts and stacks expert weights from HF's per-expert format into fused [E, 2N, K] gate_up_proj and [E, K, N] down_proj tensors. Uses softmax routing with optional normalization. The forward pass permutes tokens to expert order, applies two grouped GEMMs with SiLU activation, then unpermutes and merges topk weights. `Qwen3MoeFusedGroupedGEMMBlock` replaces `torch_grouped_gemm` with the Triton `grouped_gemm` interface, supporting fused permutation via permute_x/permute_y flags. The `GroupedGEMMResult` dataclass stores all intermediate results for debugging.

**Significance:** Provides Qwen3 MoE support with optimized Triton kernels, enabling efficient training and inference for Qwen3-MoE models while maintaining a reference for correctness testing.
