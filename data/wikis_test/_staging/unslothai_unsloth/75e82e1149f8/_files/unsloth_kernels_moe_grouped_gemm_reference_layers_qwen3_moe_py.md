# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides reference and optimized implementations of Qwen3's MoE (Mixture of Experts) block using grouped GEMM operations.

**Mechanism:**

**Qwen3MoeGroupedGEMMBlock** (torch-native grouped GEMM):
- Extracts weights from HuggingFace Qwen3MoeSparseMoeBlock
- Fuses gate_proj and up_proj into single [E, 2*N, K] tensor
- Uses torch_grouped_gemm for both gate_up_proj and down_proj
- Implements router with softmax and topk selection (topk=8 for Qwen3)
- Performs permute/unpermute and topk weight merging separately

**Qwen3MoeFusedGroupedGEMMBlock** (optimized Triton kernel):
- Replaces torch_grouped_gemm with fused Triton kernels
- Supports permute_x (gather) and permute_y (scatter) fusions
- Enables autotuning or manual kernel configuration
- Provides dW_only and dX_only modes for gradient debugging

Both store intermediate results (first_gemm, intermediate, second_gemm, etc.) for detailed correctness checking.

**Significance:** Production-ready MoE implementation for Qwen3 models. The reference implementation validates correctness against HuggingFace while the fused version delivers optimized performance. Essential for efficient training/inference of Qwen3-30B-A3B and similar dense-sparse mixture architectures.