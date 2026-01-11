# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/qwen3_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 348 |
| Classes | `GroupedGEMMResult`, `Qwen3MoeGroupedGEMMBlock`, `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference implementation of Qwen3's MoE layer using grouped GEMM kernels.

**Mechanism:** Qwen3MoeFusedGroupedGEMMBlock extends Qwen3MoeGroupedGEMMBlock (torch-native version), replacing torch_grouped_gemm with grouped_gemm Triton calls. Handles Qwen3-specific architecture: top-k routing with softmax + normalization, gate_up_proj fusion (concatenated gate and up projections), and down projection. Manages kernel configuration (autotuning or manual), permutation fusion flags, and from_hf() class method for easy instantiation from HuggingFace models. Saves all intermediate results (first_gemm, intermediate, second_gemm) in GroupedGEMMResult for debugging.

**Significance:** Primary test target for grouped GEMM correctness and performance on Qwen3 architecture. The large expert count (128 experts) and high top-k (8) make Qwen3 a challenging workload that stresses load balancing and memory bandwidth. Demonstrates the generality of the grouped GEMM approach across different MoE designs.
