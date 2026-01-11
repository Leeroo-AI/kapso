# File: `unsloth/kernels/moe/grouped_gemm/reference/moe_block.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 161 |
| Classes | `Qwen3MoeFusedGroupedGEMMBlock` |
| Imports | grouped_gemm, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Generic MoE block implementation using Triton grouped GEMM, providing a cleaner reference than model-specific versions.

**Mechanism:** Qwen3MoeFusedGroupedGEMMBlock (confusingly located here, not in layers/) extends Qwen3MoeGroupedGEMMBlock to replace torch-native grouped GEMM with Triton kernels. Implements forward pass with two grouped GEMM calls (gate_up_proj and down_proj), handling routing, permutation fusion, and topk weight merging. Accepts kernel configs for forward and backward passes, supporting both autotuning and manual configuration. The from_hf() class method extracts weights from HuggingFace models.

**Significance:** Serves as the production-ready template for integrating grouped GEMM into MoE models. Unlike the layers/ implementations which save extensive debug info, this version is cleaner and suitable for adaptation to new MoE architectures. Demonstrates the minimal code needed to accelerate MoE with grouped GEMM.
