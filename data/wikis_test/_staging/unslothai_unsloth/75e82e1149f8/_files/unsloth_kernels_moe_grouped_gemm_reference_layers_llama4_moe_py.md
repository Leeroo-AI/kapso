# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides reference and optimized implementations of Llama4's MoE (Mixture of Experts) layer using grouped GEMM operations.

**Mechanism:**

**Llama4GroupedGemmTextMoe** (torch-native grouped GEMM):
- Inherits from HuggingFace's Llama4TextMoe
- Permutes expert weights in-place to [E, N, K] layout for grouped GEMM
- Uses torch_grouped_gemm for expert computation
- Supports overlapping router and shared expert computation via CUDA streams
- Pre-multiplies router weights with hidden states (topk=1 for Llama4)

**Llama4TritonTextMoe** (optimized Triton kernel):
- Extends Llama4GroupedGemmTextMoe
- Replaces torch_grouped_gemm with fused Triton kernels
- Supports permute_y fusion (scatter) and autotuning
- Provides manual kernel config interface for performance tuning

Both implementations preserve numerical equivalence with HF while optimizing the expert computation bottleneck.

**Significance:** Production-ready MoE implementation for Llama4 models. The torch reference validates correctness while the Triton version delivers production performance. Critical for enabling efficient fine-tuning and inference of Llama4-Scout and similar architectures.