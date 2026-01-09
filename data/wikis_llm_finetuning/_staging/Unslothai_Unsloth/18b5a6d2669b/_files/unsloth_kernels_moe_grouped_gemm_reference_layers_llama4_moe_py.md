# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference implementation of Llama4's MoE layer using grouped GEMM kernels.

**Mechanism:** Llama4TritonTextMoe extends HF's Llama4TextMoe, replacing expert computation with grouped_gemm calls. Handles Llama4-specific architecture: top-1 routing with sigmoid activation, shared expert that runs in parallel with routed experts, gate_up_proj fusion, and optional router/shared expert overlap via CUDA streams. Provides Llama4GroupedGemmTextMoe with torch-native grouped GEMM for comparison. Both classes manage weight layout transformations (in-place permutation) and save intermediate results for debugging.

**Significance:** Enables testing and benchmarking of Llama4 MoE acceleration. The reference implementation proves the grouped GEMM approach works correctly on Llama4's unique architecture (shared experts, top-1 routing). Serves as integration test target and performance baseline for the Triton kernels.
