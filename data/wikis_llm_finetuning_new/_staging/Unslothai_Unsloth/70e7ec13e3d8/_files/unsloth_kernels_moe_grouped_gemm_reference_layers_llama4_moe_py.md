# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference and optimized implementations of Llama4's Mixture-of-Experts block, including torch-native and Triton grouped GEMM variants.

**Mechanism:** `Llama4GroupedGemmTextMoe` extends HF's `Llama4TextMoe`, permuting expert weights to [E, N, K] layout and using `torch_grouped_gemm` for computation. Implements Llama4's sigmoid-based routing with pre-multiplication of hidden states by routing weights. `Llama4TritonTextMoe` replaces torch operations with calls to `grouped_gemm` interface using Triton kernels. Supports CUDA stream overlap for shared expert computation via `overlap_router_shared`. The `Llama4MoeResult` dataclass captures intermediate tensors for debugging. Weight copying and validation methods enable conversion from HF checkpoints.

**Significance:** Enables efficient Llama4 MoE training/inference with Triton kernels while providing a reference implementation for correctness validation and debugging.
