# File: `unsloth/kernels/moe/grouped_gemm/reference/layers/llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 437 |
| Classes | `Llama4MoeResult`, `Llama4GroupedGemmTextMoe`, `Llama4TritonTextMoe` |
| Imports | dataclasses, grouped_gemm, torch, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Reference implementations of Llama4 MoE block using grouped GEMM for testing and benchmarking.

**Mechanism:** Two classes - Llama4GroupedGemmTextMoe (PyTorch grouped GEMM reference), Llama4TritonTextMoe (Triton kernel version); handles weight permutation, router computation, expert assignment, and fusion options.

**Significance:** Reference implementation for Llama4 MoE - essential for testing kernel correctness and benchmarking performance against optimized implementations.
