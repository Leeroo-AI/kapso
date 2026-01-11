# File: `unsloth/kernels/moe/tests/test_llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 262 |
| Functions | `annotated_context`, `get_text_config`, `prep_triton_kernel_traits`, `sparse_to_dense`, `run_backwards`, `model_config`, `test_llama4_ref` |
| Imports | argparse, contextlib, functools, grouped_gemm, pytest, sys, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration tests for Llama4 MoE layer implementation with grouped GEMM acceleration.

**Mechanism:** Parametrized pytest tests comparing three implementations: HF Llama4TextMoe (reference), Llama4GroupedGemmTextMoe (torch), and Llama4TritonTextMoe (Triton kernels). Tests forward outputs and backward gradients across sequence lengths (1024), dtypes (bf16), permutation modes, and autotuning. Uses annotated_context() for readable output, prep_triton_kernel_traits() to configure autotuning, and _check_diff() for numerical validation. Tests both standard mode and overlap_router_shared mode where shared expert runs in parallel.

**Significance:** Validates that grouped GEMM works correctly on Llama4's unique architecture (shared experts, top-1 routing, sigmoid activation). The overlap testing is particularly important as it exercises CUDA stream synchronization. Ensures Unsloth can accelerate production Llama4 models (Scout-17B-16E) without accuracy loss. Autotuning tests verify performance optimization doesn't break correctness.
