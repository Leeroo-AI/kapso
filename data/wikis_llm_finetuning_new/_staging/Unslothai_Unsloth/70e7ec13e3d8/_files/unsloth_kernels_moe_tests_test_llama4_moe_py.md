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

**Purpose:** End-to-end test suite for validating Triton grouped GEMM MoE implementation against HuggingFace's Llama4TextMoe reference for the Llama-4-Scout-17B-16E model architecture.

**Mechanism:** Tests three implementation levels: HF reference (`Llama4TextMoe`), torch grouped GEMM reference (`Llama4GroupedGemmTextMoe`), and Triton implementation (`Llama4TritonTextMoe`). The `prep_triton_kernel_traits()` function configures kernel parameters, optionally limiting autotuning configs to 50 for test runtime. Tests are parametrized over: dtype (bfloat16), sequence length (1024), autotune mode, permute_x (disabled for Llama4), permute_y, and overlap_router_shared (whether router computation overlaps with shared expert). Forward pass validates output tensors and routing logits using `_check_diff()` with dtype-specific tolerances (1e-2 for bfloat16). Backward pass uses `run_backwards()` and `_check_grads()` to validate all parameter gradients match between implementations. Can also run standalone via `__main__` with command-line arguments.

**Significance:** Validates that the Triton MoE implementation produces numerically equivalent results to HuggingFace's Llama4 implementation. Critical for ensuring Unsloth's optimized kernels can be used as drop-in replacements for Llama 4 models.
