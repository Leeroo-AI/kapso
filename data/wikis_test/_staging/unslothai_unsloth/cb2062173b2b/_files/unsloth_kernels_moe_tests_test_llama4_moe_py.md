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

**Purpose:** End-to-end integration tests for Llama 4 MoE blocks, validating torch grouped GEMM and Triton grouped GEMM implementations against HuggingFace reference across forward and backward passes.

**Mechanism:** Implements comprehensive testing flow:

1. Test Infrastructure:
   - `annotated_context()`: Context manager for formatted test output with delimiters
   - `get_text_config()`: Extracts text config from Llama4Config
   - `prep_triton_kernel_traits()`: Prepares kernel configs for autotuning or manual mode, reducing autotune configs to 50 for reasonable runtime
   - `sparse_to_dense()`: Converts sparse routing logits to dense format for comparison
   - `run_backwards()`: Executes backward pass and validates all parameter gradients exist
   - `_check_diff()` and `_check_grads()`: Validation helpers with tolerance checking

2. Main Test (`test_llama4_ref()`):
   - Parametrized across overlap_router_shared, permute_y, autotune, sequence lengths, dtypes
   - Tests against Llama-4-Scout-17B-16E configuration (5120 hidden, 8192 intermediate, 16 experts, top-1)
   - Three-way comparison:
     - HuggingFace `Llama4TextMoe` (reference)
     - `Llama4GroupedGemmTextMoe` (torch-native grouped GEMM)
     - `Llama4TritonTextMoe` (Triton kernel grouped GEMM)
   - Validates forward pass: output tensors and routing weights
   - Validates backward pass: all parameter gradients
   - Tests both sequential and overlapped router/shared-expert execution

3. Command-line Interface:
   - Allows standalone execution with configurable seqlen, dtype
   - Tests both overlap modes when run directly

Tolerances: bfloat16 (1e-2), float16 (1e-3), float32 (1e-5)

**Significance:** Critical integration test ensuring that the complete Llama 4 MoE stack (with unique features like shared experts and sigmoid routing) works correctly end-to-end. Validates that optimizations (Triton kernels, router/shared-expert overlap) maintain numerical correctness. The three-way comparison (HF → torch → Triton) helps isolate issues to either the grouped GEMM concept or the Triton implementation.
