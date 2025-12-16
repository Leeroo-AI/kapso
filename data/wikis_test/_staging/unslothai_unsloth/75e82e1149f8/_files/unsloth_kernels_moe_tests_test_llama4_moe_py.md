# File: `unsloth/kernels/moe/tests/test_llama4_moe.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 262 |
| Functions | `annotated_context`, `get_text_config`, `prep_triton_kernel_traits`, `sparse_to_dense`, `_check_diff`, `run_backwards`, `_check_grads`, `test_llama4_ref` |
| Imports | argparse, contextlib, functools, grouped_gemm, pytest, sys, torch, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** End-to-end test suite for Llama4 MoE layer implementations, validating torch reference and Triton-optimized versions against HuggingFace.

**Mechanism:**

**Test setup**:
- Loads Llama4-Scout-17B-16E config from HuggingFace
- Creates three implementations:
  - HF reference (Llama4TextMoe)
  - Torch grouped GEMM (Llama4GroupedGemmTextMoe)
  - Triton optimized (Llama4TritonTextMoe)
- Copies weights to ensure identical initialization

**Validation stages**:
1. Forward pass comparison: torch vs HF
2. Forward pass comparison: Triton vs HF
3. Backward pass comparison: torch vs HF
4. Backward pass comparison: Triton vs HF

**Test parameters**:
- Parametrized over seq_len, dtype, permute_y, overlap_router_shared
- Tests both manual and autotuned kernel configurations
- Validates hidden states and router logits

**Gradient checking**:
- Verifies all parameter gradients match HF implementation
- Uses configurable tolerances per dtype

**Significance:** Integration test that validates the complete Llama4 MoE layer implementation against production HuggingFace models. Tests the router-shared expert overlapping optimization unique to Llama4. Critical for ensuring the optimized implementation maintains correctness for actual model architectures.