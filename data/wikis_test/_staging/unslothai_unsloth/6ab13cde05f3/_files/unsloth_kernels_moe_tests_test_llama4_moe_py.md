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

**Purpose:** Llama4 MoE block validation tests

**Mechanism:** Tests comparing HF reference, torch-native, and Triton implementations

**Significance:** Validates Llama4 architecture compatibility
