# File: `unsloth/kernels/moe/tests/test_grouped_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1213 |
| Functions | `check_valid_config`, `test_grouped_gemm_forward_manual`, `test_grouped_gemm_forward_manual_autograd`, `test_grouped_gemm_forward_autotune`, `test_grouped_gemm_forward_autotune_autograd`, `test_grouped_gemm_backward_dX_manual`, `test_grouped_gemm_backward_dX_manual_autograd`, `test_grouped_gemm_backward_dX_autotune`, `... +5 more` |
| Imports | common, dataclasses, grouped_gemm, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive pytest test suite for validating grouped GEMM kernel correctness across forward pass, backward dX (input gradients), and backward dW (weight gradients) operations.

**Mechanism:** Uses pytest parametrization to test combinations of: kernel configs (block sizes, TMA load/store options), model configs (LLAMA4, QWEN, small debug sizes), data configs (sequence lengths 128/1024, bfloat16), and fusion flags (permute_x for input permutation, permute_y for output unpermutation). The `check_valid_config()` function enforces valid combinations (e.g., permute_x only for first GEMM, permute_y only for second GEMM). `_test_grouped_gemm_forward()` compares Triton kernel output against `torch_grouped_gemm` reference. `_test_grouped_gemm_backward_dX()` validates input gradient computation, handling the complexity of permute fusion in the backward pass. `_test_grouped_gemm_backward_dW()` validates weight gradient computation. Each test function has variants for manual kernel config specification vs autotuning, and for raw kernel calls vs autograd.Function interface.

**Significance:** The most comprehensive test file in the MoE test suite, providing exhaustive coverage of the grouped GEMM kernel implementation. Essential for ensuring correctness of the low-level Triton kernels that power MoE inference and training.
