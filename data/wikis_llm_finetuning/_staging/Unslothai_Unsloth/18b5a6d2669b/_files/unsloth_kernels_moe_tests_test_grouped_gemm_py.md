# File: `unsloth/kernels/moe/tests/test_grouped_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1213 |
| Functions | `check_valid_config`, `test_grouped_gemm_forward_manual`, `test_grouped_gemm_forward_manual_autograd`, `test_grouped_gemm_forward_autotune`, `test_grouped_gemm_forward_autotune_autograd`, `test_grouped_gemm_backward_dX_manual`, `test_grouped_gemm_backward_dX_manual_autograd`, `test_grouped_gemm_backward_dX_autotune`, `... +5 more` |
| Imports | common, dataclasses, grouped_gemm, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for grouped GEMM kernel primitives (forward, dX, dW).

**Mechanism:** Parametrized pytest tests covering grouped_gemm_forward, grouped_gemm_dX, and grouped_gemm_dW across: small/medium/large model sizes, different expert counts (4-128), topk values (1-8), permutation variants (permute_x/permute_y), TMA flags, kernel configurations, and data types. Uses check_valid_config() to skip invalid combinations. Compares Triton kernel outputs against torch_grouped_gemm reference with assert_close() tolerances. Tests both autotuning and manual configuration paths.

**Significance:** Foundation of correctness validation. These tests catch kernel bugs early by testing primitives in isolation before integration. The parametrization ensures coverage of edge cases (single expert, all experts, unbalanced routing). Critical for regression testing when modifying kernel implementations. Must pass before MoE integration tests are meaningful.
