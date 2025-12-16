# File: `unsloth/kernels/moe/tests/test_grouped_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1213 |
| Functions | `check_valid_config`, `_test_grouped_gemm_forward`, `test_grouped_gemm_forward_manual`, `test_grouped_gemm_forward_manual_autograd`, `test_grouped_gemm_forward_autotune`, `test_grouped_gemm_forward_autotune_autograd`, `_test_grouped_gemm_backward_dX`, `test_grouped_gemm_backward_dX_manual`, `test_grouped_gemm_backward_dX_manual_autograd`, `test_grouped_gemm_backward_dX_autotune`, `test_grouped_gemm_backward_dX_autotune_autograd`, `_test_grouped_gemm_backward_dW`, `test_grouped_gemm_backward_dW_manual`, `test_grouped_gemm_backward_dW_manual_autograd`, `test_grouped_gemm_backward_dW_autotune`, `test_grouped_gemm_backward_dW_autotune_autograd` |
| Imports | common, dataclasses, grouped_gemm, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive pytest test suite for grouped GEMM kernels, validating forward pass, backward dX, and backward dW against torch reference implementations.

**Mechanism:**

**Test structure**:
- Each operation (forward, dX, dW) has manual and autotune variants
- Tests both direct kernel interface and autograd.Function interface
- Parametrized over data configs, model configs, kernel configs
- Validates against torch_grouped_gemm reference

**Forward tests**:
- Test permute_x (first GEMM) and permute_y (second GEMM) combinations
- Validate fuse_mul_post (topk weight merging) for inference
- Test with/without TMA loads and stores

**Backward dX tests**:
- Validate gradient w.r.t. input (dX = dY @ W^T)
- Handle topk reduction when permute_x is used
- Test permutation handling in both directions

**Backward dW tests**:
- Validate gradient w.r.t. weights (dW = X^T @ dY)
- Test accumulation across tokens for each expert
- Verify proper permutation application

All tests check numerical correctness within dtype-specific tolerances.

**Significance:** Critical test suite that ensures correctness of the core grouped GEMM operations. Catches regressions in permutation logic, gradient computation, and TMA usage. The parametrized structure provides extensive coverage across different configurations and model sizes.