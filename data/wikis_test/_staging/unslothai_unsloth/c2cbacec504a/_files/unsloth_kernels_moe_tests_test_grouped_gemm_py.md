# File: `unsloth/kernels/moe/tests/test_grouped_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1213 |
| Functions | `check_valid_config`, `test_grouped_gemm_forward_manual`, `test_grouped_gemm_forward_manual_autograd`, `test_grouped_gemm_forward_autotune`, `test_grouped_gemm_forward_autotune_autograd`, `test_grouped_gemm_backward_dX_manual`, `test_grouped_gemm_backward_dX_manual_autograd`, `test_grouped_gemm_backward_dX_autotune`, `... +5 more` |
| Imports | common, dataclasses, grouped_gemm, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive test suite for grouped GEMM kernels (forward and backward) with various configurations.

**Mechanism:** Parametrized pytest tests covering forward pass (manual/autotuning), backward dX/dW passes, all permutation combinations, model sizes, data types; includes config validation and result comparison.

**Significance:** Main test suite ensuring kernel correctness - validates all kernel variants against reference implementations.
