# File: `unsloth/kernels/moe/tests/test_grouped_gemm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 1213 |
| Functions | `check_valid_config`, `test_grouped_gemm_forward_manual`, `test_grouped_gemm_forward_manual_autograd`, `test_grouped_gemm_forward_autotune`, `test_grouped_gemm_forward_autotune_autograd`, `test_grouped_gemm_backward_dX_manual`, `test_grouped_gemm_backward_dX_manual_autograd`, `test_grouped_gemm_backward_dX_autotune`, `... +5 more` |
| Imports | common, dataclasses, grouped_gemm, pytest, torch |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive pytest-based test suite for grouped GEMM forward and backward kernels, validating correctness across various configurations, model sizes, and optimization strategies.

**Mechanism:** Implements extensive testing infrastructure:

1. Configuration Validation:
   - `check_valid_config()`: Validates permute_x/permute_y combinations based on MoE fusion constraints (permute_x only for first GEMM, permute_y only for second GEMM, mutual exclusivity)

2. Forward Pass Testing (`_test_grouped_gemm_forward()`):
   - Tests both first (W1: gate_up_proj) and second (W2: down_proj) grouped GEMMs
   - Validates against torch-native grouped GEMM reference
   - Supports manual kernel configuration and autotuning
   - Tests both direct kernel interface and autograd interface
   - Validates permutation fusion (permute_x/permute_y) and topk weight fusion (fuse_mul_post)
   - Test variants: manual config, manual with autograd, autotune, autotune with autograd

3. Backward dX Testing (`_test_grouped_gemm_backward_dX()`):
   - Tests input gradient computation with permutation handling
   - Validates unpermutation and topk reduction for first GEMM (permute_x case)
   - Handles complex gradient flow with comments explaining torch autograd differences
   - Tests both manual interface and autograd interface
   - Test variants: manual, manual with autograd, autotune, autotune with autograd

4. Backward dW Testing (`_test_grouped_gemm_backward_dW()`):
   - Tests weight gradient computation
   - Validates permutation effects on weight gradients
   - Includes debug mode for detailed per-expert gradient inspection
   - Test variants: manual, manual with autograd, autotune, autotune with autograd

All tests are parametrized across:
- Data configs (sequence lengths, dtypes)
- Model configs (small test sizes, Llama, Qwen)
- Kernel configs (block sizes, warps, stages, TMA flags, permutation flags)
- W1/W2 selection (first vs second GEMM shapes)

**Significance:** The most comprehensive test coverage for grouped GEMM kernels, testing every kernel variant (forward, backward dX, backward dW) across all supported configurations. Critical for ensuring correctness of the core kernel operations that underpin MoE performance. The extensive parametrization (1000+ test cases) provides confidence in kernel correctness across diverse usage patterns.
