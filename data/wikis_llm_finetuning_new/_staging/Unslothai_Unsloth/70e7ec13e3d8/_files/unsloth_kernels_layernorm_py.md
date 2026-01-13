# File: `unsloth/kernels/layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 225 |
| Classes | `Fast_Layernorm` |
| Functions | `layernorm_forward`, `layernorm_backward`, `fast_layernorm`, `test_layernorm`, `testing_suite_layernorm` |
| Imports | torch, triton, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements fast Triton kernels for standard Layer Normalization (with mean subtraction) used in non-Llama architectures.

**Mechanism:** Provides two Triton JIT kernels: (1) layernorm_forward computes: mean = sum(X)/n, variance = sum((X-mean)^2)/n, inv_var = rsqrt(var + eps), then Y = (X - mean) * inv_var * W + b. Stores inv_var and mean for backward pass. All computations in float32 for numerical stability. (2) layernorm_backward follows Karpathy's llm.c approach: computes normed = (X - mean) * inv_var, then dX = inv_var * (dY*W - mean(dY*W) - normed * mean(dY*W*normed)) / n_cols. Fast_Layernorm autograd.Function wraps these with automatic block size and warp count calculation via calculate_settings(). The fast_layernorm() helper extracts weight, bias, and epsilon from LayerNorm modules. Includes test_layernorm() and testing_suite_layernorm() for validation.

**Significance:** Provides LayerNorm optimization for models not using RMS normalization. While most modern LLMs use RMSNorm (no mean subtraction), this supports older architectures and some vision models that use standard LayerNorm.
