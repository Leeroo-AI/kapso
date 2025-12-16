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

**Purpose:** Optimized LayerNorm implementation using Triton kernels for faster normalization operations.

**Mechanism:** Implements standard LayerNorm with learnable affine parameters (weight and bias). The forward pass computes mean and variance in float32 for numerical stability, then normalizes: output = (x - mean) / √(var + eps) * weight + bias. Stores inverse variance for backward pass. The backward pass uses the mathematical derivation from llm.c: dX = (dY*W - mean(dY*W) - normed*mean(dY*W*normed)) / √var. All computations are fused in single kernels to minimize memory transfers. Uses Welford's online algorithm for stable variance computation.

**Significance:** LayerNorm is used in every transformer layer (pre-norm or post-norm architecture), making it a critical performance bottleneck. This fused kernel implementation provides substantial speedups by avoiding multiple passes over data and reducing GPU kernel launches. The float32 accumulation ensures numerical stability even with mixed precision training. The inclusion of comprehensive tests validates correctness across different sequence lengths, batch sizes, and data types (FP16/BF16).
