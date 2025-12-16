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

**Purpose:** Provides optimized Triton kernels for LayerNorm normalization with affine transformation (weight and bias). Follows Andrej Karpathy's llm.c implementation and PyTorch's Fp32LayerNorm for numerical stability.

**Mechanism:** Two Triton kernels wrapped in `Fast_Layernorm` autograd function: (1) `layernorm_forward` - computes per-row mean and variance in float32, normalizes `(X - mean) / sqrt(var + eps)`, applies affine transform `Y = normed * W + b`, stores inverse variance for backward; (2) `layernorm_backward` - uses saved statistics to compute gradients `dX = inv_var * (dY*W - mean(dY*W) - normed*mean(dY*W*normed))`. Processes rows in parallel with automatic block size calculation. All intermediate computations in float32 for numerical stability even with FP16/BF16 inputs.

**Significance:** LayerNorm is applied multiple times per transformer layer (typically 2x per layer), making it a performance bottleneck. The custom kernel fuses mean/variance computation and affine transformation into single passes, avoiding multiple memory round-trips. Float32 intermediate computations prevent numerical instabilities that can occur with FP16 normalization. Used primarily for models with traditional LayerNorm rather than RMS LayerNorm (which is more common in recent LLMs).
