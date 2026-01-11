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

**Purpose:** Triton layer normalization kernel (standard LayerNorm, not RMS)

**Mechanism:** Implements Fast_Layernorm autograd function with Triton kernels. Forward: computes mean, variance, normalizes (X-mean)/sqrt(var+eps), applies affine transform (weight*norm+bias), stores mean and inv_var for backward. Backward: computes gradients following Karpathy's llm.c formulation. All computations in float32 per PyTorch torchtune convention. Uses calculate_settings for optimal block size and warp count

**Significance:** Faster than PyTorch LayerNorm due to fused operations in Triton. Used for models requiring standard LayerNorm (with mean centering) vs RMS LayerNorm. Includes comprehensive testing suite. Can be patched into transformers via patch_layernorm from unsloth_zoo
