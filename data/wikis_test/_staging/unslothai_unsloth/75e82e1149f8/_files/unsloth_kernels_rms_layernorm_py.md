# File: `unsloth/kernels/rms_layernorm.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 335 |
| Classes | `Fast_RMS_Layernorm`, `Unsloth_LlamaRMSNorm`, `Unsloth_MllamaTextRMSNorm` |
| Functions | `fast_rms_layernorm`, `patch_rms_layernorm`, `unpatch_rms_layernorm`, `test_rms_layernorm`, `testing_suite_layernorm` |
| Imports | torch, transformers, triton, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized RMS (Root Mean Square) LayerNorm implementation used in LLaMA and similar models.

**Mechanism:** Implements RMSNorm which simplifies LayerNorm by removing mean centering: output = x / √(mean(x²) + eps) * weight. Provides two variants: standard (output = normed * W) and Gemma-specific (output = normed * (W + 1), all in float32). Forward pass computes RMS in float32, stores inverse std for backward. Backward pass computes dX = inv_std/n * (n*dY*W - normed*sum(dY*W*normed)). Includes patching functions to replace transformers' LlamaRMSNorm with optimized version. Disabled torch.compile due to compatibility issues.

**Significance:** RMSNorm is faster and more stable than LayerNorm while maintaining similar performance, making it the normalization of choice for modern LLMs (LLaMA, Mistral, Gemma). This optimized implementation is critical since normalization occurs twice per transformer layer (input and post-attention). The Gemma variant handles the (W+1) parameterization efficiently. The patching mechanism allows seamless integration with existing transformers models without code changes.
