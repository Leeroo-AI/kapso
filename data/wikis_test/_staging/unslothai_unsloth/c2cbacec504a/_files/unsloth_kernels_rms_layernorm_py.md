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

**Purpose:** Implements RMS LayerNorm (LLaMA, Gemma style) with fused Triton kernels and special Gemma variant that adds 1.0 to weight scaling.

**Mechanism:** _rms_layernorm_forward computes: output = X*rsqrt(mean(X^2)+eps)*W in single pass. Backward pass (_rms_layernorm_backward) computes dX using chain rule with normed intermediate. Gemma variant (_gemma_rms_layernorm_forward) applies output = X*rsqrt(mean(X^2)+eps)*(W+1.0), and backward scales dY_W by (W+1.0). Handles both standard RMS and Gemma's variant through conditional kernel selection.

**Significance:** Critical for LLaMA/Gemma model efficiency, providing 1.5-2x speedup. Gemma variant support essential for accurate training of those models without numerical divergence.
