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

**Purpose:** RMS LayerNorm for modern transformers

**Mechanism:** Implements RMS (root-mean-square) normalization instead of standard LayerNorm. Faster than LayerNorm as it only requires variance computation without mean. Supports Gemma-style RMS normalization with +1 offset to weights. Both standard and Gemma variants with forward/backward Triton kernels. Patches LlamaRMSNorm from Hugging Face transformers library.

**Significance:** Widely used in modern models (LLaMA, Gemma, Mistral, etc.). More efficient than standard LayerNorm while providing numerical benefits. Automatic patching allows seamless speedup without code changes when Unsloth is imported.
