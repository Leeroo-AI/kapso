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

**Purpose:** RMS layer normalization with model-specific patching (Llama, Gemma variants)

**Mechanism:** Implements Fast_RMS_Layernorm autograd function with two Triton forward kernels: standard (_rms_layernorm_forward: norm=X*rsqrt(mean(X^2)+eps)*W) and Gemma variant (_gemma_rms_layernorm_forward: (W+1)*norm, all float32). Backward kernel handles both via GEMMA flag. Provides Unsloth_LlamaRMSNorm and Unsloth_MllamaTextRMSNorm wrapper classes. patch_rms_layernorm/unpatch_rms_layernorm functions modify transformers modules globally

**Significance:** Critical for Llama family models which use RMS LayerNorm instead of standard LayerNorm. Gemma requires special (W+1) scaling. Marked with torch.compiler.disable due to compilation issues. Provides significant speedup over PyTorch implementation through fused Triton kernels. Essential component of Unsloth's overall speed improvements
