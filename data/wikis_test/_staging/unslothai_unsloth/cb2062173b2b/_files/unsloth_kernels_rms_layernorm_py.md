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

**Purpose:** Provides optimized Triton kernels for RMS (Root Mean Square) LayerNorm, a simpler variant of LayerNorm used in modern LLMs like Llama, Mistral, and Gemma. Supports both standard RMS and Gemma's variant with `(W+1)` scaling.

**Mechanism:** Three Triton kernels: (1) `_rms_layernorm_forward` - computes RMS normalization `Y = (X / sqrt(mean(X^2) + eps)) * W` in single pass, storing inverse RMS for backward; (2) `_gemma_rms_layernorm_forward` - Gemma variant using `(W+1)` scaling and all-float32 computation following Keras exactly; (3) `_rms_layernorm_backward` - unified backward supporting both variants, computes gradients using saved RMS statistics. Provides `Unsloth_LlamaRMSNorm` and `Unsloth_MllamaTextRMSNorm` classes that replace transformers' implementations. `patch_rms_layernorm()` globally replaces transformers' RMSNorm with optimized versions.

**Significance:** RMS LayerNorm is used in virtually all modern open-source LLMs (Llama 2/3, Mistral, Gemma, etc.), applied twice per layer making it extremely performance-critical. RMS normalization is simpler than LayerNorm (no mean centering), enabling more efficient kernels. The Gemma-specific variant ensures exact numerical compatibility with Google's implementation. Patching at the transformers level makes the optimization transparent and universal across all models. Compiler disabled due to torch.compile issues.
