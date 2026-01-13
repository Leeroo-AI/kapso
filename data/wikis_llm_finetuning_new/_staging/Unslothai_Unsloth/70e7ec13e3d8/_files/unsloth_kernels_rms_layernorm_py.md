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

**Purpose:** Implements fast Triton kernels for RMS (Root Mean Square) Layer Normalization used by Llama, Mistral, and most modern LLM architectures.

**Mechanism:** Provides Triton JIT kernels: (1) _rms_layernorm_forward computes: var = sum(X^2)/n, inv_var = rsqrt(var + eps), Y = (X * inv_var) * W. Stores inv_var for backward. (2) _gemma_rms_layernorm_forward is Gemma-specific: same computation but uses W+1 instead of W (Gemma's normalization quirk). (3) _rms_layernorm_backward computes: normed = X * inv_var, dX = inv_var/n * (n * dY*W - normed * sum(dY*W*normed)). Has GEMMA heuristic to use W+1 in backward. Fast_RMS_Layernorm autograd.Function wraps these. Creates drop-in replacement classes Unsloth_LlamaRMSNorm and Unsloth_MllamaTextRMSNorm. Provides patch_rms_layernorm() and unpatch_rms_layernorm() to swap transformers' implementations. Uses @torch.compiler.disable due to torch.compile issues.

**Significance:** Critical optimization for Llama-family models. RMSNorm is simpler than LayerNorm (no mean subtraction) but still benefits greatly from Triton fusion. Called twice per transformer layer, so kernel efficiency directly impacts training throughput.
