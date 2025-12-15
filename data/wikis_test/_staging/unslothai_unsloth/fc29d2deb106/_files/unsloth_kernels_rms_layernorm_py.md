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

**Purpose:** Fast RMS LayerNorm for modern LLMs

**Mechanism:** Triton kernel implementing RMS normalization: y = x / sqrt(mean(x^2) + eps) * weight. Provides two variants: standard (used by Llama) and Gemma variant (y = x * rsqrt(...) * (weight + 1)). Forward pass computes RMS and stores inverse for reuse. Backward efficiently computes gradients using chain rule: dx = (dy * w - normed * sum(dy * w * normed) / n) * inv_var / n. Replaces transformers' LlamaRMSNorm and MllamaTextRMSNorm classes via patching.

**Significance:** RMS LayerNorm is the standard in modern LLMs (Llama, Mistral, Gemma) as it's simpler and faster than LayerNorm (no mean subtraction). This is one of the most frequently called operations during training/inference. The kernel provides 2-3x speedup over naive implementations and is essential for Unsloth's overall performance gains.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
