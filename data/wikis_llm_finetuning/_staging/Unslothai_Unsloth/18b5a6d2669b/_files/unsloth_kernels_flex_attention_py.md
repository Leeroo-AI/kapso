# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Flex attention with softcapping (PyTorch 2.5+ feature) and fallback implementation

**Mechanism:** Conditionally imports PyTorch 2.5+ flex_attention API. If available: uses compiled flex attention with custom score_mod for tanh softcapping and block masks for causal/sliding window patterns. If unavailable: provides slow_attention_softcapping fallback with torch.compile that manually implements grouped query attention, softcapping (t*tanh(A/t)), and masking. Also provides slow_inference_attention_softcapping without torch.compile

**Significance:** Enables efficient attention with logit softcapping (required for Gemma 2) while maintaining compatibility across PyTorch versions. Flex attention is significantly faster when available. Critical for models requiring attention modifications beyond standard scaled dot-product
