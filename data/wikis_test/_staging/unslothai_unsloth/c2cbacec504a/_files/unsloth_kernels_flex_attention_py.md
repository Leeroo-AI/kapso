# File: `unsloth/kernels/flex_attention.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 187 |
| Functions | `slow_inference_attention_softcapping` |
| Imports | functools, os, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides PyTorch 2.5+ flex attention implementations with logit softcapping support for Gemma models, with fallback to compiled softmax attention.

**Mechanism:** Attempts to use torch.nn.attention.flex_attention when available (PyTorch 2.5+), which natively supports score modifiers like tanh softcapping via score_mod callbacks. Falls back to torch.compile'd standard attention with manual softcapping for older PyTorch versions. Handles grouped query attention by expanding K,V tensors and supports sliding window masking through functools.lru_cache for efficient mask creation.

**Significance:** Provides state-of-the-art attention implementation leveraging PyTorch's optimizations, enabling faster training on modern hardware while maintaining compatibility with older versions.
