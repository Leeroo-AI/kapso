# File: `unsloth/models/gemma2.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 656 |
| Classes | `FastGemma2Model` |
| Functions | `Gemma2Attention_fast_forward`, `Gemma2DecoderLayer_fast_forward`, `Gemma2Attention_fast_forward_inference`, `Gemma2Model_fast_forward_inference` |
| Imports | _utils, gemma, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Gemma 2 model optimization with attention softcapping and specialized kernels.

**Mechanism:** Patches Gemma2Attention with logit softcapping support, extends RoPE dynamically, uses flash-attn 2.6.3+ for softcapping kernels, manages local/global attention mechanisms.

**Significance:** Provides Gemma 2-specific optimizations including softcapping that improves numerical stability during attention computation.
