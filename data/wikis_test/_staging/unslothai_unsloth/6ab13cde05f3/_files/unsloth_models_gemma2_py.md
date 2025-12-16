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

**Purpose:** Gemma2 model with logit softcapping

**Mechanism:** Builds on Gemma support with logit softcapping (tanh squashing), sliding window attention, and grouped query attention. Uses flash attention with softcapping support and dynamic RoPE extension.

**Significance:** Adds advanced attention mechanisms including softcapping for stability and sliding window support. Demonstrates how to integrate specialized attention backends with model-specific features.
