# File: `unsloth/models/gemma2.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 654 |
| Classes | `FastGemma2Model` |
| Functions | `Gemma2Attention_fast_forward`, `Gemma2DecoderLayer_fast_forward`, `Gemma2Attention_fast_forward_inference`, `Gemma2Model_fast_forward_inference` |
| Imports | _utils, gemma, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Google's Gemma 2 model architecture, which introduces attention logit softcapping and sliding window attention compared to the original Gemma.

**Mechanism:** Implements `FastGemma2Model` extending `FastLlamaModel` with: (1) `Gemma2Attention_fast_forward` supporting attention logit softcapping (multiplying by 1/t, applying tanh, then multiplying by t) and sliding window attention via flash attention when available; (2) `Gemma2DecoderLayer_fast_forward` with four layer norms (input, post-attention, pre-feedforward, post-feedforward) as Gemma2 uses a unique architecture; (3) `Gemma2Attention_fast_forward_inference` with paged KV cache management that dynamically grows in 256-token increments and handles sliding window masking; (4) `Gemma2Model_fast_forward_inference` alternating between sliding window attention (even layers) and global attention (odd layers). Uses `AttentionConfig` and `AttentionContext` from attention dispatch system to select optimal backend (Flash Attention with softcapping, SDPA, or xformers).

**Significance:** Core model architecture support for Gemma 2 models. Essential because Gemma 2 introduces significant architectural changes from Gemma 1: attention logit softcapping for training stability, alternating sliding/global window attention, and additional layer norms. This file adapts Unsloth's optimization framework to these Gemma 2-specific features.
