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

**Purpose:** Optimized implementation for Google's Gemma 2 model architecture with attention softcapping

**Mechanism:** Extends Gemma implementation with Gemma 2-specific features:
- Attention logit soft-capping for stability
- Local/global attention alternating pattern
- Sliding window attention support
- Flash Attention 2.6.3+ integration for efficient softcapping
- Post-normalization instead of pre-normalization

**Significance:** Gemma 2 introduced attention softcapping which requires special kernel support in Flash Attention. Unsloth detects and utilizes these optimizations when available.
