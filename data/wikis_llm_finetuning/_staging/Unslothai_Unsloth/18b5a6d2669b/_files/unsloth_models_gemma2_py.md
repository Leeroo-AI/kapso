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

**Purpose:** Implements optimized Gemma 2 model support with critical attention logit soft-capping feature, sliding window attention, and specialized inference paths. Extends Gemma 1 implementation with architectural improvements from Google's second-generation Gemma models.

**Mechanism:** Builds on gemma.py with Gemma 2-specific features: attention logit soft-capping (tanh(logits/cap)*cap to prevent extreme values), sliding window attention support, query pre-attention scalar normalization, and Flash Attention with soft-cap support (requires HAS_FLASH_ATTENTION_SOFTCAPPING). Uses attention_dispatch module to select optimal backend. Implements both training (Gemma2Attention_fast_forward) and inference (Gemma2Model_fast_forward_inference) paths with window size handling.

**Significance:** Essential for Gemma 2 family (2B, 9B, 27B models) which introduced soft-capping as a training stability technique. Requires transformers >= 4.42. Demonstrates Unsloth's ability to support cutting-edge architectural innovations while maintaining performance. The soft-capping implementation is critical for model stability and cannot be omitted.
