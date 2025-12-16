# File: `unsloth/models/gemma.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 476 |
| Classes | `GemmaFixedRotaryEmbedding`, `GemmaFixedLinearScalingRotaryEmbedding`, `FastGemmaModel` |
| Functions | `fast_geglu_inference`, `GemmaDecoderLayer_fast_forward`, `GemmaModel_fast_forward_inference` |
| Imports | _utils, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized implementation for Google's Gemma model architecture

**Mechanism:** Inherits from `FastLlamaModel` and customizes for Gemma-specific features:
- GeGLU activation function (Gelu-based GLU) instead of SwiGLU
- Custom RoPE implementations with scaling support
- Gemma's pre-normalization scheme
- Optimized attention and decoder layer forwards

**Significance:** Supports Gemma, Gemma 1.1, and CodeGemma models. Gemma uses GeGLU which requires different optimization than Llama's SwiGLU.
