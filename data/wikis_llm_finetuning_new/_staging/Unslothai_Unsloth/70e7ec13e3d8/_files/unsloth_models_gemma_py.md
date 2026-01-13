# File: `unsloth/models/gemma.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 474 |
| Classes | `GemmaFixedRotaryEmbedding`, `GemmaFixedLinearScalingRotaryEmbedding`, `FastGemmaModel` |
| Functions | `fast_geglu_inference`, `GemmaDecoderLayer_fast_forward`, `GemmaModel_fast_forward_inference` |
| Imports | _utils, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Google's Gemma model architecture, enabling faster training and inference through custom attention and decoder layer implementations.

**Mechanism:** Implements `FastGemmaModel` which extends `FastLlamaModel` and provides: (1) `GemmaFixedRotaryEmbedding` and `GemmaFixedLinearScalingRotaryEmbedding` classes for optimized rotary position embeddings with multi-GPU caching; (2) `fast_geglu_inference` for accelerated GeGLU activation in MLP layers; (3) `GemmaDecoderLayer_fast_forward` which uses fast RMS LayerNorm with Gemma-specific +1 weight offset; (4) `GemmaModel_fast_forward_inference` for optimized inference with paged attention support. The `pre_patch` method monkey-patches HuggingFace's Gemma attention classes, decoder layers, and model forward methods. The `post_patch` method handles Gemma's unique RMS normalization (with +1 weight offset) and ensures proper LoRA parameter freezing.

**Significance:** Core model architecture support file. Gemma models require special handling due to their unique RoPE formulation (division vs multiplication) and +1 weight offset in RMS normalization. This file enables Unsloth's performance optimizations for Gemma model family while maintaining compatibility with HuggingFace transformers.
