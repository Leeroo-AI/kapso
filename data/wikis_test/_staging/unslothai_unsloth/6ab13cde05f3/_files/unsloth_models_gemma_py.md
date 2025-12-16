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

**Purpose:** Gemma model optimization layer

**Mechanism:** Extends Llama optimizations for Gemma-specific architecture with fixed rotary embeddings and GegLU activation. Implements separate RoPE caching for multi-GPU setups and special layer normalization handling with +1 addition.

**Significance:** Provides Gemma model support with precision-correct rotary embeddings (float32) and inference-specific optimizations for dynamic KV caching. Introduces GemmaFixedRotaryEmbedding to fix precision issues.
