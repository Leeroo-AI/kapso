# File: `unsloth/models/granite.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 610 |
| Classes | `GraniteRotaryEmbedding`, `FastGraniteModel` |
| Functions | `GraniteAttention_fast_forward`, `GraniteDecoderLayer_fast_forward`, `GraniteAttention_fast_forward_inference`, `GraniteModel_fast_forward_inference`, `patched_init` |
| Imports | _utils, bitsandbytes, llama, math, mistral, os, peft, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for IBM's Granite model architecture, which uses residual multipliers and tied embedding weights as distinctive features.

**Mechanism:** Implements `FastGraniteModel` extending `FastLlamaModel` with: (1) `GraniteAttention_fast_forward` using the attention dispatch system with configurable dropout and softmax scaling; (2) `GraniteDecoderLayer_fast_forward` applying `residual_multiplier` parameter to scale residual connections (via `torch.add(residual, hidden_states, alpha=residual_multiplier)`); (3) `GraniteAttention_fast_forward_inference` with paged KV cache and grouped query attention support; (4) `GraniteModel_fast_forward_inference` applying `embedding_multiplier` to input embeddings. Includes `GraniteRotaryEmbedding` extending `LlamaRotaryEmbedding` and a `patched_init` wrapper to pass config through to decoder layers. The `post_patch` method handles Granite's tied weights (lm_head equals embed_tokens) and fixes BnB dtype issues.

**Significance:** Core model architecture support for IBM Granite models (requires transformers >= 4.45.0). Granite models have unique scaling multipliers for residuals and embeddings that require special handling. This file enables Unsloth optimizations while preserving Granite's architectural characteristics including proper tied weight handling.
