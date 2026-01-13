# File: `unsloth/models/cohere.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 526 |
| Classes | `FastCohereModel` |
| Functions | `fast_layernorm_inference`, `CohereAttention_fast_forward`, `CohereDecoderLayer_fast_forward`, `CohereAttention_fast_forward_inference`, `CohereModel_fast_forward_inference` |
| Imports | _utils, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Cohere's Command model architecture, featuring optional QK LayerNorm and a parallel attention/MLP residual structure.

**Mechanism:** Implements `FastCohereModel` extending `FastLlamaModel` with: (1) `fast_layernorm_inference` custom function for standard LayerNorm (not RMS) with mean subtraction and variance normalization; (2) `CohereAttention_fast_forward` with optional QK normalization via `fast_layernorm_compiled` when `self.use_qk_norm` is True; (3) `CohereDecoderLayer_fast_forward` using parallel computation where attention and MLP outputs are added together (`hidden_states = residual + hidden_states_attention + hidden_states_mlp`) instead of sequential application; (4) `CohereAttention_fast_forward_inference` with paged KV cache and conditional QK LayerNorm; (5) `CohereModel_fast_forward_inference` using `fast_layernorm_inference` instead of RMS normalization. Requires transformers >= 4.42.3.

**Significance:** Core model architecture support for Cohere Command models. Cohere's architecture differs from Llama-style models with: standard LayerNorm instead of RMSNorm, optional QK normalization for training stability, and parallel attention/MLP structure for better throughput. This file adapts Unsloth optimizations to these Cohere-specific design choices.
