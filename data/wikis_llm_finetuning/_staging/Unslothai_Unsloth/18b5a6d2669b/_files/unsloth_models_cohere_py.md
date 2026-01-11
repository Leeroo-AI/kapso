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

**Purpose:** Implements optimized Cohere Command model support with QK normalization using LayerNorm (not RMSNorm), standard LayerNorm instead of RMS LayerNorm for decoder layers, and adapted attention mechanisms for Cohere's architecture.

**Mechanism:** Similar to Qwen3 but uses LayerNorm instead of RMSNorm throughout. Provides fast_layernorm_inference with mean subtraction and variance normalization. In CohereAttention_fast_forward, applies fast_layernorm_compiled(self.q_norm, Q) and fast_layernorm_compiled(self.k_norm, K) when self.use_qk_norm is enabled. Uses CohereRotaryEmbedding and reuses LLaMA's attention dispatch. Decoder layer uses standard LayerNorm (input_layernorm, post_attention_layernorm) instead of RMS variants. Requires transformers >= 4.42.

**Significance:** Supports Cohere Command models (Command R, Command R+) which are enterprise-focused with strong RAG capabilities. Demonstrates architectural diversity: while most models converge to RMSNorm, Cohere uses standard LayerNorm. The normalization differences are subtle but critical for model compatibility. Shows Unsloth's flexibility in supporting vendor-specific architectures.
