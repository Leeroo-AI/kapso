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

**Purpose:** Optimized implementation of Gemma (Google's open language model) with custom RoPE embeddings, GEGLU activation optimization, and inference-specific forward passes.

**Mechanism:** Implements FastGemmaModel extending llama.FastLlamaModel with Gemma-specific modifications. Provides custom GemmaFixedRotaryEmbedding classes that differ from Llama's RoPE implementation to match Gemma's architecture. The fast_geglu_inference function efficiently computes Gemma's GEGLU activation (gate * tanh(gelu(up))) using fused operations. GemmaDecoderLayer_fast_forward handles decoder layers with attention and MLP, optimized for inference with memory reuse. Includes xformers-based attention path using build_sdpa_packed_attention_mask and build_xformers_block_causal_mask for efficient packed sequence handling. Supports both training and inference modes with different optimization strategies.

**Significance:** Enables efficient training and inference for Google's Gemma models (2B, 7B, and instruction-tuned variants) which use slightly different architectural choices than Llama. The GEGLU optimization is critical as this activation is more expensive than standard SwiGLU. The fixed RoPE embeddings respect Gemma's position encoding design. This implementation allows Unsloth users to leverage Google's competitive open-source models with the same performance benefits as Llama.
