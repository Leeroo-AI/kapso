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

**Purpose:** Google Gemma model optimization with GEGLU activation and fixed RoPE handling.

**Mechanism:** Patches GemmaAttention with fast_forward, implements fast_geglu_inference for gate-up-proj fusion, handles fixed rotary embeddings without dynamic extension, manages attention softcapping.

**Significance:** Extends framework to Gemma models with specialized fusion kernels for GEGLU MLPs that differ from Llama's SwiGLU.
