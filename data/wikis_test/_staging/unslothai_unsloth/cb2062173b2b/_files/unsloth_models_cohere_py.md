# File: `unsloth/models/cohere.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 528 |
| Classes | `FastCohereModel` |
| Functions | `fast_layernorm_inference`, `CohereAttention_fast_forward`, `CohereDecoderLayer_fast_forward`, `CohereAttention_fast_forward_inference`, `CohereModel_fast_forward_inference` |
| Imports | _utils, llama, math, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements optimized training and inference for Cohere's Command R models with their unique architectural features including QK normalization and parallel attention+MLP processing.

**Mechanism:**
- Extends `FastLlamaModel` but with significant architectural differences accommodated
- **Key Cohere-specific features**:
  - **QK Normalization**: Uses `fast_layernorm_compiled` (training) or `fast_layernorm_inference` for both Q and K projections when `use_qk_norm=True` (lines 115-117, 346-348)
  - **Parallel attention+MLP**: `CohereDecoderLayer_fast_forward` adds both attention and MLP outputs to residual simultaneously (lines 214-216) rather than sequentially
  - Custom `fast_layernorm_inference` implementation (lines 64-72) that explicitly handles variance computation and scaling
- Paged attention system with dynamic KV cache resizing (`KV_CACHE_INCREMENT = 256`)
- Handles sliding window attention when configured (lines 382-387)
- Uses attention dispatcher for backend selection (Flash Attention vs SDPA)
- Full model inference path (`CohereModel_fast_forward_inference`) with multi-device support via `move_to_device`
- Requires transformers >= 4.42.3 with version checking
- Replaces Cohere's rotary embeddings with optimized `LlamaRotaryEmbedding`

**Significance:** Critical for supporting Cohere's commercial-grade models (Command R series) which have unique architectural choices. The parallel attention+MLP design and QK normalization represent modern architectural innovations that Unsloth must handle efficiently. Demonstrates Unsloth's capability to optimize models with significant departures from standard Llama architecture while maintaining performance gains.
