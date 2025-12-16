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

**Purpose:** Cohere model with QK normalization

**Mechanism:** Similar to Qwen3 but with distinct QK norm implementation and optional layer normalization caching. Uses paged attention for inference and sliding window support.

**Significance:** Adds alternative QK norm pattern and demonstrates model-specific layer norm handling. Shows architecture variations for similar attention modifications.
