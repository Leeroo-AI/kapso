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

**Purpose:** Cohere Command model optimization with Cohere-specific attention patterns.

**Mechanism:** Patches CohereAttention with fast_forward kernels inheriting from Llama base, adapts for Cohere's rotary embedding variants and layernorm positioning.

**Significance:** Enables Cohere models to benefit from Unsloth's attention optimizations while respecting architecture-specific differences.
