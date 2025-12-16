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

**Purpose:** Optimized implementation for Cohere's Command models

**Mechanism:** Inherits from `FastLlamaModel` with Cohere-specific features:
- LayerNorm instead of RMSNorm
- Cohere's attention architecture variations
- Command R and Command R+ specific optimizations
- Logit scaling adjustments

**Significance:** Supports Cohere Command R family models. Note: Currently temporarily disabled in loader.py until compatibility issues are resolved, but code remains for future activation.
