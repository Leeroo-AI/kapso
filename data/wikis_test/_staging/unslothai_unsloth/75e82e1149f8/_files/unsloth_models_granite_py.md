# File: `unsloth/models/granite.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 612 |
| Classes | `GraniteRotaryEmbedding`, `FastGraniteModel` |
| Functions | `GraniteAttention_fast_forward`, `GraniteDecoderLayer_fast_forward`, `GraniteAttention_fast_forward_inference`, `GraniteModel_fast_forward_inference`, `patched_init` |
| Imports | _utils, bitsandbytes, llama, math, mistral, os, peft, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Optimized implementation for IBM's Granite model architecture

**Mechanism:** Inherits from `FastLlamaModel` with Granite-specific customizations:
- Granite's unique attention pattern with multiplier scaling
- Custom RoPE implementation
- Support for Granite 3 and Granite 4 (MoE variant) architectures
- Specialized normalization handling

**Significance:** Supports IBM Granite models including both dense and MoE variants. Granite 4 introduces hybrid MoE architecture requiring special handling for layernorm precision.
