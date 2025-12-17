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

**Purpose:** IBM Granite model optimization handling hybrid transformer-MoE architectures.

**Mechanism:** Patches GraniteAttention, manages Granite-specific linear layer wrapper handling (Bnb/Peft), supports MoE components with specialized initialization.

**Significance:** Enables IBM's Granite models including hybrid architectures with MoE support and proper quantization handling.
