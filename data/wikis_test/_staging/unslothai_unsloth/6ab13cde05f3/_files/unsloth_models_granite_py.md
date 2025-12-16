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

**Purpose:** Granite model with residual scaling

**Mechanism:** Implements Granite architecture with residual connection multipliers and embedding scaling. Uses pre-computed position embeddings and tied weight handling for efficient inference and training.

**Significance:** Introduces residual multiplier pattern for stability in deeper models. Custom post-patching handles quantization dtype corrections and embedding reconstruction.
