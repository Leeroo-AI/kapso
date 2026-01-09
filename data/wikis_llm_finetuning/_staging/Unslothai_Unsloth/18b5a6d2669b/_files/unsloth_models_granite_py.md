# File: `unsloth/models/granite.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 610 |
| Classes | `GraniteRotaryEmbedding`, `FastGraniteModel` |
| Functions | `GraniteAttention_fast_forward`, `GraniteDecoderLayer_fast_forward`, `GraniteAttention_fast_forward_inference`, `GraniteModel_fast_forward_inference`, `patched_init` |
| Imports | _utils, bitsandbytes, llama, math, mistral, os, peft, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements optimized IBM Granite model support with custom attention configurations, attention dropout support, and specific RoPE embedding handling required for Granite's enterprise-focused architecture.

**Mechanism:** Inherits from llama.py and mistral.py patterns. Key differences: requires position_embeddings parameter (asserts not None), supports attention_dropout during training, uses custom scaling factor (self.scaling) in attention config, and handles attention_mask differently (uses SDPA backend when mask present). GraniteAttention_fast_forward configures AttentionConfig with dropout_p, scale parameters, and larger window sizes (kv_seq_len, kv_seq_len). Imports both Linear4bit types (Bnb_Linear4bit, Peft_Linear4bit) suggesting quantization focus.

**Significance:** Enables IBM Granite model series (code, chat, instruct variants) aimed at enterprise applications. Granite models emphasize code generation and business use cases. Requires transformers >= 4.45. The attention dropout and scaling customization suggest Granite prioritizes training flexibility over pure inference speed. At 610 lines, it's moderately complex due to enterprise requirements.
