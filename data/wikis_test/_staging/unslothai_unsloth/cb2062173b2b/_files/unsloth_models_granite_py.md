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

**Purpose:** Provides optimized support for IBM's Granite models with their specific architectural features including residual multipliers and attention dropout.

**Mechanism:**
- Extends `FastLlamaModel` with Granite-specific modifications
- **Key Granite features**:
  - **Residual multipliers**: Uses `torch.add(..., alpha=residual_multiplier)` to scale residual connections (lines 230, 238, 254, 260, 475, 482)
  - **Embedding multiplier**: Scales embeddings by `self.model.embedding_multiplier` (line 431)
  - **Attention dropout**: Properly handles `config.attention_dropout` in training mode (line 100)
  - **Position embeddings**: Requires explicit position embeddings passed through the model (assertion at line 113, 289)
- Custom `GraniteRotaryEmbedding` class that extends `LlamaRotaryEmbedding`
- `patched_init` function (lines 500-511) ensures config is accessible throughout the model by injecting it into `__init__`
- Extensive post-patching in `post_patch` method (lines 547-612):
  - Fixes embedding matrix for torch.compile compatibility
  - Handles lm_head and tied weights properly
  - Downcasts all BnB 4-bit/8-bit modules to correct dtype
  - Downcasts RoPE embeddings to correct precision
- Handles attention with sliding windows (lines 138-149) and proper attention mask construction
- Manual attention computation in inference with custom scaling (`self.scaling` instead of standard 1/sqrt(d))

**Significance:** Essential for IBM's Granite model family which incorporates several architectural refinements (residual/embedding multipliers, attention dropout) aimed at improved training stability and performance. The extensive dtype handling and tied weights management shows the complexity of supporting enterprise-grade models with specific quantization requirements. Represents Unsloth's support for corporate-developed models with unique design choices.
