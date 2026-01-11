# File: `unsloth/models/gemma.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 474 |
| Classes | `GemmaFixedRotaryEmbedding`, `GemmaFixedLinearScalingRotaryEmbedding`, `FastGemmaModel` |
| Functions | `fast_geglu_inference`, `GemmaDecoderLayer_fast_forward`, `GemmaModel_fast_forward_inference` |
| Imports | Extends `llama.py` classes and imports various PyTorch and Transformers dependencies |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements optimized Gemma model support with architecture-specific modifications including GEGLU activations, custom RMS LayerNorm with hidden state scaling, and adapted attention mechanisms from the LLaMA base implementation.

**Mechanism:** Inherits from llama.py and applies Gemma-specific patches. Key differences from LLaMA: uses GEGLU activation (gelu(gate) * up) instead of SwiGLU, multiplies hidden states by sqrt(hidden_size) after embedding, uses RMS LayerNorm with special Gemma normalization (adds 1 to weights), and reuses LLaMA's fast attention with xformers/Flash Attention backends. Provides fast_geglu_inference for optimized MLP forward pass.

**Significance:** Enables efficient Gemma model fine-tuning with transformers >= 4.38 requirement. Demonstrates Unsloth's architecture adaptation pattern: inherit LLaMA optimizations, override only model-specific components (MLP activation, normalization). At 474 lines, it's much smaller than llama.py due to component reuse.
