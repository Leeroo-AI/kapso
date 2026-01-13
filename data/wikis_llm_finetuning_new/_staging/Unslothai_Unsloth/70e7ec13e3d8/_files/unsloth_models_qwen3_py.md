# File: `unsloth/models/qwen3.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 457 |
| Classes | `FastQwen3Model` |
| Functions | `Qwen3Attention_fast_forward`, `Qwen3Attention_fast_forward_inference` |
| Imports | _utils, llama, os, transformers, unsloth_zoo, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Provides optimized implementations for Alibaba's Qwen3 model architecture, which introduces QK normalization as its key architectural difference from Qwen2.

**Mechanism:** Implements `FastQwen3Model` extending `FastLlamaModel` with: (1) `Qwen3Attention_fast_forward` applying RMS LayerNorm to Q and K tensors before RoPE (`Q = fast_rms_layernorm(self.q_norm, Q)` and `K = fast_rms_layernorm(self.k_norm, K)`) - this QKNorm is the primary distinction from Qwen2; (2) `Qwen3Attention_fast_forward_inference` with paged KV cache, applying `fast_rms_layernorm_inference` to Q/K, and support for sliding window attention and grouped query attention; uses `_LlamaModel_fast_forward_inference` wrapper with the custom attention inference function. The attention dispatch system (`AttentionConfig`, `AttentionContext`, `run_attention`) handles backend selection for variable-length sequences. Requires transformers >= 4.50.3.

**Significance:** Core model architecture support for Qwen3 models. QK normalization is an emerging technique for training stability that normalizes query and key representations before attention computation. This file enables Unsloth optimizations while properly handling the QKNorm that distinguishes Qwen3 from its predecessor.
