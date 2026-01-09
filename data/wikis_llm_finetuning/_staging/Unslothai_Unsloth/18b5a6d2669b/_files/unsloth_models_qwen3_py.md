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

**Purpose:** Implements optimized Qwen3 model support with QK normalization (Query-Key Layer Normalization), a key architectural difference from Qwen2 that improves training stability by normalizing query and key vectors before applying RoPE.

**Mechanism:** Inherits from llama.py but adds critical QK-norm step: Q and K tensors are normalized via fast_rms_layernorm(self.q_norm, Q) and fast_rms_layernorm(self.k_norm, K) before RoPE application and attention computation. This normalization happens after projection but before transpose, requiring careful tensor shape handling. Otherwise reuses LLaMA attention dispatch, RoPE embedding, and decoder layer patterns. Requires transformers >= 4.50.3.

**Significance:** Enables Qwen3 series (latest Alibaba Cloud models) which introduced QK-norm for improved stability at larger scales. The normalization addition is subtle but critical for model correctness - omitting it would cause divergence from official Qwen3. Demonstrates architectural evolution: Qwen2 -> Qwen3 added just one normalization step but requires dedicated implementation.
