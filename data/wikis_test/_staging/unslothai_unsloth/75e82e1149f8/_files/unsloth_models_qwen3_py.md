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

**Purpose:** Optimized implementation for Alibaba's Qwen3 model family (also known as QwQ)

**Mechanism:** Extends Qwen2/Llama with Qwen3-specific features:
- YARN (Yet Another RoPE scaling) for extended context
- Qwen3's attention modifications
- QwQ reasoning model support
- Optimized long context handling

**Significance:** Supports Qwen3 and QwQ (Qwen with Questions) models that emphasize reasoning and long-context understanding. Requires transformers >= 4.50.3.
