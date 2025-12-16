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

**Purpose:** Qwen3 with QK normalization

**Mechanism:** Implements Qwen3-specific attention with Q and K layer normalization before rotary embeddings. Uses paged attention caching and custom inference loops with grouped query handling.

**Significance:** Introduces QKNorm pattern found in newer model families. Shows how to add pre-attention normalization without breaking compatibility.
