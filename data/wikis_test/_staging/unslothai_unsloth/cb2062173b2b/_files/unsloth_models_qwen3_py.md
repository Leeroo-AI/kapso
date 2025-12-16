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

**Purpose:** Implements fast training and inference for Qwen3 models with specialized QK normalization support that differentiates it from Qwen2.

**Mechanism:**
- Extends `FastLlamaModel` but implements custom attention forward passes due to Qwen3's unique QK normalization layers
- **Key difference from Qwen2**: Applies `fast_rms_layernorm` to both Q and K projections before RoPE application (lines 111-112 in training, 288-289 in inference)
- Uses attention dispatcher system (`AttentionConfig`, `AttentionContext`, `run_attention`) for flexible backend selection between Flash Attention and SDPA
- Implements paged attention for inference with KV cache management including dynamic resizing with `KV_CACHE_INCREMENT`
- Handles packed sequences via `get_packed_info_from_kwargs` for variable-length attention
- Extends RoPE embeddings dynamically to fit sequence lengths in VRAM
- Requires transformers >= 4.50.3 with explicit version checking and helpful error messages
- Training forward uses varlen (variable length) attention when appropriate; inference uses manual attention computation with optimized scalar multiplication

**Significance:** Essential for supporting the latest Qwen3 model family which introduces architectural improvements over Qwen2 (specifically QK normalization). The custom attention implementation ensures these architectural differences are handled efficiently while maintaining Unsloth's performance characteristics. Represents Unsloth's ability to adapt to newer model architectures beyond simple Llama clones.
