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

**Purpose:** Qwen3 model support for latest Alibaba architecture with attention innovations.

**Mechanism:** Patches Qwen3Attention with optimized forward, handles packed sequence masking via get_packed_info_from_kwargs, supports inference mode context management.

**Significance:** Enables cutting-edge Qwen3 models with modern attention patterns and packed sequence support.
