# File: `unsloth/utils/attention_dispatch.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 283 |
| Classes | `AttentionConfig`, `AttentionContext` |
| Functions | `select_attention_backend`, `run_attention` |
| Imports | __future__, dataclasses, models, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unified attention backend selection and execution

**Mechanism:** Defines AttentionConfig and AttentionContext dataclasses for attention metadata, implements select_attention_backend() to choose optimal backend (FlashAttention varlen/dense > xFormers > SDPA) based on availability and use case, provides run_attention() that executes attention computation with backend-specific optimizations including GQA expansion and sliding window support.

**Significance:** Critical performance optimization layer that abstracts attention computation across multiple backends (FlashAttention, xFormers, PyTorch SDPA), enabling Unsloth to use the fastest available attention implementation while supporting advanced features like variable-length sequences, grouped-query attention, and sliding window attention.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
