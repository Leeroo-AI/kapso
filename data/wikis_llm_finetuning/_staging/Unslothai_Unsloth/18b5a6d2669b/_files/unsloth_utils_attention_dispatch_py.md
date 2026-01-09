# File: `unsloth/utils/attention_dispatch.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 274 |
| Classes | `AttentionConfig`, `AttentionContext` |
| Functions | `select_attention_backend`, `run_attention` |
| Imports | __future__, dataclasses, models, torch, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Unified attention backend selection and execution for transformer models

**Mechanism:** Provides AttentionConfig and AttentionContext dataclasses to store layer/call-specific parameters, select_attention_backend() to choose optimal backend (FlashAttention varlen/dense > xFormers > SDPA), and run_attention() to execute attention with backend-specific optimizations including GQA handling and packed sequence support

**Significance:** Critical performance layer that abstracts attention implementations across hardware and library availability, enabling Unsloth to automatically use the fastest available attention kernel (Flash, xFormers, or PyTorch SDPA) with proper support for variable-length packed sequences and grouped-query attention
