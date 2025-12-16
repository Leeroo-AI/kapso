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

**Purpose:** Provides unified interface for dispatching attention computations across multiple backends (FlashAttention, xFormers, SDPA) with automatic fallback and optimization for packed sequences.

**Mechanism:** `AttentionConfig` dataclass stores per-layer metadata (backend choice, KV heads, groups, backend-specific kwargs). `AttentionContext` dataclass holds per-call information (batch size, sequence lengths, attention masks, sequence packing info). `select_attention_backend()` chooses optimal backend based on availability (FlashAttention > xFormers > SDPA) and whether variable-length packing is used. `run_attention()` executes attention using the selected backend, handling tensor reshaping, GQA expansion, block-diagonal masks for packed sequences, and sliding window attention. Supports FLASH_VARLEN (varlen_func with cu_seqlens), FLASH_DENSE (standard flash_attn_func), XFORMERS (with block causal masks), and SDPA (PyTorch's scaled_dot_product_attention).

**Significance:** Critical performance abstraction that enables Unsloth to leverage the fastest available attention implementation while maintaining code simplicity. The packed sequence support (via seq_info tuple containing lengths, cu_seqlens, max_seqlen) is essential for efficient training with variable-length inputs. GQA (Grouped Query Attention) handling ensures compatibility with modern architectures like Llama 2/3. The automatic backend selection with fallback ensures code works on CPU, consumer GPUs (SDPA), and high-end GPUs (FlashAttention).
