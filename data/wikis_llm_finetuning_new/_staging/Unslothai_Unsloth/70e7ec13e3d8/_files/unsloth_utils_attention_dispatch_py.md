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

**Purpose:** Provides a unified attention dispatch system that automatically selects and runs the optimal attention backend based on available libraries and input characteristics.

**Mechanism:** Defines two dataclasses: `AttentionConfig` (per-layer attention metadata including backend choice, head counts, and backend-specific kwargs) and `AttentionContext` (per-call information like batch size, sequence lengths, attention masks, and packed sequence info). The `select_attention_backend()` function returns the best available backend in priority order: FlashAttention varlen (for packed sequences with `seq_info`), FlashAttention dense, xFormers, or PyTorch SDPA as fallback. The `run_attention()` function executes attention using the selected backend, handling tensor reshaping, GQA (grouped query attention) expansion when `n_groups != 1`, and backend-specific operations. For SDPA fallback, it builds appropriate attention masks via helper functions from the packing module. Constants `FLASH_VARLEN`, `FLASH_DENSE`, `XFORMERS`, and `SDPA` identify the backend types.

**Significance:** Core infrastructure enabling Unsloth's performance optimizations. By abstracting attention backend selection, it ensures models always use the fastest available implementation while properly handling packed/variable-length sequences for memory-efficient training.
