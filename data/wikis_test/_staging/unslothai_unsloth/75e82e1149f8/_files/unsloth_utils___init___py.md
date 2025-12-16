# File: `unsloth/utils/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 48 |
| Imports | attention_dispatch, packing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Exposes utilities for attention mechanism dispatch and sequence packing through a clean public API.

**Mechanism:** Imports and re-exports functions and constants from two submodules: `packing` (configure_sample_packing, configure_padding_free, enable_sample_packing, enable_padding_free_metadata, mark_allow_overlength) and `attention_dispatch` (AttentionConfig, AttentionContext, FLASH_VARLEN, FLASH_DENSE, XFORMERS, SDPA, run_attention, select_attention_backend). Includes GNU LGPL v3 license header. Defines explicit `__all__` list controlling public exports.

**Significance:** Centralizes access to critical performance utilities used throughout Unsloth. The packing functions enable efficient batch processing with variable-length sequences, while attention dispatch provides abstraction over different attention implementations (FlashAttention, xFormers, SDPA). This module is a key interface point for optimizing training and inference performance.
