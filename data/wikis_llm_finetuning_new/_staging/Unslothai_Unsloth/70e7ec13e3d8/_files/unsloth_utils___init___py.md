# File: `unsloth/utils/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 48 |
| Imports | attention_dispatch, packing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initializer that exposes the utils module's public API for attention dispatch and sequence packing functionality.

**Mechanism:** Imports and re-exports key components from two submodules: (1) from `packing`: `configure_padding_free`, `configure_sample_packing`, `enable_padding_free_metadata`, `enable_sample_packing`, `mark_allow_overlength` for configuring and enabling packed/padding-free batching; (2) from `attention_dispatch`: `AttentionConfig`, `AttentionContext`, `FLASH_DENSE`, `FLASH_VARLEN`, `SDPA`, `XFORMERS`, `run_attention`, `select_attention_backend` for attention backend selection and execution. Defines `__all__` to explicitly specify the public interface.

**Significance:** Provides a unified entry point for performance-critical utilities used throughout Unsloth. These utilities handle efficient attention computation and sequence packing, which are fundamental to Unsloth's memory and speed optimizations during training.
