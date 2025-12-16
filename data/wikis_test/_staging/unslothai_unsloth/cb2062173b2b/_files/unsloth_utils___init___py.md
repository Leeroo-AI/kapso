# File: `unsloth/utils/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 48 |
| Imports | attention_dispatch, packing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initialization file that exposes key utility functions for attention backend dispatch and sequence packing from submodules to the main utils namespace.

**Mechanism:** Imports and re-exports functions and classes from two main submodules:
- From `packing`: Functions for configuring and enabling sample packing/padding-free batching (configure_padding_free, configure_sample_packing, enable_padding_free_metadata, enable_sample_packing, mark_allow_overlength)
- From `attention_dispatch`: Attention configuration dataclasses (AttentionConfig, AttentionContext), backend constants (FLASH_DENSE, FLASH_VARLEN, SDPA, XFORMERS), and execution functions (run_attention, select_attention_backend)
All exports are explicitly listed in __all__ for clean API surface.

**Significance:** Serves as the main entry point for Unsloth's utility functionality, consolidating attention kernel dispatch and sequence packing utilities that are critical for efficient training. Makes these utilities easily accessible to other parts of the codebase through a single import point.
