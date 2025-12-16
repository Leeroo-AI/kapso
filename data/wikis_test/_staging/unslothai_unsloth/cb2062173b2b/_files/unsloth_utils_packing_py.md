# File: `unsloth/utils/packing.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 344 |
| Classes | `_TrlPackingWarningFilter` |
| Functions | `mark_allow_overlength`, `configure_sample_packing`, `configure_padding_free`, `enable_sample_packing`, `enable_padding_free_metadata`, `get_packed_info_from_kwargs`, `build_xformers_block_causal_mask`, `build_sdpa_packed_attention_mask`, `... +1 more` |
| Imports | __future__, collections, logging, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements utilities for enabling and managing packed (padding-free) batches across Unsloth, allowing multiple sequences to be concatenated without padding tokens for improved training efficiency.

**Mechanism:** Provides several categories of functionality:
- Configuration functions: `configure_sample_packing()`, `configure_padding_free()` set TRL SFTConfig flags
- Runtime enablement: `enable_sample_packing()`, `enable_padding_free_metadata()` wrap trainer collators to inject sequence length metadata and remove attention masks
- Metadata extraction: `get_packed_info_from_kwargs()` extracts packed sequence info (lengths, cumulative sequence lengths, max length) from batch kwargs
- Mask building: `build_xformers_block_causal_mask()` and `build_sdpa_packed_attention_mask()` create block-diagonal causal masks for packed batches, with optional sliding window support
- Boundary handling: `mask_packed_sequence_boundaries()` masks final tokens of packed sequences to prevent loss computation across sequence boundaries
Includes xFormers mask caching (OrderedDict with LRU eviction) and TRL warning filtering to suppress noisy logs.

**Significance:** Core infrastructure for efficient training with packed sequences, which dramatically reduces wasted computation on padding tokens. Essential for maximizing GPU utilization when training on variable-length sequences. The mask building functions ensure correct causal attention behavior across multiple concatenated sequences, while the trainer integration makes packing transparent to user code.
