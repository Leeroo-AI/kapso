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

**Purpose:** Padding-free and packed sequence batch utilities

**Mechanism:** Provides configuration functions (configure_sample_packing, configure_padding_free) that modify SFTConfig, runtime enablement (enable_sample_packing, enable_padding_free_metadata) that wraps trainer collators to inject sequence length metadata, builds attention masks for xFormers (build_xformers_block_causal_mask with LRU cache) and SDPA (build_sdpa_packed_attention_mask), includes mark_allow_overlength() to permit sequences exceeding max_seq_length in packed batches.

**Significance:** Core optimization that eliminates padding overhead during training by packing multiple sequences into single batches, significantly improving GPU utilization and training speed. Handles complex attention mask construction and integrates seamlessly with TRL training infrastructure while supporting sliding window attention.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
