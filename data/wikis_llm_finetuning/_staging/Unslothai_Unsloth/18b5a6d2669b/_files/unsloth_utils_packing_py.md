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

**Purpose:** Enable padding-free training with sample packing to maximize GPU memory utilization

**Mechanism:** Provides configuration functions (configure_sample_packing, configure_padding_free, enable_sample_packing) to set up TRL trainers, builds attention masks for packed sequences (build_xformers_block_causal_mask, build_sdpa_packed_attention_mask), manages sequence length metadata extraction from batches, includes LRU cache for xFormers block masks, and masks sequence boundaries in labels to prevent cross-sample prediction

**Significance:** Critical efficiency component that eliminates wasted computation on padding tokens by packing multiple training samples into single sequences, significantly improving training throughput especially for datasets with varying sequence lengths. Integrates with multiple attention backends (Flash, xFormers, SDPA) and TRL training infrastructure
