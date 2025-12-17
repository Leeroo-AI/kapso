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

**Purpose:** Comprehensive utilities for enabling padding-free and sample-packed training batches to improve throughput and reduce memory usage.

**Mechanism:** Implements mask caching for xFormers block-diagonal masks; configuration functions for TRL trainer (configure_sample_packing, configure_padding_free); metadata injection via torch_call wrappers to pass packed sequence lengths; mask builders for xFormers and SDPA backends with sliding window support; boundary masking to ignore cross-sample predictions.

**Significance:** Enables efficient training of variable-length sequences without padding overhead, critical for improving GPU utilization and training speed on long-sequence tasks.
