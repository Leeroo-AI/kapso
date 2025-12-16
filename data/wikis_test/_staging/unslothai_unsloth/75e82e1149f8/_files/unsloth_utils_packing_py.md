# File: `unsloth/utils/packing.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 344 |
| Classes | `_TrlPackingWarningFilter` |
| Functions | `mark_allow_overlength`, `configure_sample_packing`, `configure_padding_free`, `enable_sample_packing`, `enable_padding_free_metadata`, `get_packed_info_from_kwargs`, `build_xformers_block_causal_mask`, `build_sdpa_packed_attention_mask`, `mask_packed_sequence_boundaries` |
| Imports | __future__, collections, logging, torch, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Enables efficient training with packed (multiple samples per batch) and padding-free (no wasted computation on padding tokens) sequences through trainer configuration, collator wrapping, and attention mask generation.

**Mechanism:** Configuration functions (`configure_sample_packing()`, `configure_padding_free()`) mutate TRL's SFTConfig and install log filters to suppress warnings. Runtime enablement functions (`enable_sample_packing()`, `enable_padding_free_metadata()`) mark models to allow overlength sequences and wrap data collators to inject sequence length metadata into batches. `get_packed_info_from_kwargs()` extracts packed sequence info from batch kwargs and computes cumulative sequence lengths (cu_seqlens) and max sequence length needed by FlashAttention. Mask builders (`build_xformers_block_causal_mask()`, `build_sdpa_packed_attention_mask()`) construct block-diagonal causal masks for packed batches, with LRU cache for xFormers masks (maxsize=32) and optional sliding window support. `mask_packed_sequence_boundaries()` sets boundary tokens to ignore_index to prevent cross-sequence gradients in loss computation.

**Significance:** Core infrastructure for Unsloth's training efficiency gains. Sample packing increases GPU utilization by eliminating padding waste, particularly valuable for datasets with variable-length sequences. The collator wrapping approach allows drop-in integration with HuggingFace trainers without requiring custom trainer classes. The block-diagonal mask construction ensures proper causality and prevents attention leakage between packed samples. The boundary masking prevents spurious gradients from predicting tokens across sample boundaries. Support for both xFormers and SDPA backends ensures broad compatibility. The caching strategy for xFormers masks reduces overhead for repeated batch configurations.
