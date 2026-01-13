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

**Purpose:** Provides utilities for enabling packed (padding-free) batches across Unsloth, allowing multiple sequences to be concatenated into single tensors for memory-efficient training.

**Mechanism:** Implements several key functionalities: (1) Configuration functions (`configure_sample_packing`, `configure_padding_free`) that mutate TRL's `SFTConfig` to enable packing/padding-free modes; (2) Runtime enablement functions (`enable_sample_packing`, `enable_padding_free_metadata`) that wrap the trainer's data collator to inject packed sequence length metadata into batches; (3) `mark_allow_overlength` recursively marks module hierarchies to allow packed batches exceeding `max_seq_length`; (4) `get_packed_info_from_kwargs` extracts and computes packed sequence metadata (lengths, cumulative sequence lengths, max sequence length) for attention kernels; (5) Attention mask builders: `build_xformers_block_causal_mask` creates cached xFormers `BlockDiagonalCausalMask` for efficient variable-length attention, `build_sdpa_packed_attention_mask` constructs block-diagonal masks for PyTorch SDPA; (6) `mask_packed_sequence_boundaries` marks final tokens of packed samples with `ignore_index` for cross-entropy loss to ignore boundary predictions. Includes LRU caching for xFormers masks and a logging filter to suppress TRL packing warnings.

**Significance:** Critical infrastructure for Unsloth's memory efficiency. Sequence packing eliminates padding waste by concatenating multiple sequences, significantly reducing memory usage and improving training throughput, especially for variable-length datasets.
