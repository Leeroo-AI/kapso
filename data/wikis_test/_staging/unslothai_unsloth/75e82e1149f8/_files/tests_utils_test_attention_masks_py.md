# File: `tests/utils/test_attention_masks.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 272 |
| Classes | `_FakeMask`, `_FakeBias` |
| Functions | `test_sdpa_packed_attention_mask_sliding_window`, `test_xformers_block_mask_sliding_window`, `test_run_attention_sdpa_passes_sliding_window`, `test_run_attention_xformers_passes_sliding_window`, `test_run_attention_flash_varlen_receives_window_and_softcap` |
| Imports | math, torch, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Attention mask testing

**Mechanism:** Unit tests for packed attention masks with sliding window support across SDPA, XFormers, and Flash Attention backends

**Significance:** Ensures attention mechanisms work correctly with sequence packing
