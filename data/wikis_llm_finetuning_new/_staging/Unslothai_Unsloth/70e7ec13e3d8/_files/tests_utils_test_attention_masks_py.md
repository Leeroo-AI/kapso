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

**Purpose:** Unit tests verifying correct attention mask construction for packed sequences with sliding window support across SDPA, xFormers, and Flash Attention backends.

**Mechanism:** Uses `_make_seq_info()` helper to create sequence lengths, cumulative offsets, and max length tensors. `test_sdpa_packed_attention_mask_sliding_window()` verifies the SDPA mask correctly applies -inf to upper triangular (future) tokens and positions outside the sliding window. `test_xformers_block_mask_sliding_window()` uses monkeypatching with a _FakeMask class to verify `make_local_attention()` is called with correct window size. `test_run_attention_sdpa_passes_sliding_window()` and `test_run_attention_xformers_passes_sliding_window()` use monkeypatching to capture arguments passed through AttentionConfig/AttentionContext to verify sliding_window propagation. `test_run_attention_flash_varlen_receives_window_and_softcap()` verifies Flash Attention varlen receives window_size tuple and softcap parameters.

**Significance:** Critical tests ensuring packed attention correctness across all supported attention backends, validating that sliding window and sequence boundary masking work properly for efficient batch training.
