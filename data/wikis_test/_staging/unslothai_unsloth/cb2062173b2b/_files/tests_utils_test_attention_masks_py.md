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

**Purpose:** Unit tests for packed attention mask generation with sliding window support across different attention backends (SDPA, xFormers, Flash Attention). Ensures that sequence packing and sliding window attention work correctly together.

**Mechanism:** The tests verify attention mask behavior through multiple backends:
- `test_sdpa_packed_attention_mask_sliding_window()` checks that SDPA masks properly mask future tokens and enforce sliding window constraints (e.g., position 4 cannot attend to position 1 with window=3)
- `test_xformers_block_mask_sliding_window()` uses monkeypatch to verify xFormers mask construction with `make_local_attention()` for window constraints
- `test_run_attention_sdpa_passes_sliding_window()` ensures sliding window parameters propagate through the attention dispatch system by capturing mask builder calls
- `test_run_attention_xformers_passes_sliding_window()` similarly tests xFormers attention bias propagation
- `test_run_attention_flash_varlen_receives_window_and_softcap()` verifies Flash Attention varlen receives both window_size tuples and softcap parameters

The tests use mock objects and monkeypatch to intercept and verify internal function calls without requiring actual attention computation.

**Significance:** Critical for ensuring correctness of unsloth's packed sequence training with sliding window attention. These tests verify that optimization paths don't break the complex interaction between sequence packing (multiple sequences in one batch) and sliding window attention (limiting attention span).
