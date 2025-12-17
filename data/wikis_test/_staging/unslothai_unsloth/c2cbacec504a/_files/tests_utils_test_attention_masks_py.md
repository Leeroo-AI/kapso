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

**Purpose:** Tests attention mask handling across different attention implementations (SDPA, xFormers, Flash Attention) to ensure sliding window attention and packed sequences work correctly with Unsloth's optimizations.

**Mechanism:** Creates fake mask and bias objects to simulate attention patterns, tests SDPA and xFormers mask generation with sliding windows, validates attention implementations receive correct window sizes and softcap parameters, and verifies packed sequence boundaries are properly marked.

**Significance:** Critical for validating Unsloth's attention kernel optimizations work correctly with advanced attention patterns like sliding windows and packed sequences, preventing silent correctness issues that could degrade model quality during training.
