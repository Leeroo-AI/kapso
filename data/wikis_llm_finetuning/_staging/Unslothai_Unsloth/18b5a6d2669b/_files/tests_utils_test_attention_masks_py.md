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

**Purpose:** Unit tests for packed-attention mask generation with sliding window support across multiple attention backends (SDPA, xFormers, Flash Attention).

**Mechanism:** Tests verify that attention mask builders correctly apply causal masking and sliding window constraints for packed sequences. Uses monkeypatching to capture calls and verify parameters are passed correctly through the attention dispatch layer. Tests check mask shapes, -inf values for masked positions, and that sliding_window and softcap parameters flow through to each backend.

**Significance:** Ensures correctness of attention masking in packed training scenarios where multiple sequences are concatenated in a single batch. Sliding window attention is crucial for long-context models, and these tests validate that the masking logic works consistently across different attention implementations (PyTorch SDPA, xFormers, Flash Attention).
