# File: `tests/utils/test_packing.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 391 |
| Classes | `_DummyChild`, `_DummyModel`, `_DummyTrainer`, `_PaddingFreeCollator` |
| Functions | `test_mask_packed_sequence_boundaries_marks_single_row`, `test_mask_packed_sequence_boundaries_across_multiple_rows`, `test_configure_sample_packing`, `test_configure_padding_free`, `test_enable_sample_packing`, `test_enable_sample_packing_trl_collator`, `test_enable_padding_free_metadata`, `test_packing_sdpa` |
| Imports | contextlib, datasets, pytest, torch, trl, types, unittest, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive unit tests for Unsloth's sample packing functionality, verifying sequence boundary masking, configuration, collator wrapping, and end-to-end packed training.

**Mechanism:** Tests `mask_packed_sequence_boundaries()` to verify -100 labels are set at sequence boundaries. Tests `configure_sample_packing()`/`configure_padding_free()` for correct config attribute setting. Uses _DummyModel/_DummyTrainer/_PaddingFreeCollator mock classes to test `enable_sample_packing()` which sets `_unsloth_allow_packed_overlength` flags, wraps collators to aggregate `seq_lengths` into `packed_seq_lengths` tensors, and generates correct position_ids. `_build_packed_training_setup()` creates real FastLanguageModel/SFTTrainer with tiny-random-LlamaForCausalLM for integration testing. `test_packing_sdpa()` performs full forward pass verification with monkeypatched `build_sdpa_packed_attention_mask` and `fast_cross_entropy_loss` to validate mask construction and boundary label masking.

**Significance:** Critical test coverage for Unsloth's core optimization feature that enables training efficiency through sequence packing, ensuring correct attention masking and loss computation at sequence boundaries.
