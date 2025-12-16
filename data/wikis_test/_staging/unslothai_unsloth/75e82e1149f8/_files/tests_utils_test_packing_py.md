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

**Purpose:** Sequence packing tests

**Mechanism:** Tests padding-free and sample packing configurations, validates position IDs, packed lengths, and boundary masking

**Significance:** Critical for validating memory-efficient sequence packing features
