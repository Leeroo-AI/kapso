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

**Purpose:** Comprehensive test suite for sequence packing functionality, which combines multiple training samples into a single batch to maximize GPU utilization. Tests both padding-free and sample packing configurations.

**Mechanism:** The tests cover multiple aspects of packing:
- `test_mask_packed_sequence_boundaries_*` verify that labels are masked (-100) at sequence boundaries to prevent cross-contamination during loss calculation
- `test_configure_sample_packing()` and `test_configure_padding_free()` check that configuration objects are properly set up
- `test_enable_sample_packing()` tests the core packing wrapper that aggregates `seq_lengths` into `packed_seq_lengths` and generates proper position IDs (resetting to 0 at each sequence boundary)
- `test_enable_sample_packing_trl_collator()` validates integration with TRL's data collator
- `test_enable_padding_free_metadata()` tests the simpler padding-free mode without full packing
- `test_packing_sdpa()` is the most comprehensive, verifying end-to-end packing with SDPA attention including proper mask construction, position ID generation, and boundary masking in loss computation

The helper `_build_packed_training_setup()` creates realistic training scenarios with tiny LLaMA models.

**Significance:** Essential for validating unsloth's signature sequence packing optimization, which significantly improves training efficiency by eliminating wasted computation on padding tokens. These tests ensure correctness of the complex data collation, masking, and attention handling required for packed training.
