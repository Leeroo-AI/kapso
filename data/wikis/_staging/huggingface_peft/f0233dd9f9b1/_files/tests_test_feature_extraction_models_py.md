# File: `tests/test_feature_extraction_models.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 379 |
| Classes | `TestPeftFeatureExtractionModel` |
| Functions | `skip_non_prompt_learning`, `skip_deberta_lora_tests`, `skip_deberta_pt_tests` |
| Imports | peft, pytest, testing_common, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for feature extraction models

**Mechanism:** Tests PEFT adapters on feature extraction models (BERT, RoBERTa, DeBERTa, DeBERTaV2) for FEATURE_EXTRACTION tasks, with special handling for DeBERTa-specific issues

**Significance:** Test coverage for encoder-only models used in feature extraction
