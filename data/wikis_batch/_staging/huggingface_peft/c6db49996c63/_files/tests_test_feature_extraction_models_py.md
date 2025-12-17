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

**Purpose:** Tests for feature extraction model PEFT adapters across all supported configurations.

**Mechanism:** Contains the TestPeftFeatureExtractionModel class that tests BERT, RoBERTa, and DeBERTa models with various PEFT methods. Runs parametrized tests covering adapter attributes, training, saving/loading, merging, device mapping, and special features like input embeds for prompt learning. Includes skip logic for model-specific incompatibilities (e.g., DeBERTa with certain adapters).

**Significance:** Critical validation for PEFT with feature extraction models used in embeddings and representations. Ensures adapter methods work correctly with encoder-only architectures, which are widely used for classification, NER, and other discriminative tasks.
