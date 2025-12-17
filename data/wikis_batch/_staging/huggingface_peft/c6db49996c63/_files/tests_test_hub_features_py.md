# File: `tests/test_hub_features.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 234 |
| Classes | `PeftHubFeaturesTester`, `TestLocalModel`, `TestBaseModelRevision`, `TestModelCard`, `MyNet` |
| Imports | copy, huggingface_hub, peft, pytest, testing_utils, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for HuggingFace Hub integration features including model cards, revisions, and subfolders.

**Mechanism:** Contains four test classes - PeftHubFeaturesTester validates subfolder loading from the Hub. TestLocalModel ensures local model saving doesn't generate warnings. TestBaseModelRevision tests saving/loading PEFT models with specific base model revisions to ensure version consistency. TestModelCard validates automatic model card generation with correct tags (transformers, base_model, adapter type), pipeline tags based on task type, and proper handling of custom PEFT types.

**Significance:** Ensures seamless integration with the HuggingFace Hub ecosystem. Proper model cards enable discoverability and inference API compatibility, revision support maintains reproducibility across base model versions, and subfolder support allows organized adapter repositories.
