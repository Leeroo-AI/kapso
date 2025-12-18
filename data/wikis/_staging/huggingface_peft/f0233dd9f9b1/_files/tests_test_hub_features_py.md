# File: `tests/test_hub_features.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 546 |
| Classes | `PeftHubFeaturesTester`, `TestLocalModel`, `TestBaseModelRevision` |
| Imports | copy, huggingface_hub, peft, pytest, torch, transformers |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for Hugging Face Hub integration features

**Mechanism:** Tests Hub-specific functionality including subfolder loading, local model saving without warnings, base model revision tracking and loading, and ensuring correct model reconstruction from saved PEFT models

**Significance:** Test coverage for Hub integration and model versioning
