# File: `tests/test_trainable_tokens.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 1018 |
| Classes | `ModelEmb`, `ModelEmbedIn`, `ModelEmbedMultiple`, `ModelEmbedInNoGet`, `TestTrainableTokens`, `MultiEmbeddingMLP` |
| Imports | __future__, copy, peft, pytest, safetensors, testing_utils, torch, transformers, unittest |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for TrainableTokens adapter

**Mechanism:** Tests TrainableTokens for making specific token embeddings trainable, including multi-embedding models, token selection, save/load, and integration with other adapters

**Significance:** Test coverage for trainable tokens PEFT method
