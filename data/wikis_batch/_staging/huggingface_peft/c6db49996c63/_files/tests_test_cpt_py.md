# File: `tests/test_cpt.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 305 |
| Classes | `CPTDataCollatorForLanguageModeling`, `CPTDataset` |
| Functions | `global_tokenizer`, `config_text`, `config_random`, `sst_data`, `collator`, `dataset`, `test_model_initialization_text`, `test_model_initialization_random`, `... +3 more` |
| Imports | datasets, peft, pytest, torch, tqdm, transformers, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests for CPT (Constrained Prompt Tuning) implementation.

**Mechanism:** Provides fixtures for tokenizer, CPT configs (text-based and random), SST2 dataset preparation, and custom data collator. Tests cover: model initialization with text-based/random token selection, task type validation (CAUSAL_LM only), training with constraint verification (embedding remains frozen, delta embeddings constrained by epsilon), and handling of different token type masks (input template, input text, label template, label text).

**Significance:** Validates the CPT method which constrains prompt token updates within a specified epsilon bound, ensuring embeddings remain frozen while learned projections stay within bounds, particularly for label tokens vs non-label tokens.
