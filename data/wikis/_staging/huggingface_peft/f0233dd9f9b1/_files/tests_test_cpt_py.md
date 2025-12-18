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

**Purpose:** Tests for CPT (Constrained Prompt Tuning) adapter

**Mechanism:** Tests CPT model initialization with text and random tokens, training with weighted loss and projection constraints, custom data collator functionality, and integration with SST2 sentiment dataset

**Significance:** Test coverage for CPT adapter and its unique loss weighting mechanisms
