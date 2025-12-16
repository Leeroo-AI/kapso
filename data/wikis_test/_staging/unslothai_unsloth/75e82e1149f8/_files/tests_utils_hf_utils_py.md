# File: `tests/utils/hf_utils.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 291 |
| Classes | `PeftWeightCallback` |
| Functions | `generate_responses`, `sample_responses`, `setup_tokenizer`, `setup_model`, `get_peft_config`, `setup_trainer`, `setup_lora`, `convert_weights_back_to_dtype`, `... +3 more` |
| Imports | bitsandbytes, contextlib, os, peft, torch, transformers, trl, typing |

## Understanding

**Status:** ✅ Explored

**Purpose:** HuggingFace integration utilities

**Mechanism:** Provides model/tokenizer setup, PEFT configuration, response generation, LoRA-to-linear conversion for testing with HuggingFace models

**Significance:** Essential utilities for HuggingFace-based testing and training workflows
