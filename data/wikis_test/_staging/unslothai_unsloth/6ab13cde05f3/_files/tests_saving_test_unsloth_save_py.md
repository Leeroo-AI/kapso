# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** ✅ Documented

**Purpose:** Comprehensive test suite for model save methods

**Mechanism:** Parametrized pytest tests covering all save formats (merged 16bit, 4bit, GGUF, LoRA)

**Significance:** Validates all model saving pathways ensuring correctness across formats
