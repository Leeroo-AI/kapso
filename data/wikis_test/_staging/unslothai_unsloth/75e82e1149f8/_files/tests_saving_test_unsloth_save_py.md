# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive save/merge functionality testing

**Mechanism:** Tests various save scenarios including 16-bit and 4-bit merged models, GGUF conversion, LoRA adapter saving, and quantization config handling using pytest fixtures

**Significance:** Core test suite ensuring all model saving and merging operations work correctly across different precision levels and formats
