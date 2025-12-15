# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests model saving functionality across multiple formats

**Mechanism:** Uses pytest fixtures to test saving models in 16-bit, 4-bit, and TorchAO quantized formats. Validates file existence, sizes, config correctness, and reloadability. Tests include merged_16bit (no quantization config), merged_4bit (with quantization config smaller than 16bit), and TorchAO int8 quantization (requires torchao library).

**Significance:** Critical test suite ensuring model persistence works correctly across different quantization methods and can be properly reloaded

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
