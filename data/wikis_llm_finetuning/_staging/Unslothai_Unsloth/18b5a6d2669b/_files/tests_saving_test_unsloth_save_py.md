# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive pytest suite testing multiple save formats (merged_16bit, merged_4bit, torchao) across various model types (text and vision) with validation of file structure, quantization configs, and reload capability.

**Mechanism:** Uses pytest fixtures to parameterize tests across 10 different models, tests three save methods: merged_16bit (full precision merged), merged_4bit_forced (quantized merged), and torchao (TorchAO quantization), validates config files, weight sizes, and successful model reloading with inference verification for TorchAO models.

**Significance:** Core test suite ensuring Unsloth's save functionality works correctly across all supported model architectures and quantization formats, with proper file generation, size reduction validation, and reload compatibility.
