# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive pytest test suite validating multiple model saving formats (16-bit merged, 4-bit merged, TorchAO quantized) across diverse model architectures including text and vision models.

**Mechanism:** Uses pytest fixtures to parametrize tests across 10 different models (text: TinyLlama, Qwen2.5, Phi-4; vision: Gemma-3, Llama-3.2-Vision, Qwen2.5-VL). Creates temporary directories for each test run. Test functions: (1) test_save_merged_16bit validates 16-bit merged saves produce unquantized models with proper config files, (2) test_save_merged_4bit verifies 4-bit saves are smaller and contain quantization_config, (3) test_save_torchao validates TorchAO quantization saves and loads correctly, (4) test_save_and_inference_torchao performs full inference pipeline on quantized models. Compares file sizes between formats and tests model reloading to ensure saved models are functional.

**Significance:** Core validation suite ensuring the reliability of Unsloth's save functionality across the full spectrum of supported models and quantization formats. Critical for maintaining quality assurance as new models are added. Tests both the save operations and the ability to reload and use saved models, which is essential for production deployment workflows.
