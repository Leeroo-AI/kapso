# File: `tests/saving/test_unsloth_save.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 401 |
| Functions | `loaded_model_tokenizer`, `fp16_model_tokenizer`, `model`, `tokenizer`, `temp_save_dir`, `delete_quantization_config`, `test_save_merged_16bit`, `test_save_merged_4bit`, `... +2 more` |
| Imports | importlib, json, os, pytest, shutil, tempfile, unsloth |

## Understanding

**Status:** Explored

**Purpose:** Comprehensive pytest test suite validating Unsloth's model saving functionality across multiple model types, quantization methods, and save formats.

**Mechanism:** The test suite uses pytest fixtures to load models from a list including text models (TinyLlama, Qwen2.5, Phi-4) and vision models (Gemma-3, Llama-3.2-Vision, Qwen2.5-VL) in both regular and 4-bit variants. Each model gets LoRA adapters applied to attention projections. Three main test functions validate: (1) test_save_merged_16bit - saves with merged_16bit, verifies no quantization_config in saved config.json; (2) test_save_merged_4bit - saves with merged_4bit_forced, verifies quantization_config exists and file size is smaller than 16-bit; (3) test_save_torchao - saves with TorchAO Int8DynamicActivationInt8WeightConfig quantization, verifies proper config and file sizes. An additional test_save_and_inference_torchao validates that TorchAO-quantized models can be loaded and run inference correctly.

**Significance:** Core test suite ensuring Unsloth's saving infrastructure works across the diverse range of supported models and quantization formats. This validates the reliability of save_pretrained_merged() and save_pretrained_torchao() methods that users depend on for model deployment.
