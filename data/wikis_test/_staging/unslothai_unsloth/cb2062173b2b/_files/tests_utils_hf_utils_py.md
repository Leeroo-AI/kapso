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

**Purpose:** Provides helper utilities for testing HuggingFace models with quantization, LoRA adapters, and training setups. This file contains reusable functions for model initialization, response generation, and PEFT (Parameter-Efficient Fine-Tuning) configuration.

**Mechanism:** The file offers several categories of utilities:
- Model setup functions (`setup_model`, `setup_tokenizer`) that configure HuggingFace models with optional 4-bit quantization via BitsAndBytes
- PEFT/LoRA configuration helpers (`get_peft_config`, `setup_trainer`, `setup_lora`) for efficient fine-tuning
- Response generation utilities (`generate_responses`, `sample_responses`) for inference with batch support and autocast
- Weight conversion utilities (`convert_weights_back_to_dtype`, `convert_lora_to_linear`) for handling quantized weights and merging LoRA adapters
- The `PeftWeightCallback` class for monitoring training progress with debug logging
- Tokenizer fixup functions like `fix_llama3_tokenizer` for handling model-specific quirks

**Significance:** Critical testing infrastructure that standardizes model setup across test cases. These utilities simplify writing tests by providing a consistent interface for common operations like loading quantized models, applying LoRA adapters, and generating responses. The weight conversion functions enable testing of model merging and quantization workflows.
