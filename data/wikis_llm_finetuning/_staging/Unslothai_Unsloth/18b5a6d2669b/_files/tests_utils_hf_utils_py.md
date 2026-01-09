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

**Purpose:** HuggingFace integration utilities for test environment. Provides helper functions for setting up models, tokenizers, trainers, and PEFT/LoRA configurations with proper quantization and device management.

**Mechanism:** setup_model() creates models with optional 4-bit quantization (BitsAndBytes NF4), PEFT integration, and autocast support. setup_tokenizer() loads and applies fixup functions (e.g., fix_llama3_tokenizer for padding). generate_responses() handles batched inference with temperature control. get_peft_config() creates LoraConfig with sensible defaults. setup_trainer() wraps SFTTrainer initialization. convert_lora_to_linear() merges LoRA weights by dequantizing 4-bit weights, adding LoRA deltas, and creating new Linear layers. Includes PeftWeightCallback for debugging training.

**Significance:** Centralizes model setup boilerplate for test files, ensuring consistent configuration across tests. The conversion utilities are critical for testing merged model behavior. The quantization handling ensures tests work with memory-constrained environments. Provides building blocks that make test code more concise and maintainable while ensuring proper resource management and error handling.
