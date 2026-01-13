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

**Purpose:** Provides comprehensive HuggingFace/PEFT/TRL integration utilities for setting up and testing fine-tuning workflows.

**Mechanism:** Implements `PeftWeightCallback` (TrainerCallback for debugging training steps), `generate_responses()`/`sample_responses()` for batched model inference with configurable temperature and autocast, `setup_tokenizer()` with optional fixup functions, `setup_model()` for 4-bit quantization via BitsAndBytes with NF4 quant type and optional PEFT integration, `get_peft_config()` to create LoraConfig objects, `setup_trainer()` to instantiate SFTTrainer, `convert_weights_back_to_dtype()` to restore non-LoRA weight dtypes after kbit preparation, `fix_llama3_tokenizer()` to configure pad tokens, `replace_module()` for recursive module replacement, and `convert_lora_to_linear()` to merge LoRA adapters into dequantized base weights by computing w_dq + lora_B @ lora_A * scaling.

**Significance:** Essential test harness module providing standard patterns for model loading, quantization, LoRA configuration, training setup, and adapter merging that mirrors production fine-tuning workflows.
