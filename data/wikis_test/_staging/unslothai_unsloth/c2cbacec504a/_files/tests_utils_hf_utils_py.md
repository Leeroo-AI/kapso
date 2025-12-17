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

**Purpose:** Provides comprehensive HuggingFace model setup utilities for tests, including tokenizer configuration, model initialization, LoRA setup, and training pipeline configuration with standard patterns.

**Mechanism:** Implements reusable functions for loading models with quantization, configuring PEFT adapters with common parameters, setting up trainers with proper arguments, generating model responses for validation, and managing weight dtype conversions for testing different precision scenarios.

**Significance:** Core test infrastructure that standardizes model setup across all tests, reducing code duplication and ensuring consistent configuration patterns while providing flexible utilities for different testing scenarios with HuggingFace ecosystem components.
