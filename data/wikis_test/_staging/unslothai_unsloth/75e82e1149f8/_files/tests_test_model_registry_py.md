# File: `tests/test_model_registry.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 92 |
| Classes | `ModelTestParam` |
| Functions | `test_model_registration`, `test_all_model_registration`, `test_quant_type` |
| Imports | dataclasses, huggingface_hub, pytest, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests model registry functionality

**Mechanism:** Validates that model registration works correctly for all model families (Llama, Qwen, Mistral, Phi, Gemma, DeepSeek) by checking HuggingFace Hub availability and quantization type tags

**Significance:** Ensures the model registry correctly catalogs and can locate all supported models on HuggingFace Hub, critical for model loading functionality
