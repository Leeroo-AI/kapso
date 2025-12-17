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

**Purpose:** Tests Unsloth's model registry system to ensure all supported model architectures are correctly registered and can be instantiated with proper quantization configurations.

**Mechanism:** Uses pytest to validate model registration entries, checks model names map to correct architectures, verifies quantization type support (4-bit, 8-bit, 16-bit) for each model, queries HuggingFace Hub for model availability, and ensures registry completeness.

**Significance:** Critical infrastructure test ensuring Unsloth's model catalog is correctly configured, preventing runtime errors when users load models and guaranteeing all advertised model architectures are properly supported with their respective quantization options.
