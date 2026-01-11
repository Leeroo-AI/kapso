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

**Purpose:** Unit tests for Unsloth's model registry system. Validates that model registration functions correctly populate the registry with valid, accessible HuggingFace models and that quantization type tags are properly assigned.

**Mechanism:** Uses pytest to parametrize tests across different model families (Llama, Qwen, Mistral, Phi, Gemma, DeepSeek). For each family, tests individual registration methods and verifies all registered model IDs exist on HuggingFace Hub using get_model_info(). Also tests the complete register_models() function and validates that quantization types (UNSLOTH, etc.) are correctly assigned with proper tags in model paths.

**Significance:** Critical for ensuring the model registry infrastructure works correctly. The registry is the foundation for Unsloth's model discovery and loading system. These tests prevent broken model references from entering the registry and ensure users can reliably load supported models. Catches issues early when adding new models or modifying registration logic.
