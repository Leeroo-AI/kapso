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

**Purpose:** Pytest suite that validates the model registry system by verifying that all registered models exist on Hugging Face Hub and have correct quantization type metadata.

**Mechanism:** The test module defines registration methods for six model families (llama, qwen, mistral, phi, gemma, deepseek) imported from unsloth.registry submodules. Three test functions: (1) test_model_registration is parametrized to test each model family's registration independently by clearing MODEL_REGISTRY, calling the family-specific register function, and verifying all registered model IDs exist on HF Hub via get_model_info, (2) test_all_model_registration calls the unified register_models() function and validates the complete registry, (3) test_quant_type verifies that models with QuantType.UNSLOTH have the correct quantization tag in their model_path by using search_models with quant_types filter. The _test_model_uploaded helper accumulates missing model IDs for detailed assertion messages.

**Significance:** This test ensures the integrity of Unsloth's model registry, which is foundational for the auto-model loading functionality. It validates that all declared model configurations actually point to existing Hugging Face repositories, preventing runtime failures when users try to load registered models. The quantization type test ensures model paths correctly reflect their quantization format.
