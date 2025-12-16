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

**Purpose:** Pytest-based validation suite for model registry functionality, ensuring all registered models exist on Hugging Face Hub and have correct quantization type metadata.

**Mechanism:** Defines parameterized tests using pytest fixtures with ModelTestParam dataclass containing model family names and registration functions. test_model_registration is parameterized across six model families (llama, qwen, mistral, phi, gemma, deepseek), clearing MODEL_REGISTRY before each test, calling family-specific registration method (e.g., register_llama_models), then validating each registered model ID exists on Hub using get_model_info. test_all_model_registration validates register_models() registers all families correctly. test_quant_type validates that models under "unsloth" org with QuantType.UNSLOTH have proper quant_tag in model_path using search_models query. Uses _test_model_uploaded helper to collect missing models.

**Significance:** Critical infrastructure test ensuring model registry integrity. Prevents broken model references in production by validating Hub availability of all registered models. The quantization type validation ensures metadata consistency for model search/filtering functionality. Parameterized design allows easy addition of new model families while maintaining comprehensive coverage.
