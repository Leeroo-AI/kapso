# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package initializer that provides the public API for registering and searching model configurations in the Unsloth registry.

**Mechanism:** Implements a singleton pattern using `_ARE_MODELS_REGISTERED` flag to ensure models are registered only once. The `register_models()` function calls individual registration functions for each model family (DeepSeek, Gemma, Llama, Mistral, Phi, Qwen). The `search_models()` function provides filtered queries against `MODEL_REGISTRY` by organization, base name, version, size, quantization types, or pattern matching on the full model path.

**Significance:** Core entry point for the registry system. It abstracts away individual model family registrations and provides a unified search interface for discovering available models. This enables users to programmatically find Unsloth-optimized models matching specific criteria.
