# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central initialization module that provides a unified interface for model registration and search across all supported LLM architectures (DeepSeek, Gemma, Llama, Mistral, Phi, Qwen).

**Mechanism:** Imports registration functions from model-specific modules and exposes two main functions: `register_models()` which orchestrates registration of all model families (with singleton pattern via `_ARE_MODELS_REGISTERED` flag), and `search_models()` which provides flexible filtering of registered models by organization, base name, version, size, quantization types, and search patterns. Acts as a facade pattern over the MODEL_REGISTRY.

**Significance:** This is the primary entry point for the model registry system, ensuring models are registered exactly once and providing a convenient search API. It abstracts away the complexity of individual model family registration and provides a unified interface for model discovery across the entire Unsloth ecosystem.
