# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Public API entry point for the model registry system, coordinating registration of all supported model families and providing search functionality.

**Mechanism:** Exposes two main functions: (1) `register_models()` - lazy one-time initialization that calls registration functions for all model families (DeepSeek, Gemma, Llama, Mistral, Phi, Qwen) using a global `_ARE_MODELS_REGISTERED` flag to prevent duplicate registration, and (2) `search_models()` - query interface that filters the registry by org, base_name, version, size, quant_types, or search_pattern against model paths. The function automatically triggers registration if not already done, then applies sequential filters to MODEL_REGISTRY values.

**Significance:** Central coordination point that makes the registry system usable. Aggregates all model-specific registration modules and provides a clean, unified search interface for discovering available models. Users interact with this module rather than individual model registration files, enabling model discovery without knowing the internal organization structure.
