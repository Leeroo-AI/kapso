# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Registry package initialization that provides the public API for model registration and search functionality.

**Mechanism:** Imports model-specific registration functions from all model family modules (_deepseek, _gemma, _llama, _mistral, _phi, _qwen), exposes core registry components (MODEL_REGISTRY, ModelInfo, QuantType), and provides two main functions: register_models() which calls all model family registration functions once via a global flag, and search_models() which filters the MODEL_REGISTRY based on criteria like org, base_name, version, size, quant_types, and search patterns.

**Significance:** Central entry point for the registry system that coordinates model registration across all supported model families and provides a unified search interface for discovering available models in Unsloth.
