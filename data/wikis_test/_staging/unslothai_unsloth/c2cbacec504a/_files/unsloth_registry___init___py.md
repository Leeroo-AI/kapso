# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Main registry interface that orchestrates model registration and provides model search functionality across all supported model families.

**Mechanism:** Implements lazy registration pattern where models are registered on-demand via register_models(), supporting filters for organization, base name, version, size, quantization type, and regex pattern matching. Uses global flag to prevent duplicate registrations.

**Significance:** Serves as the primary entry point for the registry system, abstracting the complexity of multi-source model metadata management and enabling flexible model discovery.
