# File: `unsloth/registry/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 78 |
| Functions | `register_models`, `search_models` |
| Imports | _deepseek, _gemma, _llama, _mistral, _phi, _qwen, registry |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central registration and search interface for supported model families

**Mechanism:** Provides a singleton pattern registration system that lazily loads model definitions from individual model-family modules (_deepseek, _gemma, _llama, etc.) and exposes a search_models() function for querying registered models by organization, base name, version, size, and quantization type.

**Significance:** Core API that enables Unsloth to discover and validate available models across multiple LLM families. The registry system allows Unsloth to provide model-aware functionality without hardcoding model paths, supporting flexible model discovery and validation.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
