# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** ✅ Explored

**Purpose:** Model package exports and aggregation

**Mechanism:** Imports and re-exports all model classes (FastLlamaModel, FastMistralModel, FastQwen2Model, Qwen3, Granite, etc.), DPO/KTO trainers, utility functions (is_bfloat16_supported, is_vLLM_available, __version__), and RL components. Provides a clean public API for accessing model types and training patches.

**Significance:** Provides a single import point for all model loading classes and utilities. Enables users to access different model architectures through a consistent interface (e.g., FastLlamaModel.from_pretrained()). The version and compatibility utilities are central to ensuring users have the right environment.
