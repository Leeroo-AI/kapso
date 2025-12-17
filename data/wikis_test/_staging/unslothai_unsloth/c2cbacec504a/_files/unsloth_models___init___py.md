# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** ✅ Explored

**Purpose:** Entry point that exports public API classes and functions from model-specific implementations and training utilities.

**Mechanism:** Imports and re-exports FastLlamaModel, FastLanguageModel, FastVisionModel, FastTextModel, FastModel, architecture-specific adapters (Mistral, Qwen2/3, Granite, Falcon), and training patches (DPO, KTO, RL, vLLM).

**Significance:** Provides clean namespace for users to access Unsloth's complete model loading and optimization framework without needing to know internal module structure.
