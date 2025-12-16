# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** ✅ Explored

**Purpose:** Package entry point that exports all model classes and utilities

**Mechanism:** Imports and re-exports key classes from submodules:
- Model implementations: FastLlamaModel, FastMistralModel, FastQwen2Model, FastQwen3Model, FastQwen3MoeModel, FastGraniteModel
- Unified loaders: FastLanguageModel, FastModel, FastVisionModel, FastTextModel
- Training utilities: PatchDPOTrainer, PatchKTOTrainer, PatchFastRL
- Helper functions: is_bfloat16_supported, is_vLLM_available, vLLMSamplingParams
- Version info: __version__

Includes try-except for optional imports like FastFalconH1Model which requires transformers >= 4.53.0

**Significance:** Defines the public API surface of the unsloth.models package. Users import from `unsloth.models` rather than individual modules. This file determines what's accessible and maintains backward compatibility when internal structure changes.
