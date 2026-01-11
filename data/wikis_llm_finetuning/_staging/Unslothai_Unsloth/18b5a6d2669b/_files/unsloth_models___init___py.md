# File: `unsloth/models/__init__.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 30 |
| Imports | _utils, dpo, granite, llama, loader, mistral, qwen2, qwen3, qwen3_moe, rl |

## Understanding

**Status:** ✅ Explored

**Purpose:** Central package initialization file that exposes the main model classes and utilities for the Unsloth models module. Acts as the public API entry point for loading and working with optimized language models.

**Mechanism:** Imports and exports key components including FastLanguageModel loader classes (FastLlamaModel, FastMistralModel, FastQwen2Model, etc.), trainer patches (PatchDPOTrainer, PatchKTOTrainer, PatchFastRL), and utility functions (is_bfloat16_supported, is_vLLM_available). Includes conditional imports for models that require newer transformers versions (e.g., falcon_h1).

**Significance:** Essential API surface for the entire models package. Provides users with a single import location to access all model-specific implementations and optimization patches. The version is exposed here (__version__) making it the canonical source for Unsloth version information.
