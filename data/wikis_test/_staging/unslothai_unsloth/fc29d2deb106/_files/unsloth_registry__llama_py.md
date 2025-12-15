# File: `unsloth/registry/_llama.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 125 |
| Classes | `LlamaModelInfo`, `LlamaVisionModelInfo` |
| Functions | `register_llama_3_1_models`, `register_llama_3_2_models`, `register_llama_3_2_vision_models`, `register_llama_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Register Meta Llama 3.x models

**Mechanism:** Defines separate ModelInfo classes for standard (LlamaModelInfo) and vision (LlamaVisionModelInfo) models, creates distinct ModelMeta configs for Llama 3.1, 3.2 base/instruct, and 3.2 vision variants with size-specific quantization support (e.g., 11B vision supports BNB/UNSLOTH, 90B only NONE).

**Significance:** Central registry for Meta's Llama 3 family enabling Unsloth to handle both text-only (1B, 3B, 8B) and multimodal vision models (11B, 90B) with appropriate quantization options per model size.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
