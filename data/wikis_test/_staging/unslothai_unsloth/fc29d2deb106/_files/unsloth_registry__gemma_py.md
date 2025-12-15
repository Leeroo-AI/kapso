# File: `unsloth/registry/_gemma.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 74 |
| Classes | `GemmaModelInfo` |
| Functions | `register_gemma_3_base_models`, `register_gemma_3_instruct_models`, `register_gemma_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Register Google Gemma 3 models

**Mechanism:** Defines GemmaModelInfo class for Gemma-specific naming (gemma-3-{size}B format), creates separate ModelMeta configs for base (pt tag) and instruct (it tag) variants supporting 1B, 4B, 12B, and 27B sizes with multimodal capability, then registers both to MODEL_REGISTRY.

**Significance:** Enables Unsloth support for Google's Gemma 3 family, treating them as multimodal models with both pretrained and instruction-tuned variants available in NONE, BNB, UNSLOTH, and GGUF quantization formats.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
