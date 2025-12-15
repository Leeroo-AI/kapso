# File: `unsloth/registry/_qwen.py`

**Category:** package

| Property | Value |
|----------|-------|
| Lines | 136 |
| Classes | `QwenModelInfo`, `QwenVLModelInfo`, `QwenQwQModelInfo`, `QwenQVQPreviewModelInfo` |
| Functions | `register_qwen_2_5_models`, `register_qwen_2_5_vl_models`, `register_qwen_qwq_models`, `register_qwen_models` |
| Imports | unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Register Qwen model family variants

**Mechanism:** Defines four ModelInfo classes for different naming schemes (Qwen2.5, Qwen2.5-VL, QwQ, QVQ-Preview), creates ModelMeta configs for text models (3B, 7B), vision-language models (3B-72B), and reasoning models (QwQ-32B, QVQ-72B Preview), registering all variants to MODEL_REGISTRY.

**Significance:** Comprehensive registry for Alibaba's Qwen family supporting standard language models, multimodal vision-language models, and specialized reasoning models (QwQ/QVQ) with appropriate quantization options per model type.

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
