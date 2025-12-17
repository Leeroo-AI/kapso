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

**Purpose:** Registers Alibaba Qwen model families including Qwen 2.5, Qwen 2.5 VL, QwQ, and QVQ Preview with multimodal support for vision models.

**Mechanism:** Defines four specialized ModelInfo subclasses for different naming schemes; creates four ModelMeta configurations with size-dependent quantization support; handles multimodal flagging for vision variants.

**Significance:** Provides comprehensive integration of Alibaba's diverse Qwen model ecosystem including reasoning models (QwQ) and vision models.
