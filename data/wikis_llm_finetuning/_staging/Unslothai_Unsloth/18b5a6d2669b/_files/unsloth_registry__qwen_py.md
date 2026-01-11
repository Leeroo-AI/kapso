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

**Purpose:** Registers Qwen model family variants including Qwen 2.5 text, Qwen 2.5-VL vision-language, QwQ reasoning, and QVQ vision-reasoning models.

**Mechanism:** Defines four ModelInfo subclasses with distinct naming patterns: QwenModelInfo ("Qwen{version}-{size}B"), QwenVLModelInfo (adds "-VL-"), QwenQwQModelInfo ("QwQ-{size}B"), and QwenQVQPreviewModelInfo ("QVQ-{size}B-Preview"). Creates four ModelMeta instances: Qwen_2_5_Meta (3B/7B text-only with base/Instruct tags, none/bnb/unsloth quants), Qwen_2_5_VLMeta (3B/7B/32B/72B multimodal Instruct-only, same quants), QwenQwQMeta (32B reasoning model with all quant types including gguf), and QwenQVQPreviewMeta (72B multimodal reasoning preview, none/bnb only). Three registration functions coordinate the families: register_qwen_2_5_models(), register_qwen_2_5_vl_models(), register_qwen_qwq_models(), unified by register_qwen_models().

**Significance:** Most diverse model family in the registry spanning standard language models, vision-language models, and specialized reasoning models (QwQ/QVQ), reflecting Qwen's comprehensive ecosystem with different capabilities and modalities while maintaining consistent naming and quantization patterns.
