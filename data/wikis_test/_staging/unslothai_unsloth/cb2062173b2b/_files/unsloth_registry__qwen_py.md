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

**Purpose:** Registers Alibaba's Qwen model family including text, vision-language, and reasoning variants with specialized naming conventions.

**Mechanism:** Defines four ModelInfo classes with distinct naming patterns: (1) `QwenModelInfo` - "Qwen{version}-{size}B" for standard text models, (2) `QwenVLModelInfo` - "Qwen{version}-VL-{size}B" for vision-language models, (3) `QwenQwQModelInfo` - "QwQ-{size}B" for reasoning models, (4) `QwenQVQPreviewModelInfo` - "QVQ-{size}B-Preview" for vision reasoning preview. Creates four ModelMeta instances: Qwen 2.5 text models (3B, 7B with optional Instruct tag), Qwen 2.5-VL vision models (3B/7B/32B/72B, instruct-only), QwQ reasoning (32B with GGUF), and QVQ vision reasoning preview (72B, limited quantization). All from org="Qwen".

**Significance:** Comprehensive support for Qwen's diverse model ecosystem spanning text-only, multimodal vision-language, and specialized reasoning models. The QwQ and QVQ variants represent models designed for enhanced reasoning capabilities. The range of sizes (3B to 72B) and modality support makes Qwen models versatile for different applications from edge to cloud deployment.
