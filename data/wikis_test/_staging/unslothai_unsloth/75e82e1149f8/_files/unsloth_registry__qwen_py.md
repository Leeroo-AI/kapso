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

**Purpose:** Registers Alibaba's Qwen model family including text-only (2.5), vision-language (2.5-VL), reasoning (QwQ), and preview vision models (QVQ).

**Mechanism:** Defines four ModelInfo subclasses with distinct naming patterns: `QwenModelInfo` ("Qwen2.5-{size}B"), `QwenVLModelInfo` ("Qwen2.5-VL-{size}B"), `QwenQwQModelInfo` ("QwQ-{size}B"), and `QwenQVQPreviewModelInfo` ("QVQ-{size}B-Preview"). Creates four ModelMeta instances: Qwen 2.5 text-only (3B/7B, with/without Instruct), Qwen 2.5-VL multimodal (3B/7B/32B/72B, Instruct-only), QwQ reasoning (32B with GGUF support), and QVQ Preview multimodal (72B, limited quantization). Uses three singleton flags for registration control.

**Significance:** Comprehensive support for Qwen's diverse model ecosystem including cutting-edge reasoning models (QwQ) and preview vision models (QVQ). The 2.5-VL family spans wide size range (3B to 72B) for multimodal tasks. QwQ's GGUF support enables local deployment of reasoning capabilities. The preview models represent experimental features. Includes Hub verification in `__main__` block.
