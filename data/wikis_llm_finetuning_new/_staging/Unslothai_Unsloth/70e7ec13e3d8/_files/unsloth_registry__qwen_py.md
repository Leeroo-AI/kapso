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

**Purpose:** Defines model metadata and registration logic for Alibaba's Qwen model family including Qwen 2.5, Qwen 2.5-VL (vision-language), QwQ (reasoning), and QVQ (vision reasoning) models.

**Mechanism:** Provides four `ModelInfo` subclasses for different naming patterns: `QwenModelInfo` (`Qwen{version}-{size}B`), `QwenVLModelInfo` (`Qwen{version}-VL-{size}B`), `QwenQwQModelInfo` (`QwQ-{size}B`), and `QwenQVQPreviewModelInfo` (`QVQ-{size}B-Preview`). Defines `ModelMeta` for each variant with appropriate sizes: Qwen 2.5 (3B, 7B), Qwen 2.5-VL (3B-72B, multimodal), QwQ (32B), and QVQ Preview (72B, multimodal).

**Significance:** Demonstrates the registry's ability to handle a diverse model family with multiple architectures and naming conventions. The Qwen family includes standard LLMs, vision-language models, and specialized reasoning models (QwQ/QVQ), showing the registry's flexibility.
