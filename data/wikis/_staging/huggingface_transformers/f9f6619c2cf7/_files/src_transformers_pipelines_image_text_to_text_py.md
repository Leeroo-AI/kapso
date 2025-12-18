# File: `src/transformers/pipelines/image_text_to_text.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 455 |
| Classes | `ReturnType`, `ImageTextToTextPipeline` |
| Imports | base, enum, generation, processing_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements a multimodal pipeline for generating text from combined image and text inputs, supporting both simple prompts and conversational chat formats with vision-language models.

**Mechanism:** The `ImageTextToTextPipeline` handles complex multimodal inputs by supporting multiple input formats: simple image-text pairs, chat-style conversations with interleaved images and text, and assistant message prefilling. It uses processors with chat template support to format inputs appropriately, calls the model's `generate()` method with configurable generation parameters, and postprocesses outputs to optionally include or exclude input text and maintain chat conversation structure. The `ReturnType` enum controls whether to return tensors, new text only, or full conversation history.

**Significance:** This is the primary pipeline for vision-language models (like BLIP, LLaVA, Qwen-VL) that need to understand both visual and textual context to generate responses. It bridges the gap between pure image-to-text captioning and advanced multimodal dialogue systems, enabling applications like visual question answering, image-grounded conversations, and conditional image description with sophisticated chat-based interactions.
