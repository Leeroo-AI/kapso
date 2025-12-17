# File: `src/transformers/pipelines/image_text_to_text.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 455 |
| Classes | `ReturnType`, `ImageTextToTextPipeline` |
| Imports | base, enum, generation, processing_utils, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements multimodal pipeline for generating text from both images and text inputs. Supports conversational AI models that can process chat histories containing mixed image/text content and generate contextual responses.

**Mechanism:** The ImageTextToTextPipeline class processes inputs in two modes: simple mode accepting image+text pairs, or chat mode accepting conversation histories with mixed modalities. In chat mode, it validates and wraps messages in Chat objects, applies chat templates via the processor, then uses the model's generate() method. The pipeline handles return types (full_text, new_text, or tensors) and supports continue_final_message for prefilling assistant responses. Uses GenerationConfig with max_new_tokens=256 default.

**Significance:** Critical component enabling vision-language models like BLIP, LLaVA, and similar multimodal conversational AI systems. Powers applications requiring visual understanding with natural language generation, such as image captioning with context, visual question answering, and multimodal chatbots. Essential for the growing field of vision-language AI.
