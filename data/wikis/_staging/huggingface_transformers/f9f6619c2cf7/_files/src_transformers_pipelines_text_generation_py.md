# File: `src/transformers/pipelines/text_generation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 500 |
| Classes | `ReturnType`, `TextGenerationPipeline` |
| Imports | base, enum, generation, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements text generation pipeline for autoregressive language models. Supports both simple prompt completion and chat-based conversation with multi-turn dialogue formatting.

**Mechanism:** Tokenizes prompts or applies chat templates to conversation history, generates tokens using model.generate() with configurable parameters (temperature, sampling, max tokens), and decodes output. Handles three return types: full text, new text only, or raw tensors. Special logic for TransformerXL and XLNet with prefix prompts. Supports assistant model for speculative decoding.

**Significance:** Central pipeline for generative AI applications. Powers chatbots, code completion, creative writing tools, and any application requiring open-ended text generation from language models.
