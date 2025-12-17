# File: `src/transformers/pipelines/text_generation.py`

**Category:** other

| Property | Value |
|----------|-------|
| Lines | 500 |
| Classes | `ReturnType`, `TextGenerationPipeline` |
| Imports | base, enum, generation, typing, utils |

## Understanding

**Status:** ✅ Explored

**Purpose:** Implements text generation pipeline for autoregressive language models that continue prompts with generated text. Supports both simple text completion and conversational chat modes with models like GPT-2, GPT-J, and LLaMA.

**Mechanism:** The TextGenerationPipeline uses model.generate() with customizable parameters (max_new_tokens=256, do_sample=True, temperature=0.7 defaults). Handles two input modes: simple text prompts or Chat objects for conversational models with chat templates. Preprocess() tokenizes inputs with special XL_PREFIX for Transformer-XL/XLNet models, handles truncation and padding. Supports continue_final_message for assistant response prefilling, stop_sequence for early termination, and handle_long_generation="hole" for truncation strategies. Postprocess() returns full_text, new_text, or tensors based on return_type.

**Significance:** Flagship component powering modern generative AI applications. Essential for chatbots, creative writing assistants, code generation, content creation, automated responses, and conversational AI systems. Represents the interface to large language models that have revolutionized natural language generation, enabling applications from customer service bots to advanced AI assistants. Most prominent pipeline type in the current AI landscape.
