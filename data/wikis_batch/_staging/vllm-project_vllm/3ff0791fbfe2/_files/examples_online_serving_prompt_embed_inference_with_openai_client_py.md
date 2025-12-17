# File: `examples/online_serving/prompt_embed_inference_with_openai_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 79 |
| Functions | `main` |
| Imports | openai, transformers, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Prompt embeddings inference example

**Mechanism:** Demonstrates passing pre-computed prompt embeddings to vLLM instead of text tokens. Uses HuggingFace Transformers to generate embeddings locally, encodes them in base64, and sends via extra_body parameter. Enables custom prompt preprocessing or working with already-embedded inputs.

**Significance:** Advanced example for scenarios requiring custom embedding manipulation or bypassing tokenization. Useful for research or specialized applications needing direct embedding control.
