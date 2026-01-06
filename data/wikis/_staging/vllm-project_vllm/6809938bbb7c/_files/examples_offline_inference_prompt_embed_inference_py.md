# File: `examples/offline_inference/prompt_embed_inference.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 97 |
| Functions | `init_tokenizer_and_llm`, `get_prompt_embeds`, `single_prompt_inference`, `batch_prompt_inference`, `main` |
| Imports | torch, transformers, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates using pre-computed prompt embeddings instead of text prompts for inference.

**Mechanism:** Extracts prompt embeddings using the model's embedding layer via tokenizer and model.get_input_embeddings(). Passes embeddings directly to LLM.generate() using prompt_inputs parameter with TokensPrompt containing prompt_embeds. Bypasses tokenization step by providing embeddings directly.

**Significance:** Shows advanced pattern for scenarios requiring custom prompt embeddings, such as embedding manipulation, retrieval-augmented generation with embedding fusion, or bypassing tokenization for specialized inputs.
