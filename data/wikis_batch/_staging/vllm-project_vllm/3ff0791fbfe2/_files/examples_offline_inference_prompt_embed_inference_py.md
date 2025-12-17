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

**Purpose:** Demonstrates prompt embeddings as input

**Mechanism:** Uses Hugging Face Transformers to generate prompt embeddings from token IDs using model's input embedding layer, then passes embeddings directly to vLLM via prompt_embeds parameter with enable_prompt_embeds=True. Shows both single and batch inference patterns.

**Significance:** Example demonstrating vLLM's ability to accept pre-computed prompt embeddings instead of text/token inputs.
