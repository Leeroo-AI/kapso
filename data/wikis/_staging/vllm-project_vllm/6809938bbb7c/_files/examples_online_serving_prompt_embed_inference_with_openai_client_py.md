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

**Purpose:** Demonstrates prompt embeddings inference via OpenAI API

**Mechanism:** Manually generates prompt embeddings using Transformers (tokenizer + embedding layer), encodes them to base64 using vLLM's tensor2base64 utility, and sends them through extra_body parameter to /v1/completions endpoint. Requires --enable-prompt-embeds flag on vLLM server. Allows bypassing tokenization on server side by sending pre-computed embeddings.

**Significance:** Advanced feature for specialized use cases: custom tokenization, embedding manipulation, prefix caching with embeddings, or integration with external embedding systems. Important for research applications and fine-grained control over model inputs. Shows vLLM's flexibility in accepting pre-computed representations rather than just text strings.
