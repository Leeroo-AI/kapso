# File: `examples/online_serving/token_generation_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 49 |
| Functions | `main` |
| Imports | httpx, transformers |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Direct token ID generation via /inference/v1/generate endpoint

**Mechanism:** Pre-tokenizes input messages using Transformers tokenizer with chat template, sends token_ids array directly to vLLM's inference endpoint, receives token_ids in response (detokenize=False), and decodes them client-side. Bypasses server-side tokenization entirely. Uses httpx for direct HTTP requests instead of OpenAI client.

**Significance:** Advanced use case for maximum control over tokenization. Useful for custom chat templates, token-level manipulation, research on tokenization effects, or integration with external tokenization services. Shows vLLM's /inference/v1/generate endpoint which accepts pre-tokenized inputs. Important for scenarios where client-side tokenization is required or beneficial for performance/customization.
