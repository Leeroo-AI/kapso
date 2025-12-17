# File: `examples/offline_inference/qwen_1m.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 70 |
| Functions | `load_prompt`, `process_requests`, `initialize_engine`, `main` |
| Imports | os, urllib, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates 1M context length inference with Qwen2.5

**Mechanism:** Loads extremely long prompts (64k-1M tokens) from URL, initializes Qwen2.5-7B-Instruct-1M with max_model_len=1048576, enable_chunked_prefill, and large max_num_batched_tokens. Sets VLLM_ALLOW_LONG_MAX_MODEL_LEN environment variable. Processes prompts with chunked prefill for memory efficiency.

**Significance:** Example demonstrating vLLM's capability to handle million-token context windows with chunked prefill.
