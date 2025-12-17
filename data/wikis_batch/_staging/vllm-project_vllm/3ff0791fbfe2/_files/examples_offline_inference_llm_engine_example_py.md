# File: `examples/offline_inference/llm_engine_example.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 74 |
| Functions | `create_test_prompts`, `process_requests`, `initialize_engine`, `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates direct LLMEngine API usage

**Mechanism:** Uses LLMEngine.from_engine_args() instead of higher-level LLM class. Manually adds requests via add_request(), continuously calls step() to process batches, and handles RequestOutput objects. Shows different sampling parameters per prompt (temperature, top_k, top_p, n, logprobs, prompt_logprobs).

**Significance:** Example demonstrating low-level LLMEngine interface for fine-grained control over request processing and batching.
