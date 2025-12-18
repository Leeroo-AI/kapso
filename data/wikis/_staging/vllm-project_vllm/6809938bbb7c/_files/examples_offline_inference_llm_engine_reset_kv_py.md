# File: `examples/offline_inference/llm_engine_reset_kv.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 98 |
| Functions | `create_test_prompts`, `process_requests`, `initialize_engine`, `parse_args`, `main` |
| Imports | argparse, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates manually resetting KV cache between requests using LLMEngine for memory management.

**Mechanism:** Uses LLMEngine with explicit reset_kv_after_output=True to clear KV cache after each request completion. Shows manual control over cache lifecycle, allowing fresh KV cache state for each prompt without maintaining state between requests.

**Significance:** Illustrates fine-grained KV cache management for scenarios requiring isolation between requests or periodic cache clearing. Useful for understanding memory management and preventing cache pollution in multi-tenant or stateless inference scenarios.
