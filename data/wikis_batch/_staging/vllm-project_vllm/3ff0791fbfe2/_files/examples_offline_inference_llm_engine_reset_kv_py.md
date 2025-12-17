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

**Purpose:** Demonstrates prefix cache resetting with LLMEngine

**Mechanism:** Similar to llm_engine_example but calls engine.reset_prefix_cache(reset_running_requests=True) at step 10 to clear cached KV pairs. Uses long repeated prompts to demonstrate preemption behavior and cache invalidation effects on running requests.

**Significance:** Example showing how to reset prefix cache mid-execution, useful for managing memory or testing cache behavior.
