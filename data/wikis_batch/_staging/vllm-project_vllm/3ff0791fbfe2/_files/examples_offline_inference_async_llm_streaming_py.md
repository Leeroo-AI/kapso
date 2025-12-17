# File: `examples/offline_inference/async_llm_streaming.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 111 |
| Functions | `stream_response`, `main` |
| Imports | asyncio, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates streaming offline inference using AsyncLLM (V1 engine) with token-by-token output in DELTA mode.

**Mechanism:** Creates an AsyncLLM engine with AsyncEngineArgs, configures SamplingParams with RequestOutputKind.DELTA for incremental token streaming, and uses async generators to iterate over streaming outputs. The stream_response function demonstrates the core streaming pattern by calling engine.generate() and printing new tokens as they arrive until the finished flag is set.

**Significance:** Example demonstrating vLLM's V1 engine async streaming capabilities for offline inference scenarios, serving as a reference for implementing streaming text generation.
