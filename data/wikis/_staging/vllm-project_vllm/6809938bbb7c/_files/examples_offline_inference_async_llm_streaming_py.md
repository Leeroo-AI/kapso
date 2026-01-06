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

**Purpose:** Demonstrates streaming token-by-token output using AsyncLLM (V1 engine) in offline inference mode.

**Mechanism:** Uses AsyncLLM with DELTA mode (RequestOutputKind.DELTA) to stream new tokens as they are generated. Processes prompts asynchronously and displays tokens incrementally as they arrive, showing the core streaming pattern with async generators.

**Significance:** Showcases vLLM's V1 engine streaming capabilities for offline inference, demonstrating how to receive incremental output rather than waiting for complete generation. Essential for real-time applications requiring progressive text display.
