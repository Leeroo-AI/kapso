# File: `examples/online_serving/openai_translation_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 75 |
| Functions | `sync_openai`, `stream_openai_response`, `main` |
| Imports | asyncio, httpx, json, openai, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Audio translation using Whisper models via OpenAI API

**Mechanism:** Uses /v1/audio/translations endpoint to translate audio to English. Shows synchronous mode with OpenAI client and streaming mode with raw httpx requests (since OpenAI client doesn't support streaming translations). Supports language parameter and vLLM-specific sampling params through extra_body. Uses Italian audio asset for testing translation functionality.

**Significance:** Demonstrates Whisper translation capabilities (convert any language audio to English text). Important for multilingual applications and international accessibility. Shows workaround for OpenAI client limitations using direct HTTP streaming. Complements transcription example to showcase full audio processing capabilities.
