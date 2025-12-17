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

**Purpose:** Audio translation API client

**Mechanism:** Demonstrates audio translation (speech-to-English-text) using Whisper models. Shows both sync and streaming translation. Uses httpx for streaming since OpenAI client doesn't support streaming translations.

**Significance:** Example for translation applications where audio in any language is converted to English text. Complements the transcription example.
