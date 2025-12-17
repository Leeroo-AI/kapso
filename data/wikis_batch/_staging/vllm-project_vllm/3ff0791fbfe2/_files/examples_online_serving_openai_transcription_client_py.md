# File: `examples/online_serving/openai_transcription_client.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 97 |
| Functions | `sync_openai`, `stream_openai_response`, `main` |
| Imports | asyncio, openai, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Audio transcription API client

**Mechanism:** Demonstrates vLLM's audio transcription endpoint using Whisper models. Shows both synchronous and streaming transcription of audio files via OpenAI-compatible API. Supports additional sampling parameters through extra_body.

**Significance:** Example for speech-to-text applications using vLLM. Shows how to use Whisper-like models for audio transcription with streaming support.
