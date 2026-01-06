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

**Purpose:** Audio transcription using Whisper models via OpenAI API

**Mechanism:** Demonstrates /v1/audio/transcriptions endpoint compatibility with OpenAI's Whisper API. Shows both synchronous and asynchronous/streaming transcription. Supports extra_body for vLLM-specific sampling parameters (seed, repetition_penalty, top_p). Uses vLLM's built-in audio assets for testing. Streaming implementation shows incremental transcription output.

**Significance:** Essential example for audio-to-text applications with vLLM. Shows OpenAI Whisper API compatibility, enabling drop-in replacement of OpenAI services. Important for accessibility applications, transcription services, and audio processing pipelines. Demonstrates vLLM's multimodal capabilities extending beyond vision to audio processing.
