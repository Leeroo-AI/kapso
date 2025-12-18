# File: `examples/offline_inference/encoder_decoder_multimodal.py`

**Category:** example

| Property | Value |
|----------|-------|
| Lines | 133 |
| Classes | `ModelRequestData` |
| Functions | `run_whisper`, `parse_args`, `main` |
| Imports | collections, dataclasses, os, time, typing, vllm |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Demonstrates encoder-decoder models with multimodal inputs, specifically Whisper for audio transcription.

**Mechanism:** Uses vLLM's encoder-decoder support with Whisper models (whisper-medium, whisper-large-v3, distil-whisper). Processes audio inputs through the encoder and generates text transcriptions via the decoder. Shows proper configuration of multimodal data with audio inputs and decoder-specific prompt formatting.

**Significance:** Illustrates vLLM's support for encoder-decoder architectures beyond decoder-only models. Critical reference for speech-to-text and other encoder-decoder multimodal tasks using vLLM.
