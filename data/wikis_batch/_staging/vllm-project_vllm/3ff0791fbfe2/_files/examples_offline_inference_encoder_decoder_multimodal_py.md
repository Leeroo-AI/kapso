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

**Purpose:** Demonstrates encoder-decoder multimodal models with Whisper

**Mechanism:** Shows both implicit prompt format (prompt with multi_modal_data) and explicit encoder/decoder prompt format (separate encoder_prompt dict and decoder_prompt string). Uses AudioAsset for audio input and demonstrates proper prompt structuring for encoder-decoder architectures.

**Significance:** Example demonstrating vLLM support for encoder-decoder multimodal models using explicit/implicit prompt formats.
