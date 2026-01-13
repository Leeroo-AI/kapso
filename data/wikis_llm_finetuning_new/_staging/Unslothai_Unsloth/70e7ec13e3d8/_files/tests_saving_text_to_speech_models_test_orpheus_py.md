# File: `tests/saving/text_to_speech_models/test_orpheus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 282 |
| Functions | `find_lora_base_model`, `redistribute_codes` |
| Imports | os, pathlib, peft, requests, snac, soundfile, sys, tests, torch, transformers, ... +2 more |

## Understanding

**Status:** Explored

**Purpose:** Tests the fine-tuning, saving, merging, and inference workflow for the Orpheus-3B text-to-speech model using SNAC (Scalable Neural Audio Codec) for audio decoding.

**Mechanism:** The test loads Orpheus-3B-0.1-ft using FastLanguageModel with LoRA adapters (r=64) applied to all attention and MLP projections. After PeftModel validation and saving the merged model, inference uses a complex token scheme with special start/end tokens (128259 for start of human, 128009/128260 for end tokens). Generated tokens are processed through redistribute_codes() which separates them into three layers for SNAC decoding, with offset adjustments (subtracting 4096 multiples per layer). The SNAC model (hubertsiuzdak/snac_24khz) converts these hierarchical codes to audio at 24kHz. The test requires ffmpeg, soundfile, and snac packages.

**Significance:** Validates Unsloth's support for Orpheus-style TTS models that use hierarchical neural audio codecs. The complex token redistribution scheme (7 tokens per audio frame across 3 layers) represents a sophisticated audio generation approach, ensuring Unsloth handles diverse TTS architectures correctly.
