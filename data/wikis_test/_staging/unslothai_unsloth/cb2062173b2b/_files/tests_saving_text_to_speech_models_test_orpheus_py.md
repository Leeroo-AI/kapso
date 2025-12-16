# File: `tests/saving/text_to_speech_models/test_orpheus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 282 |
| Functions | `find_lora_base_model`, `redistribute_codes` |
| Imports | os, pathlib, peft, requests, snac, soundfile, sys, tests, torch, transformers, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test validating the Orpheus-3B TTS model's complete workflow using SNAC codec, including multi-layer hierarchical speech token generation and decoding to audio.

**Mechanism:** Loads orpheus-3b-0.1-ft with LoRA adapters across all attention and MLP layers. Validates model type preservation. Saves and reloads merged model. Complex inference pipeline: creates custom input format with special tokens (128259=start, 128009/128260=end tokens), applies padding and attention masks for batch processing. Generates up to 1200 tokens with temperature 0.6 and repetition penalty 1.1. Post-processing crops to last occurrence of token 128257, removes token 128258, offsets tokens by -128266. redistribute_codes function implements SNAC's hierarchical structure by splitting the flat token stream into 3 layers with specific offsets (layer 2 and 3 tokens have cumulative 4096 offsets). SNAC model decodes 3-layer codes into 24kHz audio. Saves to WAV file and verifies file existence.

**Significance:** Most complex TTS test demonstrating Unsloth's capability to handle sophisticated multi-layer codec architectures. Validates preservation of intricate token generation patterns through model merging. Critical for advanced speech synthesis research using hierarchical representations. Shows Unsloth can maintain functionality for models requiring extensive post-processing and external codec integration.
