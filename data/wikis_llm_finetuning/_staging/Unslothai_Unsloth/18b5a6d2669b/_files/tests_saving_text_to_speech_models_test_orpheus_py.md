# File: `tests/saving/text_to_speech_models/test_orpheus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 282 |
| Functions | `find_lora_base_model`, `redistribute_codes` |
| Imports | os, pathlib, peft, requests, snac, soundfile, sys, tests, torch, transformers, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Orpheus-3B TTS model with LoRA merging and multi-layer hierarchical audio token generation using SNAC (multi-scale neural audio codec) for decoding.

**Mechanism:** Loads orpheus-3b-0.1-ft with rank-64 LoRA on all attention and MLP modules, saves merged model, reloads for inference, generates audio tokens with special start/end markers, processes 7-token groups into 3 hierarchical SNAC layers (1 coarse + 2 fine layers with offset arithmetic), and decodes to 24kHz audio using SNAC codec.

**Significance:** Validates Unsloth's support for sophisticated hierarchical speech generation models that use multi-resolution audio codecs, ensuring merge operations preserve the model's ability to generate properly structured tokens for multi-layer audio reconstruction.
