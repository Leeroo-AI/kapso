# File: `tests/saving/text_to_speech_models/test_lasa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 220 |
| Functions | `find_lora_base_model`, `ids_to_speech_tokens`, `extract_speech_ids` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Llasa TTS model integration

**Mechanism:** Tests Llasa-1B speech generation: loads model with LoRA, verifies class types, saves/merges, reloads, and generates speech from text by converting text to speech tokens using chat template, extracting speech IDs from model output, and decoding with XCodec2 model to produce audio waveforms.

**Significance:** Validates Llasa text-to-speech model's compatibility with Unsloth pipeline and proper speech token generation/decoding

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
