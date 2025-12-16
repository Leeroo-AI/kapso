# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests Whisper speech recognition model training and saving

**Mechanism:** Validates LoRA fine-tuning and saving for Whisper ASR models

**Significance:** Ensures speech recognition model support with correct transcription after training
