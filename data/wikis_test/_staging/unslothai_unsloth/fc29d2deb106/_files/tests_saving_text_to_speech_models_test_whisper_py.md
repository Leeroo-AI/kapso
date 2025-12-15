# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Whisper speech recognition model

**Mechanism:** Validates Whisper-large-v3 ASR: loads model with LoRA (task_type=None for Whisper), fixes generation_config, saves/merges, reloads, downloads test audio, transcribes using pipeline, and validates output contains expected phrases (birch canoe, dark blue background, depth of well, four hours).

**Significance:** Ensures Whisper automatic speech recognition works with Unsloth's fine-tuning and achieves accurate transcription results

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
