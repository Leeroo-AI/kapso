# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive test for Whisper speech recognition model loading, fine-tuning, saving, and inference.

**Mechanism:** 
- Demonstrates loading a Whisper model with FastModel
- Applies LoRA adapters to the model
- Saves and merges the model
- Performs speech-to-text inference on a downloaded audio sample
- Validates transcription accuracy by checking for expected phrases

**Significance:** Critical integration test for Unsloth's FastModel capabilities with text-to-speech models, showcasing model manipulation, fine-tuning, and inference workflows.
