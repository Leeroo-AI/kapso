# File: `tests/saving/text_to_speech_models/test_whisper.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 195 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Unsloth's integration with Whisper models for automatic speech recognition, validating LoRA fine-tuning and model persistence for audio transcription tasks.

**Mechanism:** Loads Whisper encoder-decoder models, applies LoRA adapters to audio encoder and text decoder components, fine-tunes on speech datasets, saves merged models, and validates transcription accuracy on audio files using soundfile for audio I/O.

**Significance:** Validates Unsloth's support for speech-to-text models with cross-attention between audio and text modalities, ensuring the training pipeline correctly handles Whisper's unique architecture for production ASR applications.
