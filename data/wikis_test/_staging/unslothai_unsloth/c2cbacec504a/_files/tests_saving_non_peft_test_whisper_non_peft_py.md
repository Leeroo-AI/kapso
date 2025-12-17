# File: `tests/saving/non_peft/test_whisper_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests saving Whisper ASR models without PEFT adapters to ensure Unsloth correctly handles non-LoRA speech models across different modalities.

**Mechanism:** Loads a Whisper model using Unsloth without PEFT configuration, saves the complete model with all encoder-decoder weights and audio processing components, reloads from disk, and validates model integrity and audio transcription capabilities.

**Significance:** Validates Unsloth's cross-modal support for audio models beyond text-only language models, ensuring the save/load pipeline works correctly for Whisper's unique encoder-decoder architecture used in speech recognition tasks.
