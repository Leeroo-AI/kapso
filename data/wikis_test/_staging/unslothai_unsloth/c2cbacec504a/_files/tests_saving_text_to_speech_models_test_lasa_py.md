# File: `tests/saving/text_to_speech_models/test_lasa.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 220 |
| Functions | `find_lora_base_model`, `ids_to_speech_tokens`, `extract_speech_ids` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings, ... +1 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Unsloth's compatibility with LASA (Language-Agnostic Speech Audio) models, validating the training pipeline for multilingual speech synthesis and audio generation.

**Mechanism:** Loads LASA models, applies LoRA fine-tuning, extracts and processes speech token IDs from model outputs, converts tokens to audio waveforms, saves and reloads trained models, and validates audio quality across different languages.

**Significance:** Ensures Unsloth supports multilingual speech models with complex token-to-audio conversion pipelines, validating that the training infrastructure handles language-agnostic audio generation architectures used for cross-lingual speech synthesis.
