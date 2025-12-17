# File: `tests/saving/text_to_speech_models/test_orpheus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 282 |
| Functions | `find_lora_base_model`, `redistribute_codes` |
| Imports | os, pathlib, peft, requests, snac, soundfile, sys, tests, torch, transformers, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Unsloth's support for Orpheus speech models, which use SNAC (Structured Neural Audio Codec) for high-quality audio generation with advanced code redistribution.

**Mechanism:** Loads Orpheus models with SNAC codec integration, applies LoRA training, redistributes audio codes for optimal quality, generates speech outputs, validates code redistribution logic, and ensures trained models save/load correctly with codec components.

**Significance:** Validates Unsloth's compatibility with state-of-the-art neural audio codec models, ensuring the training pipeline correctly handles complex audio encoding schemes and code manipulation required for high-fidelity speech synthesis.
