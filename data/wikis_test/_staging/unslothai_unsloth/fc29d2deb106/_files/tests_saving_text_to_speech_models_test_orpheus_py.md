# File: `tests/saving/text_to_speech_models/test_orpheus.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 282 |
| Functions | `find_lora_base_model`, `redistribute_codes` |
| Imports | os, pathlib, peft, requests, snac, soundfile, sys, tests, torch, transformers, ... +2 more |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests Orpheus TTS model workflow

**Mechanism:** Validates Orpheus-3B speech model: loads with LoRA adapters, verifies model types, saves/merges, reloads, generates speech by producing audio token codes, redistributes codes across 3 SNAC codec layers, and decodes to 24kHz audio waveform using SNAC model for high-quality speech synthesis.

**Significance:** Ensures Orpheus speech generation model integrates properly with hierarchical audio codec (SNAC) for multi-layer token generation

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
