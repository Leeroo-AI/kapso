# File: `tests/saving/text_to_speech_models/test_csm.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 168 |
| Functions | `find_lora_base_model` |
| Imports | pathlib, peft, requests, soundfile, sys, tests, torch, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests CSM text-to-speech model integration

**Mechanism:** Validates complete CSM-1B model workflow: loads model with LoRA adapters, verifies PeftModel instance and config, saves/merges model, reloads for inference, and generates audio from text input using the model's generate method with depth decoder parameters. Validates audio output file creation.

**Significance:** Ensures CSM speech synthesis models work properly with Unsloth's training and saving pipeline, validating the text-to-speech capability end-to-end

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
