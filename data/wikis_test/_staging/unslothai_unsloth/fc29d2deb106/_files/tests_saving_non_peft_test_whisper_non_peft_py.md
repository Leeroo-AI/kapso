# File: `tests/saving/non_peft/test_whisper_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests save_pretrained_merged behavior for Whisper without PEFT

**Mechanism:** Loads Whisper-large-v3 audio model using FastModel without LoRA adapters, attempts save_pretrained_merged which should warn, then validates save_pretrained works

**Significance:** Validates non-PEFT error handling extends to audio models (not just language models), ensuring consistent behavior across all model types Unsloth supports

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
