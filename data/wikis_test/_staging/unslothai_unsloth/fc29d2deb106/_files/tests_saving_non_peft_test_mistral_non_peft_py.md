# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests save_pretrained_merged behavior without PEFT adapters

**Mechanism:** Loads Mistral-7B without applying LoRA adapters, attempts save_pretrained_merged which should warn about non-PeftModel, then validates save_pretrained works normally

**Significance:** Ensures proper error handling when users accidentally call merge functions on base models without LoRA adapters, preventing confusion and guiding users to correct save method

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
