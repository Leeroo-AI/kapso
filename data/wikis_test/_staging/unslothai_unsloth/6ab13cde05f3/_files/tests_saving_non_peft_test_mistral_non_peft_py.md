# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests saving Mistral models without PEFT adapters

**Mechanism:** Validates warning system when attempting to save non-PEFT models

**Significance:** Ensures users are warned when saving models without LoRA adapters
