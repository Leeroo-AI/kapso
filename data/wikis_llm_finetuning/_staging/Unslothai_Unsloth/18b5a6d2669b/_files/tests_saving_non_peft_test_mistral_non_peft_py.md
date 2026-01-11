# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests proper handling of save_pretrained_merged when called on non-PEFT models (models without LoRA adapters), ensuring appropriate warnings are raised for Mistral models.

**Mechanism:** Loads mistral-7b-v0.3 in 4-bit without applying LoRA adapters, attempts save_pretrained_merged which should warn about missing PEFT adapters, verifies the warning message matches expected text, then successfully uses standard save_pretrained as the correct alternative.

**Significance:** Ensures Unsloth gracefully handles misuse of merge functionality on base models, providing clear user guidance to use standard save methods when LoRA adapters are not present.
