# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** Explored

**Purpose:** Tests that save_pretrained_merged() correctly warns users when called on a non-PEFT model (without LoRA adapters) and that save_pretrained() works as the proper alternative.

**Mechanism:** The test loads Mistral-7B-v0.3 using FastLanguageModel.from_pretrained() in 4-bit without applying any LoRA adapters. It then calls save_pretrained_merged() and captures warnings using warnings.catch_warnings(). The test asserts that the expected warning message "Model is not a PeftModel (no Lora adapters detected). Skipping Merge. Please use save_pretrained() or push_to_hub() instead!" is raised. In Phase 3, it verifies that the standard save_pretrained() method works without warnings when called on the non-PEFT model.

**Significance:** Ensures proper user guidance when attempting to merge models without LoRA adapters. This prevents confusion by clearly directing users to use the appropriate saving method based on whether their model has adapters applied.
