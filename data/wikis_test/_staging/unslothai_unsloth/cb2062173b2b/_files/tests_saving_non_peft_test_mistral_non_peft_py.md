# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validation test ensuring that save_pretrained_merged properly handles non-PEFT models by warning users instead of attempting to merge when no LoRA adapters are present.

**Mechanism:** Loads mistral-7b-v0.3 base model without applying any LoRA adapters. Phase 2 calls save_pretrained_merged and captures warnings, asserting that a specific warning message is raised indicating the model is not a PeftModel and recommending use of save_pretrained() instead. Phase 3 verifies that standard save_pretrained() works correctly without warnings. Uses Python warnings module with warnings.catch_warnings context manager to test both scenarios.

**Significance:** Important edge case test ensuring proper error handling and user guidance. Prevents confusion when users mistakenly call save_pretrained_merged on models without adapters, providing clear guidance to use the correct saving method. Validates defensive programming in the save functionality.
