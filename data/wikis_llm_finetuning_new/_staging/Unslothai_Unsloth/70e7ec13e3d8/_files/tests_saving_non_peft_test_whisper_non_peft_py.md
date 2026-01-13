# File: `tests/saving/non_peft/test_whisper_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** Explored

**Purpose:** Tests that save_pretrained_merged() correctly warns users when called on a Whisper speech model without LoRA adapters and validates save_pretrained() as the proper alternative.

**Mechanism:** The test loads Whisper-large-v3 using FastModel.from_pretrained() with WhisperForConditionalGeneration as the auto_model class, configured for English transcription. Without applying LoRA adapters, it calls save_pretrained_merged() and verifies the expected warning "Model is not a PeftModel (no Lora adapters detected). Skipping Merge. Please use save_pretrained() or push_to_hub() instead!" is raised. Phase 3 confirms that save_pretrained() works correctly without warnings on the non-PEFT Whisper model.

**Significance:** Validates that Unsloth's non-language models (speech recognition) properly handle the non-PEFT saving scenario. This ensures consistent behavior and error messages across different model types supported by FastModel.
