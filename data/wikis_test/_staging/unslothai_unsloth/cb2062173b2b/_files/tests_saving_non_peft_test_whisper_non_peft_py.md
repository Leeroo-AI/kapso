# File: `tests/saving/non_peft/test_whisper_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validation test for Whisper speech recognition models ensuring save_pretrained_merged properly handles non-PEFT models by warning users when no LoRA adapters are present.

**Mechanism:** Loads whisper-large-v3 using FastModel.from_pretrained with WhisperForConditionalGeneration auto_model parameter, without applying LoRA adapters. Phase 2 attempts save_pretrained_merged and verifies the expected warning message is raised about the model not being a PeftModel. Phase 3 confirms standard save_pretrained() works correctly. Uses warnings context manager to capture and validate warning behavior.

**Significance:** Extends the non-PEFT save validation to audio models, ensuring consistent behavior across different model architectures (text and speech). Validates that Unsloth's save utilities handle Whisper models appropriately and provide clear guidance when users attempt to merge models without adapters. Important for audio processing workflows.
