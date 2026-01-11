# File: `tests/saving/non_peft/test_whisper_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests proper handling of save_pretrained_merged on non-PEFT audio models (Whisper), parallel to test_mistral_non_peft but for speech recognition models.

**Mechanism:** Loads whisper-large-v3 using FastModel without LoRA adapters, attempts save_pretrained_merged and verifies warning about non-PeftModel, then validates standard save_pretrained works correctly for non-LoRA models.

**Significance:** Extends non-PEFT save validation to audio models, ensuring consistent behavior across different model modalities (language vs speech) when merge operations are inappropriately attempted on base models.
