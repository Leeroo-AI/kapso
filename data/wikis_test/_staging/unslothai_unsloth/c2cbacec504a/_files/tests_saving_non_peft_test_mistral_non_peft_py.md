# File: `tests/saving/non_peft/test_mistral_non_peft.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 65 |
| Imports | pathlib, peft, sys, tests, transformers, unsloth, warnings |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests saving Mistral models without PEFT adapters to validate Unsloth can correctly save and load full-weight models that haven't been fine-tuned with LoRA.

**Mechanism:** Loads a Mistral model using Unsloth without applying PEFT/LoRA adapters, saves the model to disk using standard HuggingFace format, reloads the saved model, and verifies all weights and configurations are correctly preserved.

**Significance:** Ensures Unsloth's save functionality works for both PEFT and non-PEFT workflows, allowing users to save base models or fully fine-tuned models without adapter layers, maintaining compatibility with standard Transformers infrastructure.
