# File: `tests/qlora/test_unsloth_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 211 |
| Functions | `get_unsloth_model_and_tokenizer`, `get_unsloth_peft_model` |
| Imports | datasets, itertools, pathlib, sys, tests, torch, trl, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests QLoRA training and merging using Unsloth's FastLanguageModel optimizations.

**Mechanism:**
- Uses `FastLanguageModel.from_pretrained()` to load Llama-3.2-1B-Instruct in 4-bit with Unsloth optimizations
- Applies LoRA via `FastLanguageModel.get_peft_model()` with rank 64 targeting transformer layers
- Trains for 100 steps on synthetic dataset with SFTTrainer
- Validates responses before/after training to check learning
- Uses `save_pretrained_merged()` with "merged_16bit" method to merge LoRA weights into base model
- Reloads merged model and validates it produces correct responses

**Significance:** Core test for Unsloth's QLoRA workflow. Validates the complete training pipeline from loading quantized models, applying LoRA, training, and merging weights back into full precision. Tests Unsloth's custom merge implementation which is critical for deployment.
