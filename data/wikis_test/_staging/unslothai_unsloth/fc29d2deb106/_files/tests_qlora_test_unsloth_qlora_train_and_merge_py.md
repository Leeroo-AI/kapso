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

**Purpose:** Tests Unsloth QLoRA training and merging workflow

**Mechanism:** Uses FastLanguageModel to load Llama-3.2-1B with 4-bit quantization, applies Unsloth's PEFT implementation, trains for 100 steps, then saves and reloads merged 16-bit model using save_pretrained_merged method to validate Unsloth's optimized merge functionality

**Significance:** Validates Unsloth's QLoRA implementation produces functionally equivalent results to HuggingFace baseline while testing the save_pretrained_merged feature that merges LoRA weights back into base model

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
