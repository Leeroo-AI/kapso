# File: `tests/qlora/test_unsloth_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 211 |
| Functions | `get_unsloth_model_and_tokenizer`, `get_unsloth_peft_model` |
| Imports | datasets, itertools, pathlib, sys, tests, torch, trl, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Test script that validates QLoRA training and model merging using Unsloth's native FastLanguageModel API.

**Mechanism:** Defines helper functions `get_unsloth_model_and_tokenizer()` and `get_unsloth_peft_model()` that wrap Unsloth's `FastLanguageModel.from_pretrained()` and `FastLanguageModel.get_peft_model()`. Loads Llama-3.2-1B-Instruct in 4-bit mode with max sequence length 512, applies LoRA (rank 64) to attention and MLP projection layers, trains for 100 steps with TRL's SFTConfig. After training, uses Unsloth's `save_pretrained_merged()` with `save_method="merged_16bit"` to export, then reloads the merged model for response verification.

**Significance:** Primary test for Unsloth's QLoRA workflow, validating that FastLanguageModel's training and 16-bit merged model export produces correct outputs. This directly tests Unsloth's core value proposition of optimized QLoRA fine-tuning.
