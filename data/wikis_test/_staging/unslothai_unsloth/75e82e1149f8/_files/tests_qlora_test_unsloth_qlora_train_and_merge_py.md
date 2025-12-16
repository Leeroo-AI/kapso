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

**Purpose:** Tests Unsloth QLoRA training workflow

**Mechanism:** Validates Unsloth's optimized QLoRA implementation by training a model, merging adapters, and verifying model outputs are consistent across stages (base, adapter, merged)

**Significance:** Core integration test ensuring Unsloth's QLoRA optimization correctly trains models and produces valid merged weights
