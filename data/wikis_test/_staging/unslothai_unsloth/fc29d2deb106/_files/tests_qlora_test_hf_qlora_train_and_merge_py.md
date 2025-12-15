# File: `tests/qlora/test_hf_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 159 |
| Imports | copy, datasets, itertools, pathlib, sys, tests, torch, trl |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests HuggingFace QLoRA training and merging workflow

**Mechanism:** Loads Llama-3.2-1B model with 4-bit quantization, applies PEFT LoRA adapters, trains on synthetic dataset for 100 steps, tests model responses before/after training, then merges LoRA weights using both custom convert_lora_to_linear function and PEFT's merge_and_unload method to validate equivalence

**Significance:** Validates baseline HuggingFace QLoRA implementation works correctly before comparing with Unsloth's optimized version, ensuring merge operations produce expected results

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
