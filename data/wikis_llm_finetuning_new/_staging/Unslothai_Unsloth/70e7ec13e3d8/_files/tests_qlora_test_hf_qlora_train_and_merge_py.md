# File: `tests/qlora/test_hf_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 159 |
| Imports | copy, datasets, itertools, pathlib, sys, tests, torch, trl |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Test script that validates QLoRA training and model merging using standard HuggingFace/PEFT methods (not Unsloth-specific APIs).

**Mechanism:** The script loads Llama-3.2-1B-Instruct in 4-bit quantized mode, applies LoRA adapters via PEFT with rank 64, trains for 100 steps using TRL's SFTConfig, then tests two merging approaches: (1) a custom `convert_lora_to_linear` function that merges LoRA weights into base model, and (2) PEFT's native `merge_and_unload()`. It compares model responses before/after training and after each merge method to verify correctness.

**Significance:** Serves as a baseline comparison test for HuggingFace's standard QLoRA workflow, allowing developers to benchmark Unsloth's optimizations against vanilla PEFT implementations and ensure merge compatibility.
