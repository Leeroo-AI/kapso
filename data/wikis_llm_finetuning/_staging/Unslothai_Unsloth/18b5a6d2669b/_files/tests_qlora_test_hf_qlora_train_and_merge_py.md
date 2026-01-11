# File: `tests/qlora/test_hf_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 159 |
| Imports | copy, datasets, itertools, pathlib, sys, tests, torch, trl |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** HF QLoRA baseline test that validates the standard HuggingFace PEFT QLoRA training and merging workflow as a reference implementation.

**Mechanism:** Loads LLaMA 3.2-1B in 4-bit quantization, applies QLoRA adapters using standard HuggingFace PEFT, trains for 100 steps on a synthetic dataset, then tests two merge approaches: (1) custom convert_lora_to_linear function and (2) PEFT's merge_and_unload() method. Compares model responses before training, after training, and after both merge methods to verify correctness.

**Significance:** Serves as the ground truth baseline for comparing Unsloth's QLoRA implementation against standard HuggingFace PEFT behavior. Critical for regression testing to ensure Unsloth maintains compatibility with expected HF PEFT semantics.
