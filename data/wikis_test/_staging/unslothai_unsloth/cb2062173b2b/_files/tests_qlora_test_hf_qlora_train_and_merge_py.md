# File: `tests/qlora/test_hf_qlora_train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 159 |
| Imports | copy, datasets, itertools, pathlib, sys, tests, torch, trl |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests QLoRA training and merging using pure HuggingFace/PEFT implementation without Unsloth-specific optimizations.

**Mechanism:**
- Loads Llama-3.2-1B-Instruct in 4-bit quantization using standard HuggingFace tools
- Applies LoRA adapters via PEFT config with rank 64 on all linear layers
- Trains on synthetic dataset for 100 steps using TRL's SFTTrainer
- Validates responses before and after training
- Tests two merge strategies: custom `convert_lora_to_linear()` and PEFT's `merge_and_unload()`
- Compares model outputs after each merge method to ensure correctness

**Significance:** Baseline test for validating Unsloth's QLoRA implementation against standard HuggingFace approach. Ensures compatibility and correctness by comparing custom merge operations with PEFT's built-in merge functionality.
