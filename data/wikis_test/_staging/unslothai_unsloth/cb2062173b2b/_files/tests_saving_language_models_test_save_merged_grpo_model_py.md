# File: `tests/saving/language_models/test_save_merged_grpo_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 825 |
| Functions | `evaluate_merged_model`, `training_run` |
| Imports | gc, multiprocessing, pathlib, sys, tests, torch, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** ✅ Explored

**Purpose:** Comprehensive integration test for GRPO (Group Relative Policy Optimization) training pipeline with model merging, validating that merged GRPO models can be saved and loaded in different quantization formats while maintaining performance on AIME reasoning tasks.

**Mechanism:** Implements a complete two-stage training pipeline: (1) Supervised fine-tuning (SFT) on LIMO dataset with Llama-3.2-3B-Instruct using LoRA, (2) GRPO reinforcement learning fine-tuning on GSM8K dataset with custom reward functions for format compliance and answer correctness. Defines extensive helper functions for dataset preparation, answer evaluation, and format checking. After training, saves the model using save_pretrained_merged with merged_16bit method. Then loads and evaluates the merged model in 16-bit, 8-bit, and 4-bit modes in separate subprocesses using AIME benchmarks. Uses multiprocessing Queue for inter-process result communication and compares performance across all model variants.

**Significance:** Critical validation test for advanced RL training workflows combined with model merging. Demonstrates that complex multi-stage training (SFT + GRPO) preserves model quality through the merge-save-reload cycle. Tests reasoning capabilities on mathematical problems, ensuring the merged model maintains the benefits of both training stages. Essential for users doing reward-based fine-tuning with Unsloth.
