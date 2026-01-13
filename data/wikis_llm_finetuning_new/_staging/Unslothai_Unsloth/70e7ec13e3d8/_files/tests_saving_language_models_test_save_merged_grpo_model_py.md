# File: `tests/saving/language_models/test_save_merged_grpo_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 825 |
| Functions | `evaluate_merged_model`, `training_run` |
| Imports | gc, multiprocessing, pathlib, sys, tests, torch, unsloth |
| Executable | Yes (`__main__`) |

## Understanding

**Status:** Explored

**Purpose:** Tests an end-to-end GRPO (Group Relative Policy Optimization) training pipeline including SFT pre-training, GRPO fine-tuning, model merging, and evaluation on AIME benchmarks.

**Mechanism:** The test implements a two-stage training pipeline on Llama-3.2-3B-Instruct: (1) SFT fine-tuning on the LIMO dataset for math reasoning, followed by (2) GRPO training on GSM8K with multiple reward functions (format matching, approximate format matching, answer correctness, and numerical verification). The model uses fast_inference mode with vLLM. After training, it saves the merged model using save_pretrained_merged() with merged_16bit format. The test then evaluates the merged model loaded at different precisions (4-bit, 8-bit, 16-bit) using AIME benchmark evaluation via separate subprocesses to manage GPU memory. Results are compared using compare_aime_results().

**Significance:** Validates that complex multi-stage RL training workflows produce models that can be correctly merged and saved. This is critical for GRPO and similar reinforcement learning fine-tuning approaches where the training process is more complex than standard SFT.
