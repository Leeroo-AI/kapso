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

**Purpose:** Tests saving merged models after GRPO training

**Mechanism:** Performs two-stage training on Llama-3.2-3B: first SFT on LIMO dataset, then GRPO on GSM8K with custom reward functions for format compliance and answer correctness, evaluates on AIME dataset, saves merged model, then loads in 4/8/16-bit to compare performance

**Significance:** Validates Unsloth supports advanced GRPO (Group Relative Policy Optimization) training and can properly merge models after complex multi-stage training pipelines, critical for reasoning model development

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
