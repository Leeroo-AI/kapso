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

**Purpose:** Comprehensive test of GRPO (Generative Reward-based Policy Optimization) training pipeline with merged model saving, including multi-stage training (SFT then GRPO) and merged model evaluation at different quantization levels.

**Mechanism:** Uses multiprocessing to isolate stages: trains Llama-3.2-3B on LIMO dataset via SFT, applies GRPO training on GSM8K with custom reward functions for format matching and answer correctness, saves merged model with save_pretrained_merged, then evaluates the merged model loaded in 16-bit, 8-bit, and 4-bit configurations on AIME benchmarks.

**Significance:** Critical end-to-end test validating the complete GRPO training workflow with merged model persistence and cross-quantization compatibility, ensuring models trained with advanced techniques can be properly saved and reloaded at different precision levels.
