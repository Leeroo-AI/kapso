# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** GPT training and merge flow that demonstrates the complete pipeline of fine-tuning GPT-OSS-20B and merging LoRA weights back to 16-bit.

**Mechanism:** Loads GPT-OSS-20B in 4-bit with Mxfp4 quantization, applies rank-8 LoRA to attention and MLP layers, trains for 10 steps on 50 examples from Multilingual-Thinking dataset, then saves the merged 16-bit model to disk using save_pretrained_merged(). Cleans up training artifacts but leaves the merged model for test_merged_model.py to validate.

**Significance:** Tests Unsloth's support for the GPT-OSS architecture (20B parameter model) and validates the Mxfp4 quantization format works correctly through the train-merge-save pipeline. Critical for ensuring Unsloth supports newer model architectures beyond standard LLaMA/Mistral.
