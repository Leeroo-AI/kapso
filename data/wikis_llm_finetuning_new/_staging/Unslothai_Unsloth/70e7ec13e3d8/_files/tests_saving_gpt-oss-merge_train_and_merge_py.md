# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** Explored

**Purpose:** Training script that fine-tunes the GPT-OSS 20B model using QLoRA and saves a merged 16-bit version for inference testing.

**Mechanism:** Loads `unsloth/gpt-oss-20b` in 4-bit mode with max sequence length 1024, applies LoRA (rank 8, alpha 16) to all projection layers with Unsloth gradient checkpointing. Uses HuggingFaceH4/Multilingual-Thinking dataset (first 50 samples) with chat template formatting. Trains for 10 steps with SFTTrainer, then calls `save_pretrained_merged()` to export the merged model to `./gpt-oss-finetuned-merged`. Includes cleanup of training outputs and cache directories.

**Significance:** Tests Unsloth's ability to fine-tune and merge the GPT-OSS 20B model, which is a larger reasoning-focused model. Validates the complete training-to-merge workflow for this specific architecture, paired with test_merged_model.py for inference validation.
