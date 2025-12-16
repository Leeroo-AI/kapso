# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Complete workflow test for training and merging the GPT-OSS-20B model with Unsloth.

**Mechanism:**
- Loads "unsloth/gpt-oss-20b" in 4-bit quantization (mxfp4 format)
- Loads first 50 examples from "HuggingFaceH4/Multilingual-Thinking" dataset
- Applies LoRA (rank 8) to attention and MLP layers
- Fine-tunes for 10 steps using Unsloth gradient checkpointing
- Saves merged 16-bit model to "./gpt-oss-finetuned-merged" using `save_pretrained_merged()`
- Cleans up training outputs and compilation cache

**Significance:** End-to-end test for the complete GPT-OSS fine-tuning pipeline. Critical for validating Unsloth's support for large reasoning models (20B parameters) including mxfp4 quantization, efficient training, and proper model merging for deployment.
