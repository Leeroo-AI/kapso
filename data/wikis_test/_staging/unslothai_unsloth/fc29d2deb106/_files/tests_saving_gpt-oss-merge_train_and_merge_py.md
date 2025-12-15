# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests GPT-OSS model training and merging workflow

**Mechanism:** Loads 4-bit GPT-OSS-20B model with Mxfp4 quantization, applies LoRA adapters, trains for 10 steps on Multilingual-Thinking dataset, then merges and saves as 16-bit model using save_pretrained_merged

**Significance:** Validates Unsloth can fine-tune and merge the GPT-OSS model family with Mxfp4 quantization, demonstrating support for reasoning models with special quantization formats

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
