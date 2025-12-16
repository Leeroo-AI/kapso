# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests training and merging workflow for large GPT-OSS model

**Mechanism:** Performs 4-bit LoRA training then merges to 16-bit model for deployment

**Significance:** Validates complete workflow for large model fine-tuning and deployment
