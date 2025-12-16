# File: `tests/saving/gpt-oss-merge/train_and_merge.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 102 |
| Functions | `safe_remove_directory`, `formatting_prompts_func` |
| Imports | datasets, gc, os, shutil, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests GPT-OSS training and merge workflow

**Mechanism:** Implements full training pipeline for GPT models, applies LoRA adapters, and merges them using GPT-OSS compatible format

**Significance:** Validates end-to-end training and merging workflow for GPT models with OSS compatibility
