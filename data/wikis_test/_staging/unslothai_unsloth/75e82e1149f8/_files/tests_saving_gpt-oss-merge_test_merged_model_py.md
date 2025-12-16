# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates GPT-OSS merged model functionality

**Mechanism:** Tests loading and inference with merged GPT models using the gpt-oss-merge path, ensuring merged models work correctly

**Significance:** Ensures compatibility with GPT-OSS model merging workflow and validates that merged models can be loaded and used for inference
