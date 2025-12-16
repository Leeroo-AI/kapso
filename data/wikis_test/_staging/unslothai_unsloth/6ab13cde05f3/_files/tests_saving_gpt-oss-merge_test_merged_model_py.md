# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Documented

**Purpose:** Tests inference on merged GPT-OSS model

**Mechanism:** Loads previously merged model and runs inference to validate merge correctness

**Significance:** Validates that merged models work correctly for inference in production
