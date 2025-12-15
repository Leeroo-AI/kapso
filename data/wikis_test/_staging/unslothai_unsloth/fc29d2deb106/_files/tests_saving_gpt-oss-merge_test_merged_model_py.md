# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests loading and inference on merged GPT-OSS model

**Mechanism:** Loads previously merged 16-bit GPT-OSS model from disk in 4-bit mode, runs inference with reasoning_effort parameter, streams output, then cleans up artifacts

**Significance:** Validates that merged GPT-OSS models can be loaded and used for inference, testing the end-to-end save-load-inference pipeline for the GPT-OSS model family

## Relationships

**Depends on:** <!-- What files in this repo does it import? -->

**Used by:** <!-- What files import this? -->
