# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Validates inference quality of merged models from the GPT-OSS (Open Source Software) training workflow by loading saved merged models and testing their generation capabilities.

**Mechanism:** Loads a previously merged and saved GPT model using Transformers, runs inference with sample prompts, and verifies the model generates coherent outputs, with cleanup utilities to manage temporary directories.

**Significance:** Ensures merged models from Unsloth training can be successfully saved, reloaded, and used for inference with standard HuggingFace tools, validating the complete save/load cycle for production deployment.
