# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** GPT model merge validation that verifies a previously merged GPT-OSS model can be loaded and used for inference with reasoning capabilities.

**Mechanism:** Loads a 16-bit merged GPT-OSS model from disk (created by train_and_merge.py) in 4-bit quantization, runs inference with a mathematical problem using the reasoning_effort parameter, streams the output, then cleans up the merged model directory and compilation cache.

**Significance:** Validates the end-to-end workflow for GPT-OSS models specifically, ensuring merged models can be reloaded and perform inference correctly with reasoning features. Part of a two-script test suite where train_and_merge.py produces the artifact and this script validates it.
