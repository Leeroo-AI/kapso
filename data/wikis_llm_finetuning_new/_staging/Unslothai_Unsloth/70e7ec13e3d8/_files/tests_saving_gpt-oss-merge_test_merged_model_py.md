# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** Explored

**Purpose:** Inference test script that loads a previously merged GPT-OSS model and verifies it can generate responses correctly.

**Mechanism:** Loads a merged 16-bit model from `./gpt-oss-finetuned-merged` using FastLanguageModel with 4-bit quantization. Constructs a math problem prompt ("Solve x^5 + 3x^4 - 10 = 3"), applies chat template with `reasoning_effort="low"`, generates responses using TextStreamer for streaming output. Includes `safe_remove_directory()` helper for cleanup of model files and compiled cache after testing.

**Significance:** Part of a two-script test workflow (with train_and_merge.py) that validates the complete GPT-OSS model fine-tuning and merge pipeline. Tests that merged models can be reloaded and perform inference with Unsloth's reasoning effort feature.
