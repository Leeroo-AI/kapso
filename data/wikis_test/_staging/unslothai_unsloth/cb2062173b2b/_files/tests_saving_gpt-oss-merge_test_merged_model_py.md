# File: `tests/saving/gpt-oss-merge/test_merged_model.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 60 |
| Functions | `safe_remove_directory` |
| Imports | gc, os, shutil, torch, transformers, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Tests inference on a previously merged and saved GPT-OSS-20B model using reasoning-capable inference.

**Mechanism:**
- Loads a pre-merged 16-bit model from "./gpt-oss-finetuned-merged" in 4-bit quantization
- Runs inference with chat template using `reasoning_effort="low"` parameter (new GPT-OSS feature)
- Uses TextStreamer for streaming generation output
- Tests model on math problem: "Solve x^5 + 3x^4 - 10 = 3"
- Cleans up model directory and compiled cache after testing

**Significance:** Integration test for GPT-OSS model deployment workflow. Validates that merged models can be reloaded and used for inference with reasoning capabilities. Tests the complete end-to-end workflow from merged model to production inference.
