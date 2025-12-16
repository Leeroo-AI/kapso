# File: `tests/saving/vision_models/test_push_to_hub_merged.py`

**Category:** test

| Property | Value |
|----------|-------|
| Lines | 273 |
| Functions | `format_data` |
| Imports | datasets, os, pathlib, sys, tests, torch, trl, unsloth |

## Understanding

**Status:** ✅ Explored

**Purpose:** Integration test for complete vision model fine-tuning, merging, and Hugging Face Hub upload workflow using smaller Qwen2-VL-2B model.

**Mechanism:** Follows similar pipeline to test_index_file_sharded_model.py but uses smaller Qwen2-VL-2B-Instruct model: loads French OCR dataset, formats with OAI message structure including PIL images, fine-tunes with LoRA (r=16, alpha=32, 4-bit quantization) for 10 steps, saves adapter locally, uploads merged model to HF_USER's Hub repository using push_to_hub_merged with HF_TOKEN, performs two-stage validation (upload success and download test), generates validation report showing stage-by-stage success status, and cleans up all temporary artifacts including checkpoints and compiled cache.

**Significance:** Validates end-to-end Hub integration for vision models. Simpler than sharded model test (no index file check) because 2B model is small enough for single file upload. Critical for testing basic Hub push/pull functionality, ensuring merged vision models maintain quality after upload, and validating authentication flow with HF tokens.
